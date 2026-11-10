"""Ground-truth evaluation against GuitarSet.

Unlike `analyze_transcription.py`, this measures accuracy. GuitarSet was recorded with a
hexaphonic pickup, so every note's pitch, timing **and string** are known, which makes it
the only way to score `tab_engine` -- a plain audio file carries no signal saying which
string was played.

Scores follow the MIREX conventions in `mir_eval.transcription`: a 50 ms onset window and
a 50 cent pitch tolerance. Reference pitches are continuous f0 estimates, so the cent
tolerance also absorbs the rounding to integer MIDI that our pipeline performs.

Three families are reported:

* `onset` -- precision/recall/F1 on onset and pitch, ignoring note ends. The standard
  headline number for transcription.
* `offset` -- the same with note ends required to match within 20% of the reference
  duration. Sensitive to sustain handling.
* `string` -- of the notes that matched the reference, the fraction placed on the correct
  string. Measured on the `--tab` output, and the offset scores are reported for that
  output too, since overlap resolution shortens notes.

Setup (~664MB, both from https://zenodo.org/records/3371780):
    data/guitarset/annotation/*.jams        <- annotation.zip
    data/guitarset/audio_mono-mic/*.wav     <- audio_mono-mic.zip

Tracks are split by player (see SPLITS): tune on `dev`, confirm on `test`.

Usage:
    .\\venv\\Scripts\\python.exe tools\\evaluate_guitarset.py --limit 20
    .\\venv\\Scripts\\python.exe tools\\evaluate_guitarset.py --limit 24 --engine-presets baseline hpss
    .\\venv\\Scripts\\python.exe tools\\evaluate_guitarset.py --split test --limit 0
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import basic_pitch_engine as bpe  # noqa: E402
from audio2midi import GUITAR_MAX_FREQ, GUITAR_MIN_FREQ  # noqa: E402
from engine_cache import preprocessed_audio, transcribe_cached  # noqa: E402
from post_process import clean_notes, merge_notes  # noqa: E402
from retrigger import absorb_ringing_retriggers  # noqa: E402
from tab_engine import TabMapper  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "guitarset"
ANNOTATION_DIR = DATA_DIR / "annotation"
AUDIO_DIR = DATA_DIR / "audio_mono-mic"
OUT_DIR = REPO_ROOT / "analysis"
CACHE_DIR = OUT_DIR / "guitarset_cache"

ONSET_TOLERANCE = 0.05
PITCH_TOLERANCE = 50.0  # cents
OFFSET_RATIO = 0.2

# GuitarSet numbers its six note_midi annotations 0 (low E) to 5 (high e). tab_engine
# numbers strings 6 (low E) to 1 (high e), so the two run in opposite directions.
def guitarset_source_to_string(source):
    return 6 - int(source)


ENGINE_PRESETS = {
    "baseline": {},  # the CLI defaults (BasicPitchEngine's DEFAULT_* values)
    # The README's opt-in values for clean recordings (basic_pitch_engine.CLEAN_RECORDING_*).
    "tuned_clean": {
        "onset_threshold": bpe.CLEAN_RECORDING_ONSET_THRESHOLD,
        "frame_threshold": bpe.CLEAN_RECORDING_FRAME_THRESHOLD,
        "min_note_len": bpe.CLEAN_RECORDING_MIN_NOTE_LENGTH_MS,
    },
    "hpss": {"use_hpss": True},
    "freq_bounds": {"min_freq": GUITAR_MIN_FREQ, "max_freq": GUITAR_MAX_FREQ},  # = --freq-bounds
}

# Split by player, so test results also show whether tuning carries over to players it never
# saw. Tune on `dev` only; run `test` to confirm a decision, never to make one -- once a
# setting is chosen by looking at test scores, those scores stop being an honest estimate.
# Caveat: two decisions predate this split (merging off, the per-note position cost) and were
# first measured on 20 tracks spread over all six players, 6 of them from players 04-05. Those
# 6 are a small part of the 120 test tracks (results without them differ by under 0.01), but
# the test set is not perfectly unseen for those two. Later decisions (the open-string
# penalty, re-checking max_frets) were made on `dev` alone.
SPLITS = {
    "dev": ("00", "01", "02", "03"),
    "test": ("04", "05"),
}

# Named for what the CLI does: merging is opt-in via --merge.
POST_PRESETS = {
    "cli_default": {"merge": False, "absorb": True},
    "keep_retriggers": {"merge": False, "absorb": False},
    "merge": {"merge": True, "absorb": True},
}


def load_reference(jams_path):
    """Reads note events and their true string from a GuitarSet JAMS file."""
    doc = json.loads(jams_path.read_text())
    notes = []
    for annotation in doc["annotations"]:
        if annotation["namespace"] != "note_midi":
            continue
        string = guitarset_source_to_string(annotation["annotation_metadata"]["data_source"])
        for obs in annotation["data"]:
            if obs["duration"] <= 0:
                continue
            notes.append({
                "pitch": float(obs["value"]),
                "start": float(obs["time"]),
                "end": float(obs["time"]) + float(obs["duration"]),
                "string": string,
            })
    notes.sort(key=lambda n: n["start"])
    return notes


def to_arrays(notes):
    """mir_eval wants intervals in seconds and pitches in Hz."""
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n["start"], n["end"]] for n in notes], dtype=float)
    pitches = librosa.midi_to_hz(np.array([n["pitch"] for n in notes], dtype=float))
    return intervals, pitches


def score(ref_notes, est_notes, offset_ratio):
    ref_intervals, ref_pitches = to_arrays(ref_notes)
    est_intervals, est_pitches = to_arrays(est_notes)
    if len(ref_notes) == 0 or len(est_notes) == 0:
        return np.nan, np.nan, np.nan
    precision, recall, f_measure, _overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=ONSET_TOLERANCE,
            pitch_tolerance=PITCH_TOLERANCE,
            offset_ratio=offset_ratio,
        )
    )
    return precision, recall, f_measure


def string_accuracy(ref_notes, est_notes):
    """Fraction of correctly matched notes that also landed on the right string.

    Only matched notes can be scored: a spurious or missed note has no counterpart whose
    string to compare against, and folding those in would conflate two different errors.
    """
    ref_intervals, ref_pitches = to_arrays(ref_notes)
    est_intervals, est_pitches = to_arrays(est_notes)
    if len(ref_notes) == 0 or len(est_notes) == 0:
        return np.nan, 0, np.nan

    matches = mir_eval.transcription.match_notes(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        pitch_tolerance=PITCH_TOLERANCE,
        offset_ratio=None,
    )
    if not matches:
        return np.nan, 0, np.nan

    correct = 0
    unassigned = 0
    for ref_idx, est_idx in matches:
        est_string = est_notes[est_idx].get("string", 0)
        if est_string == 0:
            unassigned += 1
            continue
        if est_string == ref_notes[ref_idx]["string"]:
            correct += 1

    scorable = len(matches) - unassigned
    accuracy = (correct / scorable) if scorable else np.nan
    return accuracy, len(matches), unassigned / len(matches)


def naive_string_assignment(notes):
    """Per-note placement nearest the preferred fret, with no solver at all.

    A baseline for `string_accuracy`: on its own that number cannot say whether the
    Viterbi solver is doing well or badly, because many notes are playable on only one
    or two strings and would be assigned correctly by luck. This ignores context,
    continuity and collisions. It is a harder bar than it looks: on held-out players (test
    split, 120 tracks) the solver trailed it (0.610 vs 0.644) until open strings stopped
    being free in its cost; it now leads, 0.664 overall and 0.715 vs 0.678 on chords, but
    is only level on single-note lines (0.612 vs 0.610).
    """
    mapper = TabMapper()
    out = []
    for note in notes:
        placements = mapper.get_possible_placements(note["pitch"])
        copy = dict(note)
        if placements:
            string, fret = min(placements, key=lambda p: abs(p[1] - mapper.PREFERRED_FRET))
            copy["string"], copy["fret"] = string, fret
        else:
            copy["string"], copy["fret"] = 0, -1
        out.append(copy)
    return out


def evaluate(wav_path, jams_path, engine_name, post_name, use_cache):
    ref_notes = load_reference(jams_path)
    raw = transcribe_cached(wav_path, ENGINE_PRESETS[engine_name], CACHE_DIR, use_cache=use_cache)

    notes = clean_notes([dict(n) for n in raw])
    if POST_PRESETS[post_name]["absorb"]:
        notes = absorb_ringing_retriggers(
            notes, *preprocessed_audio(wav_path, ENGINE_PRESETS[engine_name]))
    if POST_PRESETS[post_name]["merge"]:
        notes = merge_notes(notes)

    # Overlap resolution shortens notes, so the tab pipeline is scored separately rather
    # than assumed to be a free annotation pass over the same note list.
    tabbed = TabMapper().assign_strings([dict(n) for n in notes])

    onset_p, onset_r, onset_f = score(ref_notes, notes, offset_ratio=None)
    off_p, off_r, off_f = score(ref_notes, notes, offset_ratio=OFFSET_RATIO)
    tab_off_p, tab_off_r, tab_off_f = score(ref_notes, tabbed, offset_ratio=OFFSET_RATIO)
    str_acc, matched, unassigned_rate = string_accuracy(ref_notes, tabbed)
    naive_acc, _naive_matched, _naive_unassigned = string_accuracy(
        ref_notes, naive_string_assignment(notes)
    )

    return {
        "track": wav_path.stem.replace("_mic", ""),
        "player": player_of(wav_path),
        "style": "comp" if "_comp" in wav_path.stem else "solo",
        "engine_preset": engine_name,
        "post_preset": post_name,
        "ref_notes": len(ref_notes),
        "est_notes": len(notes),
        "onset_precision": onset_p,
        "onset_recall": onset_r,
        "onset_f": onset_f,
        "offset_precision": off_p,
        "offset_recall": off_r,
        "offset_f": off_f,
        "tab_offset_f": tab_off_f,
        "string_accuracy": str_acc,
        "string_accuracy_naive": naive_acc,
        "matched_notes": matched,
        "unassigned_rate": unassigned_rate,
    }


def player_of(wav_path):
    """GuitarSet file names start with the player's two-digit id, e.g. `03_BN1-129-Eb_comp`."""
    return wav_path.stem.split("_", 1)[0]


def select_tracks(limit, split="dev"):
    """Deterministic, style-balanced spread over one split, rather than the first N files.

    Sorted order alternates `_comp` and `_solo` for each excerpt, so striding the whole
    list by an even step silently yields one style only. The two are very different
    material -- chords versus single-note lines -- so each is sampled separately.
    """
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        raise SystemExit(
            f"no audio in {AUDIO_DIR}. Extract audio_mono-mic.zip from "
            "https://zenodo.org/records/3371780 into that directory."
        )
    if split != "all":
        wavs = [w for w in wavs if player_of(w) in SPLITS[split]]
    if limit is None or limit >= len(wavs):
        return wavs

    chosen = []
    for style in ("comp", "solo"):
        pool = [w for w in wavs if f"_{style}" in w.stem]
        take = limit // 2 + (limit % 2 if style == "comp" else 0)
        stride = len(pool) / take if take else 1
        chosen.extend(pool[int(i * stride)] for i in range(take))
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="dev",
                        help="Players to draw from: dev (00-03, for tuning), test (04-05, for "
                             "confirming a decision), or all. Default: dev.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Tracks to evaluate from the split, style-balanced (0 = every "
                             "track in it). Default: 20.")
    parser.add_argument("--engine-presets", nargs="*", default=["baseline"])
    parser.add_argument("--post-presets", nargs="*", default=list(POST_PRESETS))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default=None,
                        help="CSV path. Default: analysis/guitarset_<split>.csv.")
    args = parser.parse_args()
    args.out = args.out or str(OUT_DIR / f"guitarset_{args.split}.csv")

    wavs = select_tracks(args.limit or None, split=args.split)
    rows = []
    for i, wav in enumerate(wavs, 1):
        jams = ANNOTATION_DIR / f"{wav.stem.replace('_mic', '')}.jams"
        if not jams.exists():
            print(f"[!] {wav.name}: no annotation at {jams.name}, skipped")
            continue
        print(f"[*] ({i}/{len(wavs)}) {wav.stem}")
        for engine_name in args.engine_presets:
            for post_name in args.post_presets:
                rows.append(evaluate(wav, jams, engine_name, post_name, not args.no_cache))

    frame = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    print(f"\n[+] {args.split} split: {len(frame)} rows over {frame['track'].nunique()} tracks "
          f"-> {out_path}\n")

    cols = [
        "ref_notes", "est_notes", "onset_precision", "onset_recall", "onset_f",
        "offset_f", "tab_offset_f", "string_accuracy", "string_accuracy_naive",
        "unassigned_rate",
    ]
    print("By engine and post preset:")
    print(frame.groupby(["engine_preset", "post_preset"])[cols].mean().round(3))
    print("\nBy style:")
    print(frame.groupby(["post_preset", "style"])[cols].mean().round(3))


if __name__ == "__main__":
    main()
