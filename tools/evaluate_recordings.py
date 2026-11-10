"""Ground-truth evaluation against your own, hand-corrected recordings.

GuitarSet is the only other ground truth here, and it was recorded under controlled
conditions. Settings that win there (see README, "Clean recordings") lost audible notes on
the less controlled recordings in tests/, but without annotations that could not be
measured -- only inferred. This tool closes that gap: score the pipeline against references
made from recordings like the ones users actually make.

The full user-facing checklist, including how to keep a draft aligned with the audio in a
DAW, is docs/ANNOTATING.md.

Workflow:

1. Write drafts to correct:
       .\\venv\\Scripts\\python.exe tools\\evaluate_recordings.py --write-drafts
   For every tests/*.wav without a reference, this writes tests/references/<name>.draft.mid
   as a starting point: every note found by either the CLI defaults or the clean-recording
   thresholds, so neither setting gets a head start (see draft_notes).
2. Correct each draft **by listening**, in a DAW: delete notes that are not there, add
   missed ones, fix wrong pitches, move onsets. Save the result as
   tests/references/<name>.mid (no ".draft"). Only pitches and onsets matter for the main
   score; note ends matter for the offset score, so fix those too if you can.
3. Score:
       .\\venv\\Scripts\\python.exe tools\\evaluate_recordings.py
   Every recording that has a reference is scored for each engine preset (by default the CLI
   defaults and the clean-recording thresholds), with the same MIREX settings as
   evaluate_guitarset.py (50 ms onset window, 50 cent pitch tolerance).

Caveat: a draft comes from the pipeline itself, so a reference corrected from it can lean
toward the pipeline's choices -- an extra note that is hard to hear may survive, a note both
settings missed may stay missed. The union draft keeps that lean from favouring either
setting, but it cannot remove it. Correct by ear, not by eye. Scores here are only as good as
the references.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pretty_midi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluate_guitarset as eg  # noqa: E402
from engine_cache import preprocessed_audio, transcribe_cached  # noqa: E402
from midi_gen import generate_midi  # noqa: E402
from post_process import clean_notes  # noqa: E402
from retrigger import absorb_ringing_retriggers  # noqa: E402

RECORDINGS_DIR = REPO_ROOT / "tests"
REFERENCES_DIR = RECORDINGS_DIR / "references"
CACHE_DIR = REPO_ROOT / "analysis" / "recordings_cache"
DRAFT_SUFFIX = ".draft.mid"
# The fixtures in tests/ were recorded at 100 BPM (see test_regression.py). The tempo only
# affects how a draft lines up with a DAW's grid; scoring uses seconds throughout.
DRAFT_BPM = 100
# The settings the references are meant to decide between. One list on purpose: drafts
# combine exactly these, and scoring compares exactly these by default -- a draft built from
# a different set would favour some of the settings being compared.
COMPARED_PRESETS = ("baseline", "tuned_clean")


def reference_path(wav_path, references_dir=REFERENCES_DIR):
    return references_dir / f"{wav_path.stem}.mid"


def draft_path(wav_path, references_dir=REFERENCES_DIR):
    return references_dir / f"{wav_path.stem}{DRAFT_SUFFIX}"


def load_reference_midi(midi_path):
    """Reads every note on every track of a reference MIDI file, sorted by onset.

    Drum tracks are skipped: a percussion note's number is a drum, not a pitch.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [
        {"pitch": float(n.pitch), "start": float(n.start), "end": float(n.end)}
        for inst in pm.instruments if not inst.is_drum
        for n in inst.notes if n.end > n.start
    ]
    notes.sort(key=lambda n: n["start"])
    return notes


def pipeline_notes(wav_path, engine_params, use_cache=True):
    """The CLI's default post-processing: clean, then absorb phantom re-onsets (merging is
    off by default)."""
    raw = transcribe_cached(wav_path, engine_params, CACHE_DIR, use_cache=use_cache)
    notes = clean_notes([dict(n) for n in raw])
    return absorb_ringing_retriggers(notes, *preprocessed_audio(wav_path, engine_params))


def draft_notes(wav_path, presets=COMPARED_PRESETS):
    """Every note any of `presets` finds, with duplicates (same pitch, onsets within the 50 ms
    match window) kept once.

    A draft from one setting would hand that setting a head start: whatever mistakes survive
    correction are its own, so the reference would favour it in exactly the comparison the
    references exist to settle. A union gives no setting that advantage; correcting it is
    mostly deleting, which is also the easier edit to do by ear.
    """
    merged = []
    for name in presets:
        for note in pipeline_notes(wav_path, eg.ENGINE_PRESETS[name]):
            if not any(m["pitch"] == note["pitch"]
                       and abs(m["start"] - note["start"]) < eg.ONSET_TOLERANCE
                       for m in merged):
                merged.append(note)
    merged.sort(key=lambda n: n["start"])
    return merged


def write_drafts(wavs, references_dir=REFERENCES_DIR, overwrite=False):
    """Writes a draft per recording that has no reference yet. Returns the paths written."""
    references_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for wav in wavs:
        if reference_path(wav, references_dir).exists():
            continue
        path = draft_path(wav, references_dir)
        if path.exists() and not overwrite:
            continue
        generate_midi(draft_notes(wav), DRAFT_BPM, str(path))
        written.append(path)
    return written


def paired_recordings(wavs, references_dir=REFERENCES_DIR):
    """(wav, reference) pairs for the recordings that have been annotated."""
    return [(w, reference_path(w, references_dir)) for w in wavs
            if reference_path(w, references_dir).exists()]


def evaluate(pairs, presets, use_cache=True):
    rows = []
    for wav, ref_path in pairs:
        ref = load_reference_midi(ref_path)
        for name in presets:
            est = pipeline_notes(wav, eg.ENGINE_PRESETS[name], use_cache=use_cache)
            on_p, on_r, on_f = eg.score(ref, est, offset_ratio=None)
            _, _, off_f = eg.score(ref, est, offset_ratio=eg.OFFSET_RATIO)
            rows.append({
                "recording": wav.stem, "engine_preset": name,
                "ref_notes": len(ref), "est_notes": len(est),
                "onset_precision": on_p, "onset_recall": on_r, "onset_f": on_f,
                "offset_f": off_f,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, default=RECORDINGS_DIR,
                        help="Folder of .wav recordings (default: tests/).")
    parser.add_argument("--references", type=Path, default=REFERENCES_DIR,
                        help="Folder of reference .mid files (default: tests/references/).")
    parser.add_argument("--write-drafts", action="store_true",
                        help="Write <name>.draft.mid for recordings without a reference, then stop.")
    parser.add_argument("--overwrite-drafts", action="store_true",
                        help="With --write-drafts, replace existing drafts.")
    parser.add_argument("--engine-presets", nargs="*", default=list(COMPARED_PRESETS),
                        help="Settings to score (default: the ones the drafts combine).")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / "analysis" / "recordings.csv"))
    args = parser.parse_args()

    wavs = sorted(args.recordings.glob("*.wav"))
    if not wavs:
        parser.error(f"no .wav files in {args.recordings}")

    if args.write_drafts:
        written = write_drafts(wavs, args.references, overwrite=args.overwrite_drafts)
        for path in written:
            print(f"[+] draft: {path}")
        print(f"[*] {len(written)} draft(s) written. Correct each by ear and save it without "
              f"'.draft' to make it a reference.")
        return

    pairs = paired_recordings(wavs, args.references)
    missing = [w.name for w in wavs if not reference_path(w, args.references).exists()]
    if missing:
        print(f"[!] no reference yet for: {', '.join(missing)}")
    if not pairs:
        raise SystemExit("no references to score against; run with --write-drafts first.")

    frame = evaluate(pairs, args.engine_presets, use_cache=not args.no_cache)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    cols = ["ref_notes", "est_notes", "onset_precision", "onset_recall", "onset_f", "offset_f"]
    print(f"\n[+] {len(pairs)} annotated recording(s) -> {out_path}\n")
    print(frame.pivot_table(index="recording", columns="engine_preset", values="onset_f").round(3))
    print("\nMean by preset:")
    print(frame.groupby("engine_preset")[cols].mean().round(3))
    if len(pairs) < 5:
        print(f"\n[!] only {len(pairs)} recording(s): too few to decide defaults on. Aim for 5-10, "
              "covering fingerpicked, strummed and picked playing, including a noisy take.")


if __name__ == "__main__":
    main()
