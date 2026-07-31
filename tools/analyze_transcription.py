"""Transcription analysis: plausibility metrics and audio self-consistency.

No ground-truth annotations exist for the fixtures in `tests/`, so nothing here is
an accuracy measurement. Every metric is either

* a *plausibility* check -- a pattern that indicates a transcription error (engine
  chatter, out-of-range pitch, string collision, more than six simultaneous voices), or
* a *self-consistency* score against features derived from the source audio
  (onset agreement, chroma agreement).

Both kinds are meaningful only **relatively**, for ranking configurations against
each other and for locating suspicious material. Neither can tell you the true note
list, so neither yields real precision/recall -- that needs annotated data such as
GuitarSet, which also carries the per-string labels needed to score `tab_engine`.

Reference features are computed from the *original* audio, never from the
preset-specific preprocessed audio: the reference has to be identical across
presets or the scores are not comparable.

Usage:
    .\\venv\\Scripts\\python.exe tools\\analyze_transcription.py
    .\\venv\\Scripts\\python.exe tools\\analyze_transcription.py --fixtures Fingerpick_mono_44-16.wav
    .\\venv\\Scripts\\python.exe tools\\analyze_transcription.py --engine-presets baseline hpss
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import pandas as pd
import pretty_midi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from basic_pitch_engine import BasicPitchEngine  # noqa: E402
from midi_gen import generate_midi  # noqa: E402
from post_process import (  # noqa: E402
    clean_notes,
    merge_notes,
    quantize_notes,
    ticks_to_seconds,
)
from processor import load_audio, preprocess_pipeline  # noqa: E402
from tab_engine import TabMapper  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests"
OUT_DIR = REPO_ROOT / "analysis"
CACHE_DIR = OUT_DIR / "cache"
MIDI_DIR = OUT_DIR / "midi"

# The fixtures were recorded at 100 BPM -- the same tempo tests/test_regression.py uses.
DEFAULT_BPM = 100

TARGET_SR = 22050
HOP_LENGTH = 512

# Standard tuning, open low E (E2) to the 24th fret of the high E (E6) -- anything
# outside this cannot be an acoustic guitar note.
GUITAR_MIN_PITCH = 40
GUITAR_MAX_PITCH = 88

# Notes within this window count as one musical onset (a strummed chord is one event).
CHORD_WINDOW = 0.015
# Onset match tolerance. 50 ms is the MIREX convention for onset detection.
ONSET_TOLERANCE = 0.05
# A note this short is more likely a detection artefact than a plucked string.
SHORT_NOTE_SECONDS = 0.05
# Notes this close together belong to one playing event: a strummed chord is spread
# over more than a chord frame as the pick crosses the strings.
EVENT_WINDOW = 0.08
# How much weaker than its fundamental an octave note must be to look like a harmonic
# artefact rather than a voiced octave.
WEAK_OCTAVE_RATIO = 0.5

# Preprocessing / engine settings. Changing any of these requires re-running
# inference, which is the expensive part, so results are cached on disk.
ENGINE_PRESETS = {
    "baseline": {},
    "gate_off": {"noise_threshold": 0.0},
    "hpss": {"use_hpss": True},
    "freq_bounds": {"min_freq": 80.0, "max_freq": 1400.0},
    "hpss_freq_bounds": {"use_hpss": True, "min_freq": 80.0, "max_freq": 1400.0},
}

# Post-processing settings. These operate on cached engine output, so sweeping them
# costs nothing but metric computation.
POST_PRESETS = {
    "default": {},
    "no_merge": {"merge": False},
    "quantize_1_16": {"quantize": "1/16"},
}


# --------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------

def transcribe_cached(wav_path, engine_params, use_cache=True):
    """Runs preprocessing + inference, caching raw engine output keyed by parameters."""
    preprocess_keys = {"noise_threshold", "use_hpss"}
    pre_params = {k: v for k, v in engine_params.items() if k in preprocess_keys}
    infer_params = {k: v for k, v in engine_params.items() if k not in preprocess_keys}

    payload = json.dumps(
        {"wav": wav_path.name, "pre": pre_params, "infer": infer_params}, sort_keys=True
    )
    key = hashlib.sha1(payload.encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f"{wav_path.stem}__{key}.json"

    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())

    audio, sr = preprocess_pipeline(str(wav_path), **pre_params)
    engine = BasicPitchEngine()
    notes = engine.transcribe(audio, sr, **infer_params)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(notes))
    return notes


def post_process(notes, bpm, merge=True, quantize=None):
    """Mirrors the CLI's post-processing order, which is load-bearing."""
    notes = [dict(n) for n in notes]
    notes = clean_notes(notes)
    if merge:
        notes = merge_notes(notes)
    if quantize:
        notes = quantize_notes(notes, bpm, grid_resolution=quantize)
        notes = ticks_to_seconds(notes, bpm)
    return notes


# --------------------------------------------------------------------------------------
# plausibility metrics
# --------------------------------------------------------------------------------------

def group_events(notes):
    """Groups notes into playing events. A strum is spread over more than a chord frame."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: n["start"])
    events = [[ordered[0]]]
    for note in ordered[1:]:
        if note["start"] - events[-1][-1]["start"] < EVENT_WINDOW:
            events[-1].append(note)
        else:
            events.append([note])
    return events


def weak_octave_rate(notes):
    """Fraction of notes that sit an octave above a *much stronger* note in their event.

    Counting octave pairs outright does not work: guitar voicings contain octaves by
    construction -- an open C major is E2-C3-E3-G3-C4-E4, three octave pairs in six
    notes -- so a plain octave count flags correct chords as errors. It measured 45% on
    the strummed fixture, whose events resolve to ordinary C/F/G voicings with the two
    members at comparable velocity (median ratio 0.84) and simultaneous onsets.

    Requiring the upper note to be markedly weaker keeps the case that a real voicing
    does not produce: a harmonic reported as its own note. Still a lead rather than
    proof -- an intentionally light octave doubling looks the same.
    """
    if not notes:
        return 0.0
    flagged = 0
    for event in group_events(notes):
        by_pitch = {n["pitch"]: n for n in event}
        for note in event:
            lower = by_pitch.get(note["pitch"] - 12)
            if lower is not None and note["velocity"] < WEAK_OCTAVE_RATIO * lower["velocity"]:
                flagged += 1
    return flagged / len(notes)


def oversized_events(notes):
    """Events with more than six simultaneous notes, which no six-string guitar can play.

    Unlike the octave heuristics this needs no interpretation: at least one note in such
    an event is spurious.
    """
    return sum(1 for event in group_events(notes) if len(event) > 6)


def chatter_rate(notes, gap=0.05):
    """Fraction of notes that re-onset the same pitch almost immediately.

    A real re-pluck of the same string needs the player to get back to it; a gap this
    small is usually the engine splitting one sustained note in two.
    """
    if not notes:
        return 0.0
    by_pitch = {}
    for note in notes:
        by_pitch.setdefault(note["pitch"], []).append(note)

    flagged = 0
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda n: n["start"])
        for prev, nxt in zip(pitch_notes, pitch_notes[1:], strict=False):
            if nxt["start"] - prev["end"] <= gap:
                flagged += 1
    return flagged / len(notes)


def polyphony_stats(notes):
    """Max and mean simultaneous voices, via a sweep over note on/off events."""
    if not notes:
        return 0, 0.0
    events = []
    for note in notes:
        events.append((note["start"], 1))
        events.append((note["end"], -1))
    events.sort()

    active = 0
    peak = 0
    weighted = 0.0
    span = 0.0
    prev_time = events[0][0]
    for time, delta in events:
        if active > 0 and time > prev_time:
            weighted += active * (time - prev_time)
            span += time - prev_time
        active += delta
        peak = max(peak, active)
        prev_time = time
    return peak, (weighted / span if span > 0 else 0.0)


def tab_metrics(notes):
    """String-collision count and fretting-hand plausibility for --tab output.

    A collision means two overlapping notes were assigned the same string, which is
    physically impossible; `tab_engine` guards against it, so any hit is a real defect.
    """
    assigned = [n for n in notes if n.get("string", 0) > 0]
    if not assigned:
        return {"tab_collisions": np.nan, "tab_max_fret_span": np.nan, "tab_median_shift": np.nan}

    collisions = 0
    by_string = {}
    for note in assigned:
        by_string.setdefault(note["string"], []).append(note)
    for string_notes in by_string.values():
        string_notes.sort(key=lambda n: n["start"])
        for prev, nxt in zip(string_notes, string_notes[1:], strict=False):
            if nxt["start"] < prev["end"] - 1e-6:
                collisions += 1

    # Group into chord frames the same way tab_engine does, then measure the stretch
    # within each frame and the hand travel between frames.
    ordered = sorted(assigned, key=lambda n: n["start"])
    frames = [[ordered[0]]]
    for note in ordered[1:]:
        if abs(note["start"] - frames[-1][0]["start"]) < CHORD_WINDOW:
            frames[-1].append(note)
        else:
            frames.append([note])

    spans = []
    positions = []
    for frame in frames:
        frets = [n["fret"] for n in frame if n.get("fret", 0) > 0]
        if frets:
            spans.append(max(frets) - min(frets))
            positions.append(sum(frets) / len(frets))

    shifts = [abs(b - a) for a, b in zip(positions, positions[1:], strict=False)]
    return {
        "tab_collisions": collisions,
        "tab_max_fret_span": max(spans) if spans else np.nan,
        "tab_median_shift": float(np.median(shifts)) if shifts else np.nan,
    }


# --------------------------------------------------------------------------------------
# self-consistency against the source audio
# --------------------------------------------------------------------------------------

def reference_features(wav_path):
    """Onset times and chroma for the untouched audio, shared by every preset."""
    audio, sr = load_audio(str(wav_path), target_sr=TARGET_SR)
    onsets = librosa.onset.onset_detect(
        y=audio, sr=sr, hop_length=HOP_LENGTH, units="time", backtrack=True
    )
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP_LENGTH)
    return {"sr": sr, "duration": len(audio) / sr, "onsets": onsets, "chroma": chroma}


def onset_agreement(notes, reference):
    """F-measure of MIDI onsets against librosa's onset detector.

    The detector is itself an estimate, not ground truth: it finds *events*, and cannot
    say how many notes an event contains. Chords are therefore collapsed to one onset
    before scoring, and the result is a timing/event-count sanity check rather than an
    accuracy figure.
    """
    if not notes or len(reference["onsets"]) == 0:
        return {"onset_f": np.nan, "onset_precision": np.nan, "onset_recall": np.nan}

    starts = sorted(n["start"] for n in notes)
    collapsed = [starts[0]]
    for start in starts[1:]:
        if start - collapsed[-1] > CHORD_WINDOW:
            collapsed.append(start)

    f_measure, precision, recall = mir_eval.onset.f_measure(
        np.asarray(reference["onsets"]), np.asarray(collapsed), window=ONSET_TOLERANCE
    )
    return {"onset_f": f_measure, "onset_precision": precision, "onset_recall": recall}


def chroma_agreement(midi_path, reference):
    """Mean per-frame cosine similarity between MIDI and audio chroma.

    Compares pitch-class content without needing note-level annotations. Absolute
    values run low by construction -- audio chroma carries harmonic leakage that a
    symbolic piano roll has none of -- so only differences between presets mean anything.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    fs = reference["sr"] / HOP_LENGTH
    midi_chroma = pm.get_chroma(fs=fs)
    audio_chroma = reference["chroma"]

    frames = min(midi_chroma.shape[1], audio_chroma.shape[1])
    if frames == 0:
        return np.nan
    a = midi_chroma[:, :frames]
    b = audio_chroma[:, :frames]

    # Only score frames where both sides have energy; silence is trivially similar and
    # would inflate the mean on sparse passages.
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    active = (a_norm > 1e-8) & (b_norm > 1e-8)
    if not active.any():
        return np.nan

    sims = (a[:, active] * b[:, active]).sum(axis=0) / (a_norm[active] * b_norm[active])
    return float(sims.mean())


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

def analyze(wav_path, engine_name, post_name, bpm, reference, use_cache):
    engine_params = ENGINE_PRESETS[engine_name]
    post_params = POST_PRESETS[post_name]

    raw_notes = transcribe_cached(wav_path, engine_params, use_cache=use_cache)
    post_notes = post_process(raw_notes, bpm, **post_params)

    # Every metric is computed on this list, which is also what the MIDI is written from.
    # String assignment is not a free annotation pass: resolving same-string overlaps
    # shortens notes and can withdraw an assignment, so measuring the pre-tab list would
    # describe a different file than the one on disk. This means the harness reports the
    # `--tab` pipeline, not the CLI default.
    notes = TabMapper().assign_strings([dict(n) for n in post_notes])

    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    midi_path = MIDI_DIR / f"{wav_path.stem}__{engine_name}__{post_name}.mid"
    generate_midi(notes, bpm, str(midi_path))

    peak_poly, mean_poly = polyphony_stats(notes)
    durations = [n["end"] - n["start"] for n in notes] or [0.0]
    pitches = [n["pitch"] for n in notes] or [0]

    row = {
        "fixture": wav_path.stem,
        "engine_preset": engine_name,
        "post_preset": post_name,
        "raw_notes": len(raw_notes),
        "notes": len(notes),
        "weak_octave": weak_octave_rate(notes),
        "oversized_events": oversized_events(notes),
        "chatter": chatter_rate(notes),
        "short_notes": sum(d < SHORT_NOTE_SECONDS for d in durations) / len(durations),
        "out_of_range": sum(
            p < GUITAR_MIN_PITCH or p > GUITAR_MAX_PITCH for p in pitches
        ) / len(pitches),
        "median_duration": float(np.median(durations)),
        # Overlap resolution shortens notes, so the floor is tracked explicitly: a note
        # cut to a few milliseconds is a click, and averages hide it.
        "min_duration": float(min(durations)),
        "unassigned": sum(1 for n in notes if n.get("string", 0) == 0),
        "peak_polyphony": peak_poly,
        "mean_polyphony": mean_poly,
    }
    row.update(onset_agreement(notes, reference))
    row["chroma_agreement"] = chroma_agreement(midi_path, reference)
    row.update(tab_metrics(notes))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", nargs="*", default=None, help="WAV file names in tests/.")
    parser.add_argument("--engine-presets", nargs="*", default=list(ENGINE_PRESETS))
    parser.add_argument("--post-presets", nargs="*", default=list(POST_PRESETS))
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM)
    parser.add_argument("--no-cache", action="store_true", help="Force re-running inference.")
    parser.add_argument("--out", default=str(OUT_DIR / "metrics.csv"))
    args = parser.parse_args()

    if args.fixtures:
        wavs = [FIXTURE_DIR / name for name in args.fixtures]
    else:
        wavs = sorted(FIXTURE_DIR.glob("*.wav"))
    missing = [w for w in wavs if not w.exists()]
    if missing:
        parser.error(f"missing fixtures: {', '.join(m.name for m in missing)}")

    rows = []
    for wav in wavs:
        print(f"[*] {wav.name}: computing reference features")
        reference = reference_features(wav)
        for engine_name in args.engine_presets:
            for post_name in args.post_presets:
                print(f"    - {engine_name} / {post_name}")
                rows.append(
                    analyze(wav, engine_name, post_name, args.bpm, reference, not args.no_cache)
                )

    frame = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(f"\n[+] {len(frame)} rows written to {out_path}\n")

    summary_cols = [
        "notes", "weak_octave", "oversized_events", "chatter", "min_duration", "unassigned",
        "onset_f", "onset_precision", "onset_recall", "chroma_agreement", "tab_collisions",
    ]
    print("Mean across fixtures, by engine preset:")
    print(frame.groupby("engine_preset")[summary_cols].mean().round(3))
    print("\nMean across fixtures, by post preset:")
    print(frame.groupby("post_preset")[summary_cols].mean().round(3))


if __name__ == "__main__":
    main()
