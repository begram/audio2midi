# Project Overview & Architecture Guide: audio2midi

## Executive Summary

**audio2midi** is a high-fidelity, polyphonic acoustic guitar-to-MIDI transcription system written in Python. It converts acoustic guitar `.wav` recordings into Standard MIDI Files (`.mid`) with optional tablature string assignment (Strings 1–6), expressive pitch bend writing, logarithmic dynamic loudness curves, and transient click filtering.

---

## Technical Stack & Architecture

- **Language & Runtime:** Python 3.10.x
- **Audio Processing:** `librosa` (HPSS, spectral loading, RMS envelope), `scipy.signal` (vectorized one-pole 80 Hz high-pass IIR via `lfilter`), `numpy` (<2.0.0)
- **Deep Learning Transcription Engine:** Spotify `basic-pitch` (ONNX model on `onnxruntime`; no TensorFlow)
- **Post-Processing:** NumPy vectorized quantization, opt-in windowed note merging, logarithmic velocity transformation
- **Tablature Solver:** Viterbi dynamic programming solver (`TabMapper`) with per-frame position emission cost and polyphonic string collision guard
- **MIDI Output Engine:** `pretty_midi`, one track per assigned string, with per-note pitch bend contours
- **Lint, Coverage & CI:** `ruff` (`ruff.toml`), `pytest-cov` (`.coveragerc`, branch mode, `fail_under = 95`), GitHub Actions (`.github/workflows/ci.yml`)

---

## Directory Structure

```
audio2midi/
├── src/
│   ├── audio2midi.py         # Main CLI entry point & workflow orchestrator
│   ├── processor.py          # Audio loading, HPF, HPSS, RMS-envelope noise gate, peak normalization
│   ├── basic_pitch_engine.py # Spotify Basic Pitch ML inference & per-note pitch bend extraction
│   ├── post_process.py       # Note cleaning, windowed merging, log velocity, quantization
│   ├── tab_engine.py         # Viterbi solver for string/fret fingering & chord collision guard
│   ├── midi_gen.py           # Standard MIDI File generator with per-string tracks
│   └── engine_base.py        # Abstract base class for transcription engines
├── specs/
│   ├── requirements.md       # Requirements specification, learnings & implementation status
│   ├── design.md             # System design & component architecture
│   ├── test-plan.md          # Test matrix & coverage targets
│   └── tasks.md              # Phased implementation plan
├── docs/
│   └── PROJECT_OVERVIEW.md   # This file: architecture & component guide
├── tools/
│   ├── evaluate_guitarset.py     # Ground-truth accuracy against GuitarSet (manual)
│   ├── analyze_transcription.py  # Plausibility sweep over presets on tests/ fixtures
│   └── engine_cache.py           # Shared, code-aware cache of raw engine output
├── data/guitarset/           # GuitarSet corpus, downloaded separately (gitignored)
├── analysis/                 # Generated tool output & caches (gitignored)
├── tests/
│   ├── conftest.py           # Shared fixtures: interpreter, CLI path, audio fixture resolver
│   ├── test_processor.py     # Unit tests for audio pre-processing & HPSS
│   ├── test_basic_pitch.py   # Unit tests for frequency bounds, per-note bends & velocity floor
│   ├── test_post_process.py  # Unit tests for log velocity, quantization & bend re-timing
│   ├── test_tab_engine.py    # Unit tests for Viterbi solver & chord collision guard
│   ├── test_midi_gen.py      # Unit tests for per-string MIDI output & pitch bend writing
│   ├── test_audio2midi.py    # Integration tests for core pipeline & noise gate behaviour
│   ├── test_cli.py           # CLI subprocess invocation & argument validation
│   ├── test_cli_wiring.py    # In-process CLI orchestration, option wiring & full-pipeline invariants
│   ├── test_tools.py         # GuitarSet bookkeeping, track selection & engine-cache invalidation
│   └── test_regression.py    # End-to-end transcription regressions (marked `slow`)
├── .github/workflows/ci.yml  # Lint + fast test suite with coverage gate on push / PR
├── pytest.ini                # Pytest configuration & marker registration
├── ruff.toml                 # Lint configuration
├── .coveragerc               # Coverage settings (branch mode, fail_under = 95)
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Runtime + test/lint toolchain
├── setup.py                  # Package setup (flat `py_modules` under src/)
├── CLAUDE.md                 # Agent-facing project context (source of truth)
├── GEMINI.md                 # Pointer to CLAUDE.md
└── README.md                 # User guide & documentation
```

---

## Execution Pipeline

1. **Audio Pre-processing ([`src/processor.py`](file:///E:/sw_ws/repo1/audio2midi/src/processor.py)):**
   Audio loaded $\rightarrow$ Peak normalized to -1 dB $\rightarrow$ one-pole 80 Hz high-pass filtered $\rightarrow$ Optional HPSS (`librosa.effects.hpss`) $\rightarrow$ RMS-envelope noise gate.
2. **Inference ([`src/basic_pitch_engine.py`](file:///E:/sw_ws/repo1/audio2midi/src/basic_pitch_engine.py)):**
   Basic Pitch DNN model detects pitch onsets, offsets, amplitudes, and a per-note bend contour, optionally filtered by frequency bounds (`--freq-bounds`: 80–1400 Hz, which keeps MIDI 39–88). Velocity is floored at 1 so a near-silent onset never becomes a note-off.
3. **Post-Processing ([`src/post_process.py`](file:///E:/sw_ws/repo1/audio2midi/src/post_process.py)):**
   Notes filtered by duration & velocity $\rightarrow$ Optional logarithmic velocity curve $\rightarrow$ Optional same-pitch merging (`--merge`) $\rightarrow$ Optional grid quantization (with bend re-timing).
4. **Tablature Solver ([`src/tab_engine.py`](file:///E:/sw_ws/repo1/audio2midi/src/tab_engine.py)):**
   Viterbi solver calculates global minimum energy neck path (Strings 1–6) while enforcing strict non-collision rules for chord notes.
5. **MIDI Generation ([`src/midi_gen.py`](file:///E:/sw_ws/repo1/audio2midi/src/midi_gen.py)):**
   `pretty_midi` writes notes onto one track per assigned string, with each note's pitch bend events recentred to 0 at the note's end.

---

## Design Notes

### Why the high-pass filter is not a Butterworth

`high_pass_filter` implements a one-pole RC response, `y[n] = α·(y[n-1] + x[n] - x[n-1])`,
expressed for `scipy.signal.lfilter` as `b = [α, -α]`, `a = [1, -α]` with the initial
condition set so `y[0] == x[0]`. Writing it as a direct Python loop over samples is
~170× slower (1.25 s vs 0.007 s for 30 s of audio at 22.05 kHz) for an identical result.

### Why the noise gate is envelope-based

The gate thresholds a smoothed short-term RMS envelope and applies the result as a
gain curve. Zeroing individual samples below an amplitude threshold clips the waveform
at every zero crossing; the resulting step discontinuities are broadband harmonic
distortion, which Basic Pitch reports as phantom notes — the very artefact `--hpss`
exists to suppress. The gain curve is edge-padded before smoothing so the envelope
does not fade in the first note's attack.

### Why pitch bends are per-note

Basic Pitch returns each note's own bend contour as the fifth element of its note
event (units of 1/3 semitone; `× 4096/3` gives MIDI ticks, and the times are spread
evenly across the note's span). Matching bends to notes by time window instead would
copy one string's bend onto every simultaneously sounding note. Bends are re-timed
when notes are quantized, carried across merges, and recentred at each note's end so
they cannot bleed into the next note sharing that channel. Only one pitch-wheel event is
written per MIDI tick, because `pretty_midi` orders same-tick events by value; the one
kept is the one meant to be in force (the later note's, or the later point in time).

### Note merging complexity

`merge_notes` groups by pitch and merges consecutive same-pitch notes that overlap (or
nearly touch) while another pitch is sounding through them. The concurrency check
binary-searches the start-sorted note list for the relevant time window rather than
scanning every note, so cost is bounded by window density rather than total note count
(20 000 notes in ~0.37 s). Earlier revisions described this as a sweep-line; it is not
one, and the unused event list that hinted at that design has been removed.

Merging is opt-in (`--merge`). Against GuitarSet it lowered onset F1 from 0.742 to 0.646
on the tuning tracks and from 0.775 to 0.693 on held-out players:
the end-to-start gap test also absorbs notes that simply sustained until the same pitch was
played again, so it removes genuine repeats. See the README for the full measurement.

### Per-string tracks vs. MPE

`--tab` places each assigned string on its own `pretty_midi.Instrument`, which becomes
its own track and channel on write. Because a physical string sounds one note at a
time, this gives each voice an independent bend stream. Note that `pretty_midi`
allocates channel numbers itself at write time — the code controls the track/instrument
split, not the literal channel indices.
