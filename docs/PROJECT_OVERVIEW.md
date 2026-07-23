# Project Overview & Architecture Guide: audio2midi

## Executive Summary

**audio2midi** is a high-fidelity, polyphonic acoustic guitar-to-MIDI transcription system written in Python. It converts acoustic guitar `.wav` recordings into Standard MIDI Files (`.mid`) with optional tablature string assignment (Strings 1–6), expressive pitch bend writing, logarithmic dynamic loudness curves, and transient click filtering.

---

## Technical Stack & Architecture

- **Language & Runtime:** Python 3.10.x
- **Audio Processing:** `librosa` (HPSS, spectral loading), `scipy.signal` (Butterworth 80Hz HPF), `numpy` (<2.0.0)
- **Deep Learning Transcription Engine:** Spotify `basic-pitch`, `tensorflow` (2.7+)
- **Post-Processing:** NumPy vectorized quantization, sweep-line note merging, logarithmic velocity transformation
- **Tablature Solver:** Viterbi dynamic programming solver (`TabMapper`) with polyphonic string collision guard
- **MIDI Output Engine:** `pretty_midi` with multi-channel MPE track separation and pitch bend event writing

---

## Directory Structure

```
audio2midi/
├── src/
│   ├── audio2midi.py         # Main CLI entry point & workflow orchestrator
│   ├── processor.py          # Audio loading, HPF, HPSS, noise gate, peak normalization
│   ├── basic_pitch_engine.py # Spotify Basic Pitch ML inference & pitch bend extraction
│   ├── post_process.py       # Note cleaning, sweep-line merging, log velocity, quantization
│   ├── tab_engine.py         # Viterbi solver for string/fret fingering & chord collision guard
│   ├── midi_gen.py           # Standard MIDI File generator with MPE multi-channel tracks
│   └── engine_base.py        # Abstract base class for transcription engines
├── specs/
│   ├── requirements.md       # Requirements specification & learnings
│   ├── design.md             # System design & component architecture
│   ├── test-plan.md          # Test matrix & coverage targets
│   └── tasks.md              # Phased implementation plan
├── tests/
│   ├── test_processor.py     # Unit tests for audio pre-processing & HPSS
│   ├── test_basic_pitch.py   # Unit tests for Basic Pitch frequency bounds & pitch bends
│   ├── test_post_process.py  # Unit tests for logarithmic velocity & partial quantization
│   ├── test_tab_engine.py    # Unit tests for Viterbi solver & chord collision guard
│   ├── test_midi_gen.py      # Unit tests for MPE MIDI output & pitch bend writing
│   ├── test_audio2midi.py    # Integration tests for core pipeline
│   ├── test_cli.py           # CLI invocation tests
│   └── test_regression.py   # End-to-end transcription accuracy regression tests
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup file
└── README.md                 # User guide & documentation
```

---

## Execution Pipeline

1. **Audio Pre-processing ([`src/processor.py`](file:///E:/sw_ws/repo1/audio2midi/src/processor.py)):**
   Audio loaded $\rightarrow$ Peak normalized to -1 dB $\rightarrow$ 80Hz Butterworth High-Pass Filtered $\rightarrow$ Optional HPSS (`librosa.effects.hpss`) $\rightarrow$ Noise Gated.
2. **Inference ([`src/basic_pitch_engine.py`](file:///E:/sw_ws/repo1/audio2midi/src/basic_pitch_engine.py)):**
   Basic Pitch DNN model detects pitch onsets, offsets, amplitudes, and frame-level pitch bends, filtered by frequency bounds (80–1400 Hz).
3. **Post-Processing ([`src/post_process.py`](file:///E:/sw_ws/repo1/audio2midi/src/post_process.py)):**
   Notes filtered by duration & velocity $\rightarrow$ Logarithmic dynamic velocity curve applied $\rightarrow$ Sweep-line note merging $\rightarrow$ Optional grid quantization.
4. **Tablature Solver ([`src/tab_engine.py`](file:///E:/sw_ws/repo1/audio2midi/src/tab_engine.py)):**
   Viterbi solver calculates global minimum energy neck path (Strings 1–6) while enforcing strict non-collision rules for chord notes.
5. **MIDI Generation ([`src/midi_gen.py`](file:///E:/sw_ws/repo1/audio2midi/src/midi_gen.py)):**
   `pretty_midi` writes notes and pitch bend events into multi-channel MPE tracks.
