# Implementation Plan - Polyphonic Guitar-to-MIDI Converter

## Task Breakdown

### Phase 1: Environment & Project Scaffolding
- **Task 1.1:** Initialize project structure and `setup.py`/`requirements.txt`.
  - **Estimate:** 30m
  - **Completion Criteria:** Project structure matches spec; `pip install -r requirements.txt` succeeds.
  - **Verification:** Run `python --version` and verify all libs are importable.
- **Task 1.2:** Implement basic CLI skeleton in `audio2midi.py`.
  - **Estimate:** 1h
  - **Completion Criteria:** `python audio2midi.py --help` shows all arguments (BPM, quantization, etc.).
  - **Verification:** Test CLI with invalid/missing arguments (TR-09).

### Phase 2: Audio Processing & Engines
- **Task 2.1:** Implement `processor.py` for 16/24-bit loading, normalization, and high-pass filtering.
  - **Estimate:** 2h
  - **Completion Criteria:** Audio is correctly pre-processed and resampled.
  - **Verification:** Unit tests for loading and filtering (TR-01, TR-03, TR-04).
- **Task 2.2:** Integrate `basic-pitch` as the primary transcription engine.
  - **Estimate:** 3h
  - **Completion Criteria:** Raw note events (pitch, onset, offset, amplitude) are returned from audio.
  - **Verification:** Transcribe a single sine wave; verify correct MIDI pitch.

### Phase 3: Post-Processing & MIDI Generation
- **Task 3.1:** Implement quantization and note cleaning in `post_process.py`.
  - **Estimate:** 2h
  - **Completion Criteria:** Notes are filtered by duration/velocity and optionally snapped to grid.
  - **Verification:** Unit tests for quantization and cleaning logic (TR-05, TR-06).
- **Task 3.2:** Implement `midi_gen.py` using `pretty_midi`.
  - **Estimate:** 1h
  - **Completion Criteria:** Valid .mid file is saved with correct BPM metadata.
  - **Verification:** Verify MIDI header and track content using `mido` (TR-02).

### Phase 4: Integration & Validation
- **Task 4.1:** Wire all components together in `audio2midi.py`.
  - **Estimate:** 2h
  - **Completion Criteria:** Full E2E flow from .wav to .mid.
  - **Verification:** Run E2E test with a short guitar clip (TR-07).
- **Task 4.2:** Implement Accuracy Benchmark script.
  - **Estimate:** 2h
  - **Completion Criteria:** Script outputs Precision, Recall, and F-measure.
  - **Verification:** Run against a subset of GuitarSet (TR-08).

## Summary of Tasks vs. Requirements

| Requirement | Task(s) | Verification Test |
|-------------|---------|-------------------|
| 16/24-bit Support | 2.1 | TR-01 |
| BPM Metadata | 3.2 | TR-02 |
| High-Pass Filter | 2.1 | TR-03 |
| Normalization | 2.1 | TR-04 |
| Note Cleaning | 3.1 | TR-05 |
| Quantization | 3.1 | TR-06 |
| Polyphony | 2.2, 4.1 | TR-07 |
| 90% Accuracy | 4.2 | TR-08 |
| CLI Interface | 1.2 | TR-09 |

## Dependencies
- **Task 2.2** depends on **Task 2.1** (pre-processed audio).
- **Task 3.1** depends on **Task 2.2** (note events).
- **Task 4.1** depends on all Phase 2 & 3 tasks.
