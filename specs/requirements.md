# Requirements Specification - Polyphonic Guitar-to-MIDI Enhancements

## 1. Executive Summary & Key Learnings

Based on analysis and empirical evaluation of acoustic guitar transcription using `basic-pitch` and `pretty_midi`, several key limitations in the baseline system were identified:

### Learnings from Baseline Evaluation:
1. **Phantom Notes from Transients:** Guitar pick strikes and finger plucks generate strong percussive transients. The baseline ML model misidentifies these high-frequency attack spikes as short, unintended high/low notes.
2. **Sub-Bass & Hyper-Treble Noise:** Unconstrained frequency detection allows rumble (<80 Hz) and upper harmonics (>1,400 Hz) to produce false MIDI notes outside standard guitar playing range.
3. **Unnatural Dynamic Response:** Linear velocity mapping (`raw_val * 127`) results in quiet notes sounding abruptly silent while moderate strums cap out at maximum velocity (127), failing to match human auditory loudness perception (dB scale).
4. **Static Pitch Limitations:** Bends, slides, hammer-ons, and vibrato—core characteristics of acoustic guitar expression—are flattened to static pitch boxes because pitch bend predictions from `basic-pitch` were ignored.
5. **Greedy Fingering Inefficiencies:** Step-by-step greedy string assignment in `tab_engine.py` causes physically unnatural fret jumps across sequential notes and allows string collision during multi-note chord onsets.

---

## 2. Functional Requirements

### FR-01: Frequency Bounding & Model Hyperparameter Tuning
- Restrict `basic-pitch` inference frequency range between **E2 (~80 Hz)** and **E6 (~1400 Hz)** to eliminate sub-bass and ultra-treble noise.
- Expose model thresholds (`--onset-threshold`, `--frame-threshold`, `--min-note-length`) via the CLI to enable tuning for fingerstyle vs. heavy strumming.

### FR-02: Harmonic/Percussive Source Separation (HPSS) & Audio Cleaning
- Provide optional `--hpss` flag using `librosa.effects.hpss` to separate percussive pick clicks from sustained harmonic resonance prior to pitch detection.
- Provide smooth spectral noise suppression to protect long note decay tails from being prematurely truncated by the hard noise gate.

### FR-03: Logarithmic Perceptual Velocity Curve
- Apply a non-linear logarithmic transformation curve to raw model confidence/amplitude values:
  $$\text{Velocity}_{\text{MIDI}} = \text{round}\left(127 \times \frac{\ln(1 + k \cdot a)}{\ln(1 + k)}\right)$$
  where $a \in [0, 1]$ is normalized amplitude and $k$ is the curvature parameter (default $k=5.0$).

### FR-04: Pitch Bend & Expressive MIDI Output
- Extract frame-level pitch bend values from `basic-pitch`.
- Write `pretty_midi.PitchBend` events into the generated MIDI tracks for expressive vibrato, slides, and bends.

### FR-05: MPE / Multi-Channel MIDI Separation
- When `--tab` mode is active, map guitar strings 1–6 to separate **MIDI Channels 1–6** within the output MIDI file, allowing DAWs to apply per-string pitch bends and expression automation (CC 11).

### FR-06: Viterbi Global Fingering & Polyphonic Chord Guard
- Replace the greedy string allocator in `TabMapper` with a **Viterbi dynamic programming solver** that minimizes total physical hand movement energy (fret distance + string skip penalties) across phrases.
- Enforce strict polyphonic collision constraints ensuring no two simultaneous notes in a chord share the same physical string.

---

## 3. Non-Functional Requirements & Performance Budgets

- **Transcription Latency:** Overall execution time must remain under **1.5x real-time audio length** on standard CPU hardware.
- **Accuracy Target:** Maintain or exceed **92% F-measure** (Precision & Recall) on the `GuitarSet` benchmark dataset.
- **Memory Overhead:** Memory consumption must not exceed 2 GB for audio files up to 10 minutes in length.
- **Backwards Compatibility:** All new CLI arguments must be optional with sensible defaults so existing invocation scripts continue working seamlessly.

---

## 4. Documentation & Developer Standards

- **CLI Help & Usage Guides:** Update `README.md` with complete documentation for all new flags (`--hpss`, `--onset-threshold`, `--velocity-curve`, `--pitch-bend`, `--tuning`).
- **Inline Documentation:** Every new module and function must include complete Google-style Python docstrings and type annotations.
- **Project Overview & Architecture Docs:** Maintain structured architecture and component documentation in `docs/PROJECT_OVERVIEW.md`.

---

## 5. Acceptance Criteria

| Requirement ID | Acceptance Criterion | Verification Method |
| :--- | :--- | :--- |
| **AC-01 (FR-01)** | Notes below 80 Hz or above 1400 Hz are excluded from MIDI output. | Unit test with out-of-bounds sine waves |
| **AC-02 (FR-02)** | High-transient pick strikes do not produce phantom notes when `--hpss` is enabled. | Integration test on fingerpick sample |
| **AC-03 (FR-03)** | Velocity distribution matches logarithmic loudness curve without clipping quiet tails. | Velocity distribution benchmark |
| **AC-04 (FR-04)** | MIDI file contains valid `PitchBend` events when pitch bend option is enabled. | MIDI inspection via `pretty_midi` |
| **AC-05 (FR-05)** | MIDI track in `--tab` mode separates string events into MIDI channels 1 through 6. | Inspection of channel headers in MIDI |
| **AC-06 (FR-06)** | Consecutive notes produce ergonomic fret transitions; simultaneous chord notes never share a string. | Unit tests in `test_tab_engine.py` |
