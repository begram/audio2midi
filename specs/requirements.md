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
| **AC-06 (FR-06)** | Consecutive notes produce ergonomic fret transitions; simultaneous chord notes never share a string **whenever a collision-free assignment exists**. Where none exists the number of colliding notes must be the physical minimum. | Unit tests in `test_tab_engine.py` |

---

## 6. Implementation Status

This section records where the implementation stands against the requirements above.
It is descriptive: unmet requirements are recorded as gaps rather than rewritten away.

| Requirement | Status | Notes |
| :--- | :--- | :--- |
| **FR-01** Frequency bounding | Met | `--freq-bounds`, `--min-freq`, `--max-freq`, `--onset-threshold`, `--frame-threshold`, and `--min-note-length` are all exposed. Note that `--min-note-length` (engine-level, milliseconds) and `--min-duration` (post-filter, seconds) are two independent gates. |
| **FR-02** HPSS | Met | `--hpss` wraps `librosa.effects.hpss`. |
| **FR-02** Smooth noise suppression | Met, renamed | Implemented as an RMS-envelope `noise_gate`, not the `apply_spectral_denoise` entry point named in `design.md`. The earlier sample-wise gate clipped the waveform at every zero crossing; the envelope gate protects decay tails as this requirement intended. |
| **FR-03** Log velocity curve | Met | `--velocity-curve`; output clamped to $[1, 127]$. |
| **FR-04** Pitch bends | Met | Sourced per-note from the model's own bend contour, re-timed on quantization, carried across merges, and recentred at each note's end. |
| **FR-05** Per-string channel separation | Met with caveat | One track per string, which becomes one channel per string on write. `pretty_midi` allocates the literal channel numbers itself, so "Channels 1–6" in AC-05 is not something the code pins directly. This is not full MPE (no per-note channel rotation, no CC 11 automation). |
| **FR-06** Viterbi solver & chord guard | Met, with a stated limit | Global DP solver with a per-frame position emission cost, travel transition cost, collision guard, and a candidate cap per frame. Collision freedom is guaranteed only when it is physically achievable — see the note below. |
| **NFR** Latency < 1.5x real-time | Unverified | No timing harness exists. The dominant cost is Basic Pitch inference. |
| **NFR** 92% F-measure on GuitarSet | **Not implemented** | No corpus, scoring script, or ground-truth comparison in the repo. `test_regression.py` asserts note-count bands, which is a regression tripwire rather than an accuracy measurement. |
| **NFR** Memory < 2 GB for 10-minute audio | Unverified | No memory profiling harness exists. |
| **NFR** Backwards compatibility | Met | All added options have defaults; existing invocations are unaffected. |
| **Docs** README flag coverage | Met, one exception | All implemented flags are documented. The `--tuning` flag named in §4 of this document **does not exist**; `TabMapper.TUNING` is hardcoded to standard tuning, so alternate tunings are unsupported. |
| **Docs** Architecture docs | Met | `docs/PROJECT_OVERVIEW.md`, `specs/design.md`, and `specs/test-plan.md` reflect the current implementation. |
| **Docs** Google-style docstrings & type annotations | **Not met** | The codebase uses plain prose docstrings and has no type annotations. This would be a mechanical but repo-wide change. |
| **Test** Coverage measurement | Met | `pytest-cov` + `.coveragerc` (branch mode, `fail_under = 95`), enforced in CI. Currently 99.8% over `src/`. |

### Note on FR-06 collision freedom

Two situations admit no collision-free assignment, and the solver cannot invent
strings to fix either:

1. **More than six simultaneous notes.** At most six can occupy distinct strings.
2. **Duplicate same-pitch detections where the pitch has one placement.** Low E
   (MIDI 40) is only playable as string 6 / fret 0, so two simultaneous
   detections of it must share that string. `merge_notes` normally removes such
   duplicates; they reach the solver when `--no-merge` is combined with `--tab`.

In both cases the fallback maximizes the number of notes on distinct strings via
exact bipartite matching, then places any remainder on its best placement. A
residual collision is therefore a signal that the *detection* is not physically
playable as transcribed, not that the solver failed.
