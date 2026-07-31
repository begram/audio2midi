# Test Plan Specification - Polyphonic Guitar-to-MIDI Enhancements

## 1. Test Matrix

The test matrix maps each requirement from `requirements.md` to specific unit, integration, and benchmark test cases.

| ID | Requirement | Test Case Description | Test Level | Verification File |
| :--- | :--- | :--- | :--- | :--- |
| **TR-01** | FR-01: Frequency Bounding | Pass note events outside the 80–1400 Hz range; verify pitch events outside bounds are filtered out. | Unit | `tests/test_basic_pitch.py` |
| **TR-02** | FR-01: Threshold Tuning | Pass `--onset-threshold 0.7` (and `--frame-threshold`, `--min-note-length`) to the CLI; verify the engine receives them. | Integration | `tests/test_cli_wiring.py` |
| **TR-03** | FR-02: HPSS Filter | Process audio with simulated pick attack transients; verify transient spikes are separated. | Unit | `tests/test_processor.py` |
| **TR-04** | FR-03: Log Velocity Mapping | Pass linear amplitudes `[0.1, 0.5, 0.9]`; verify mapped MIDI velocities follow logarithmic curve. | Unit | `tests/test_post_process.py` |
| **TR-05** | FR-04: Pitch Bend Events | Provide a per-note bend contour from the model mock; verify scaled `PitchBend` objects are written to the MIDI track, clamped and de-duplicated. | Integration | `tests/test_basic_pitch.py`, `tests/test_midi_gen.py` |
| **TR-06** | FR-05: Per-String Track Output | Enable `--tab`; verify output MIDI splits string events onto distinct tracks. | Integration | `tests/test_midi_gen.py` |
| **TR-07** | FR-06: Viterbi String Solver | Provide sequence of notes; verify the solver outputs an ergonomic fingering path and that position preference does not decay after the first frame. | Unit | `tests/test_tab_engine.py` |
| **TR-08** | FR-06: Chord Collision Guard | Provide simultaneous multi-note chord onset; verify all notes are assigned to unique physical strings. | Unit | `tests/test_tab_engine.py` |
| **TR-09** | Non-Functional Accuracy | Run accuracy benchmark against `GuitarSet` clips; verify F-measure $\ge 0.92$. | Benchmark | **Not implemented.** No corpus or scoring exists; see TR-12. |
| **TR-10** | FR-04: Bend Ownership & Timing | Verify a bend is never shared between simultaneous notes, survives merging, and stays inside its note's span after quantization. | Unit | `tests/test_basic_pitch.py`, `tests/test_post_process.py` |
| **TR-11** | Robustness: Argument Validation | Reject `--bpm 0`, inverted `--min-freq`/`--max-freq`, and out-of-range strengths; report undecodable audio without a traceback. | Integration | `tests/test_cli.py` |
| **TR-12** | Regression Tripwire | Transcribe each reference recording at 100 BPM; assert note count falls in its expected band and output tempo matches. Marked `slow`. | End-to-End | `tests/test_regression.py` |
| **TR-13** | FR-02: Noise Gate Behaviour | Verify quiet passages are attenuated, above-threshold passages pass without sample-wise zeroing, and the leading attack is not faded by envelope smoothing. | Unit | `tests/test_audio2midi.py` |
| **TR-14** | FR-02: High-Pass Response | Verify sub-cutoff content is attenuated, gain is monotonic across the cutoff, `y[0] == x[0]`, and the vectorized IIR matches the direct recurrence sample-for-sample. | Unit | `tests/test_processor.py` |
| **TR-15** | Input Format Support | Load 16-bit/44.1 kHz, 24-bit/48 kHz and 24-bit/96 kHz stereo WAVs; verify each is down-mixed to mono and resampled to `target_sr`. | Unit | `tests/test_processor.py` |
| **TR-16** | FR-06: Collision Fallback | For frames with no collision-free assignment, verify the number of colliding notes is the physical minimum (six strings used before any doubling) and that a single-placement pitch is never displaced. | Unit | `tests/test_tab_engine.py` |
| **TR-17** | CLI Option Wiring | Verify `--freq-bounds` / `--min-freq` / `--max-freq` resolution, `--pitch-bend`, `--no-merge`, `--quantize`, `--tab`, `--velocity-curve`, `--bpm` and `--instrument` all reach the correct pipeline stage. | Integration | `tests/test_cli_wiring.py` |
| **TR-18** | Resource Hygiene | Verify the scratch WAV handed to the model is deleted after both a successful run and a failed one. | Unit | `tests/test_basic_pitch.py` |

---

## 2. Risk-Based Prioritization

1. **Critical Risk (Priority 1):**
   - **Viterbi String Assignment & Chord Collision Guard (TR-07, TR-08):** String collision in chords renders multi-track MIDI unplayable in DAWs.
   - **Logarithmic Velocity Transformation (TR-04):** Directly impacts musical dynamics and velocity response.
2. **High Risk (Priority 2):**
   - **Pitch Bend MIDI Export (TR-05):** Ensures pitch articulation (slides/vibrato) is preserved without corrupting timing.
   - **Frequency Bounding & HPSS Cleaning (TR-01, TR-03):** Crucial for suppressing phantom note artifacts.
3. **Medium Risk (Priority 3):**
   - **MPE Multi-Channel Assignment (TR-06):** Important for DAW compatibility.
4. **Low Risk (Priority 4):**
   - **CLI Threshold Arguments (TR-02):** UI/UX convenience.

---

## 3. Test Data & Environment Needs

- **Synthetic Audio Generation:** Python scripts using `scipy.signal` to synthesize pure sine waves, harmonic chords, and pitch-swiped sine waves for exact ground-truth testing.
- **Acoustic Test Clips:** WAV samples in `tests/` (`Fingerpick_mono_44-16.wav`, `Fingerpick_stereo_48-24.wav`, `plektrumpick_mono_44-16.wav`, `plektrumstrum_mono_44-16.wav`). These are **fixtures and must stay tracked in git** — `.gitignore` excludes `*.wav` globally and re-includes `tests/*.wav`, otherwise the end-to-end tests cannot run on a fresh clone. Every `.mid` in `tests/` is generated output and stays ignored.
- **Benchmark Corpus:** `GuitarSet` dataset audio and annotated MIDI ground truth files. **Not present in this repository** (see TR-09).
- **Test Runner:** `pytest`. Tests resolve the interpreter via `sys.executable` and fixture paths relative to `tests/` (see `tests/conftest.py`), so they are not tied to a specific machine or working directory.
  - `pytest -m "not slow"` — fast suite, no full model inference over real audio.
  - `pytest` — everything, including the TR-12 regressions.

---

## 4. Coverage Goals per Component

> **Status:** measured. `pytest-cov` is a dev dependency, settings live in
> `.coveragerc` (branch mode, `fail_under = 95`), and CI runs
> `pytest -m "not slow" --cov`. Current totals: **99.8%** statement / branch
> coverage over `src/`, with zero missed statements.
>
> The one remaining partial branch is the `os.path.exists(tmp_path)` guard in
> `basic_pitch_engine.transcribe`'s `finally` block, which never evaluates false
> in practice. It is deliberately left in as a defensive guard rather than
> chased with a contrived test.
>
> Note that coverage only counts in-process execution: `test_cli.py` drives the
> CLI as a subprocess and contributes nothing to these figures, which is why
> `test_cli_wiring.py` exercises the same orchestration in-process.

| Component | Target Line Coverage | Target Branch Coverage |
| :--- | :--- | :--- |
| [`src/processor.py`](file:///E:/sw_ws/repo1/audio2midi/src/processor.py) | $\ge 95\%$ | $\ge 90\%$ |
| [`src/post_process.py`](file:///E:/sw_ws/repo1/audio2midi/src/post_process.py) | $\ge 95\%$ | $\ge 95\%$ |
| [`src/tab_engine.py`](file:///E:/sw_ws/repo1/audio2midi/src/tab_engine.py) | $\ge 95\%$ | $\ge 90\%$ |
| [`src/midi_gen.py`](file:///E:/sw_ws/repo1/audio2midi/src/midi_gen.py) | $\ge 90\%$ | $\ge 85\%$ |
| [`src/basic_pitch_engine.py`](file:///E:/sw_ws/repo1/audio2midi/src/basic_pitch_engine.py) | $\ge 85\%$ | $\ge 80\%$ |

---

## 5. Mocking & Isolation Strategy

To ensure fast unit test execution without loading heavy TensorFlow neural networks on every test run:
- **Direct patching:** `tests/test_basic_pitch.py` patches `basic_pitch_engine.predict` with `unittest.mock` and predefined note event tuples, including per-note bend contours. (Earlier revisions named a `MockTranscriptionEngine` class; no such class exists — the `TranscriptionEngine` ABC in `src/engine_base.py` is the only abstraction, and it has one implementation.)
- **Pure-function tests:** `post_process.py`, `tab_engine.py`, and `midi_gen.py` operate on plain note dicts, so their tests construct input directly and need no engine mock at all.
- **In-process CLI harness:** `tests/test_cli_wiring.py` drives `audio2midi.main` with `click.testing.CliRunner` while patching `preprocess_pipeline`, `BasicPitchEngine` and `generate_midi`. This verifies orchestration — option resolution, stage ordering, error paths — with no audio decoding or inference, and unlike the subprocess tests it is visible to coverage.
- **Integration Tests:** Only the `slow`-marked `test_regression.py` runs actual model inference against real WAV audio. `test_cli.py` spawns the CLI but fails fast on argument validation or undecodable input, so it does not reach inference.
