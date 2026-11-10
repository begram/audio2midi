# Test Plan Specification - Polyphonic Guitar-to-MIDI Enhancements

## 1. Test Matrix

The test matrix maps each requirement from `requirements.md` to specific unit, integration, and benchmark test cases.

| ID | Requirement | Test Case Description | Test Level | Verification File |
| :--- | :--- | :--- | :--- | :--- |
| **TR-01** | FR-01: Frequency Bounding | Pass note events around the 80–1400 Hz bounds; verify exactly MIDI 39–88 survive, matching Basic Pitch's own `round(min)`…`round(max)` exclusive binning. | Unit | `tests/test_basic_pitch.py` |
| **TR-02** | FR-01: Threshold Tuning | Pass `--onset-threshold 0.7` (and `--frame-threshold`, `--min-note-length`) to the CLI; verify the engine receives them. Verify the CLI's defaults are the engine's `DEFAULT_*` constants (one source of truth), and that the README's clean-recording values are accepted and forwarded. | Integration | `tests/test_cli_wiring.py` |
| **TR-03** | FR-02: HPSS Filter | Process audio with simulated pick attack transients; verify transient spikes are separated. | Unit | `tests/test_processor.py` |
| **TR-04** | FR-03: Log Velocity Mapping | Pass linear amplitudes `[0.1, 0.5, 0.9]`; verify mapped MIDI velocities follow logarithmic curve. | Unit | `tests/test_post_process.py` |
| **TR-05** | FR-04: Pitch Bend Events | Provide a per-note bend contour from the model mock; verify scaled `PitchBend` objects are written to the MIDI track, clamped and de-duplicated. | Integration | `tests/test_basic_pitch.py`, `tests/test_midi_gen.py` |
| **TR-06** | FR-05: Per-String Track Output | Enable `--tab`; verify output MIDI splits string events onto distinct tracks. | Integration | `tests/test_midi_gen.py` |
| **TR-07** | FR-06: Viterbi String Solver | Provide sequence of notes; verify the solver outputs an ergonomic fingering path and that position preference does not decay after the first frame. | Unit | `tests/test_tab_engine.py` |
| **TR-08** | FR-06: Chord Collision Guard | Provide simultaneous multi-note chord onset; verify all notes are assigned to unique physical strings. | Unit | `tests/test_tab_engine.py` |
| **TR-09** | Non-Functional Accuracy | Run accuracy benchmark against `GuitarSet` clips; verify F-measure $\ge 0.92$. | Benchmark (manual) | `tools/evaluate_guitarset.py`. Runs on demand, not in CI. Target **not met**: onset F1 0.784 on the held-out test players (04–05, 120 tracks). Also scores `--tab` string accuracy against the hexaphonic labels. |
| **TR-10** | FR-04: Bend Ownership & Timing | Verify a bend is never shared between simultaneous notes, survives merging, and stays inside its note's span after quantization. | Unit | `tests/test_basic_pitch.py`, `tests/test_post_process.py` |
| **TR-11** | Robustness: Argument Validation | Reject `--bpm 0`, inverted `--min-freq`/`--max-freq`, and out-of-range strengths; report undecodable audio without a traceback. | Integration | `tests/test_cli.py` |
| **TR-12** | Regression Tripwire | Transcribe each reference recording at 100 BPM with CLI defaults; assert note count falls in its band (about ±8% around the measured count) and output tempo matches. Marked `slow`. | End-to-End | `tests/test_regression.py` |
| **TR-13** | FR-02: Noise Gate Behaviour | Verify quiet passages are attenuated, above-threshold passages pass without sample-wise zeroing, and the leading attack is not faded by envelope smoothing. | Unit | `tests/test_audio2midi.py` |
| **TR-14** | FR-02: High-Pass Response | Verify sub-cutoff content is attenuated, gain is monotonic across the cutoff, `y[0] == x[0]`, and the vectorized IIR matches the direct recurrence sample-for-sample. | Unit | `tests/test_processor.py` |
| **TR-15** | Input Format Support | Load 16-bit/44.1 kHz, 24-bit/48 kHz and 24-bit/96 kHz stereo WAVs; verify each is down-mixed to mono and resampled to `target_sr`. | Unit | `tests/test_processor.py` |
| **TR-16** | FR-06: Collision Fallback | For frames with no collision-free assignment, verify the number of colliding notes is the physical minimum (six strings used before any doubling) and that a single-placement pitch is never displaced. | Unit | `tests/test_tab_engine.py` |
| **TR-17** | CLI Option Wiring | Verify `--freq-bounds` / `--min-freq` / `--max-freq` resolution, `--pitch-bend`, `--merge` / `--no-merge` (off by default), `--quantize`, `--tab`, `--velocity-curve`, `--velocity-scale`, `--absorb-retriggers` / `--keep-retriggers`, `--bpm` and `--instrument` all reach the correct pipeline stage. | Integration | `tests/test_cli_wiring.py` |
| **TR-18** | Resource Hygiene | Verify the scratch WAV handed to the model is deleted after both a successful run and a failed one. | Unit | `tests/test_basic_pitch.py` |
| **TR-19** | FR-04: Same-Tick Bend Ordering | Verify one pitch-wheel event per tick, the one meant to be in force: a downward first bend on the next note survives the previous note's recentre, an upward last bend at a note's end does not bleed past it, the later of two same-tick bends inside one note wins, a bend timed past its note's end is dropped, and output is independent of input note order. | Unit | `tests/test_midi_gen.py` |
| **TR-20** | FR-06: Compact Chord Fingering | Verify position cost is per note (frets 1 + 9 cost more than 5 + 5) and that an octave shape is fingered at frets 1 and 3, not stretched across seven frets. | Unit | `tests/test_tab_engine.py` |
| **TR-21** | Stage Ordering | Verify the velocity threshold judges raw engine confidence (applied before `--velocity-curve` and `--velocity-scale`), that the scale applies after the curve, that re-onset absorption runs after cleaning and before merging, and that preprocessing, cleaning and quantization options reach their stages. | Integration | `tests/test_cli_wiring.py` |
| **TR-22** | Pipeline Invariants | Run every post-engine stage for real with quantization, `--tab` and bends, then check the written file: no overlapping notes on a string channel, every bend inside one of its channel's notes, wheel at 0 after the last note. | Integration | `tests/test_cli_wiring.py` |
| **TR-23** | Evaluation Bookkeeping | Verify GuitarSet string numbering (`6 − data_source`), that `string_accuracy` excludes unassigned and unmatched notes, style-balanced track selection, that the player splits are disjoint and the default never draws test players, that the engine cache misses when engine source changes, that byte-identical recording references count as one take, and that the recording pipeline absorbs re-onsets as the CLI does. | Unit | `tests/test_tools.py` |
| **TR-24** | FR-06: Open-String Penalty | Verify a note playable open or fretted (E4) takes the fretted placement (string 2, fret 5), that a note with only an open placement (low E) is still placed, and that the penalty is averaged rather than summed over a frame. | Unit | `tests/test_tab_engine.py` |
| **TR-25** | Own-Recording Ground Truth | Verify references are read across all tracks with drum tracks skipped, only recordings with a reference (not a draft) are scored, a draft is the union of the compared settings with duplicates kept once, and draft writing never overwrites a reference or an existing draft. | Unit | `tests/test_tools.py` |
| **TR-26** | Phantom Re-onset Absorption | On synthesised audio: a re-onset over a decaying, un-replucked fundamental is absorbed (onset and velocity of the real pluck kept, end extended, bends concatenated); a real re-pluck, a re-onset after a gap, a silent fundamental and an out-of-range pitch are kept; phantom chains collapse; input is not mutated. CLI: on by default with the engine's audio, off with `--keep-retriggers`, runs clean → absorb → merge. | Unit / Integration | `tests/test_retrigger.py`, `tests/test_cli_wiring.py` |
| **TR-27** | Velocity Scale | `scale_velocity` rounds and clips to 1–127; the CLI scales by 0.8 by default, 1.0 keeps the engine level, and the scale applies after `--velocity-curve`. | Unit / Integration | `tests/test_post_process.py`, `tests/test_cli_wiring.py` |

---

## 2. Risk-Based Prioritization

1. **Critical Risk (Priority 1):**
   - **Viterbi String Assignment & Chord Collision Guard (TR-07, TR-08):** String collision in chords renders multi-track MIDI unplayable in DAWs.
   - **Logarithmic Velocity Transformation (TR-04):** Directly impacts musical dynamics and velocity response.
2. **High Risk (Priority 2):**
   - **Pitch Bend MIDI Export (TR-05):** Ensures pitch articulation (slides/vibrato) is preserved without corrupting timing.
   - **Frequency Bounding & HPSS Cleaning (TR-01, TR-03):** Crucial for suppressing phantom note artifacts.
3. **Medium Risk (Priority 3):**
   - **Per-String Track/Channel Assignment (TR-06):** Important for DAW compatibility. (Not MPE: there is no per-note channel rotation.)
4. **Low Risk (Priority 4):**
   - **CLI Threshold Arguments (TR-02):** UI/UX convenience.

---

## 3. Test Data & Environment Needs

- **Synthetic Audio Generation:** Python scripts using `scipy.signal` to synthesize pure sine waves, harmonic chords, and pitch-swiped sine waves for exact ground-truth testing.
- **Acoustic Test Clips:** WAV samples in `tests/`. The four TR-12 regression fixtures are `Fingerpick_mono_44-16.wav`, `Fingerpick_stereo_48-24.wav`, `plektrumpick_mono_44-16.wav` and `plektrumstrum_mono_44-16.wav`. A fifth, `ample_stereo_44-16.wav`, has no regression band and is used only by `tools/analyze_transcription.py`, which sweeps every `.wav` in `tests/`. These are **fixtures and must stay tracked in git** — `.gitignore` excludes `*.wav` globally and re-includes `tests/*.wav`, otherwise the end-to-end tests cannot run on a fresh clone. Every `.mid` in `tests/` is generated output and stays ignored.
- **Benchmark Corpus:** `GuitarSet` (https://zenodo.org/records/3371780): `annotation.zip` and `audio_mono-mic.zip`, extracted to `data/guitarset/annotation/` and `data/guitarset/audio_mono-mic/`. Not committed (`data/` is gitignored, ~664 MB); see TR-09. Split by player: `dev` (players 00–03, 240 tracks) for tuning, `test` (players 04–05, 120 tracks) only for confirming decisions.
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

To ensure fast unit test execution without running Basic Pitch inference on every test run:
- **Direct patching:** `tests/test_basic_pitch.py` patches `basic_pitch_engine.predict` with `unittest.mock` and predefined note event tuples, including per-note bend contours. (Earlier revisions named a `MockTranscriptionEngine` class; no such class exists — the `TranscriptionEngine` ABC in `src/engine_base.py` is the only abstraction, and it has one implementation.)
- **Pure-function tests:** `post_process.py`, `tab_engine.py`, and `midi_gen.py` operate on plain note dicts, so their tests construct input directly and need no engine mock at all.
- **In-process CLI harness:** `tests/test_cli_wiring.py` drives `audio2midi.main` with `click.testing.CliRunner` while patching `preprocess_pipeline`, `BasicPitchEngine` and `generate_midi`. This verifies orchestration — option resolution, stage ordering, error paths — with no audio decoding or inference, and unlike the subprocess tests it is visible to coverage.
- **Integration Tests:** Only the `slow`-marked `test_regression.py` runs actual model inference against real WAV audio. `test_cli.py` spawns the CLI but fails fast on argument validation or undecodable input, so it does not reach inference.
