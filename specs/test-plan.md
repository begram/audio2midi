# Test Plan Specification - Polyphonic Guitar-to-MIDI Enhancements

## 1. Test Matrix

The test matrix maps each requirement from `requirements.md` to specific unit, integration, and benchmark test cases.

| ID | Requirement | Test Case Description | Test Level | Verification File |
| :--- | :--- | :--- | :--- | :--- |
| **TR-01** | FR-01: Frequency Bounding | Pass sine waves outside 80–1400 Hz range; verify pitch events outside bounds are filtered out. | Unit | `tests/test_audio2midi.py` |
| **TR-02** | FR-01: Threshold Tuning | Pass `--onset-threshold 0.7` to CLI; verify engine config receives setting. | Integration | `tests/test_cli.py` |
| **TR-03** | FR-02: HPSS Filter | Process audio with simulated pick attack transients; verify transient spikes are separated. | Unit | `tests/test_processor.py` |
| **TR-04** | FR-03: Log Velocity Mapping | Pass linear amplitudes `[0.1, 0.5, 0.9]`; verify mapped MIDI velocities follow logarithmic curve. | Unit | `tests/test_post_process.py` |
| **TR-05** | FR-04: Pitch Bend Events | Provide frame pitch bends from model mock; verify `PitchBend` objects are written to MIDI track. | Integration | `tests/test_midi_gen.py` |
| **TR-06** | FR-05: MPE 6-Channel Output | Enable `--tab`; verify output MIDI contains 6 distinct tracks mapped to MIDI Channels 1–6. | Integration | `tests/test_midi_gen.py` |
| **TR-07** | FR-06: Viterbi String Solver | Provide sequence of notes (E2 -> G2 -> B2); verify Viterbi solver outputs ergonomic fingering path. | Unit | `tests/test_tab_engine.py` |
| **TR-08** | FR-06: Chord Collision Guard | Provide simultaneous 3-note chord onset; verify all 3 notes are assigned to 3 unique physical strings. | Unit | `tests/test_tab_engine.py` |
| **TR-09** | Non-Functional Accuracy | Run accuracy benchmark script against `GuitarSet` clips; verify F-measure $\ge 0.92$. | Benchmark | `tests/test_regression.py` |

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
- **Acoustic Test Clips:** Existing WAV samples in `tests/` directory (`Fingerpick_mono_44-16.wav`, `plektrumpick_mono_44-16.wav`, `plektrumstrum_mono_44-16.wav`).
- **Benchmark Corpus:** `GuitarSet` dataset audio and annotated MIDI ground truth files.
- **Test Runner:** `pytest` executed inside Python 3.10 virtual environment (`.\venv\Scripts\python.exe`).

---

## 4. Coverage Goals per Component

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
- **`MockTranscriptionEngine`:** Unit tests for `post_process.py`, `tab_engine.py`, and `midi_gen.py` will mock `basic_pitch.inference.predict` using predefined note event dictionaries and pitch-bend arrays.
- **Integration Tests:** Only `test_regression.py` and full CLI tests will run actual model inference against real WAV audio.
