# Test Plan - Polyphonic Guitar-to-MIDI Converter

## Test Strategy
The testing strategy focuses on validating the audio processing chain, the accuracy of the transcription engine, and the correctness of the MIDI output. We will use a mix of unit tests for logic, integration tests for component interaction, and an accuracy benchmark for the ML model.

## Test Matrix

| ID | Requirement | Test Case | Type |
|----|-------------|-----------|------|
| **TR-01** | 16/24-bit .wav support | Load 16-bit and 24-bit files; verify `librosa` reads them without error. | Unit |
| **TR-02** | BPM Metadata | Generate MIDI with BPM 120; verify the MIDI header reflects 120 BPM. | Unit |
| **TR-03** | High-Pass Filter | Pass audio with sub-80Hz noise; verify noise reduction in frequency domain. | Unit |
| **TR-04** | Normalization | Load quiet audio; verify peak amplitude is -1.0 dB post-processing. | Unit |
| **TR-05** | Note Cleaning | Provide input with <30ms notes; verify they are removed in output. | Unit |
| **TR-06** | Optional Quantization | Transcribe with 1/16th grid; verify all note onsets align with grid ticks. | Integration |
| **TR-07** | Polyphony (6 notes) | Process audio of a 6-string chord; verify 6 distinct MIDI notes are generated. | Integration |
| **TR-08** | Success Metric (90%) | Run benchmark against **GuitarSet**; verify F-measure is >= 0.90. | Benchmark |
| **TR-09** | CLI Interface | Execute `audio2midi.py` with missing BPM; verify error and help message. | E2E |

## Test Data & Environment
- **Synthetic Audio:** Pure sine wave chords (C major, G major) to test pitch detection and polyphony in ideal conditions.
- **Acoustic Samples:** Short recordings of an acoustic guitar (strumming and fingerstyle).
- **GuitarSet:** A subset of the [GuitarSet dataset](https://guitarset.weebly.com/) for objective accuracy metrics.
- **Tools:** `pytest` for running tests, `mido` or `pretty_midi` for inspecting output MIDI files.

## Risk-Based Prioritization
1. **Critical:** Transcription accuracy (TR-08) and Polyphony (TR-07). If these fail, the tool is not useful.
2. **High:** Timing/BPM (TR-02) and Quantization (TR-06). Essential for musical usability.
3. **Medium:** Audio pre-processing (TR-03, TR-04).
4. **Low:** CLI UX and error handling (TR-09).

## Mocking Strategy
- **ML Engine:** For unit tests of the MIDI generator and post-processor, the transcription engine will be mocked to return a predefined list of note events.
- **File System:** `pytest`'s `tmp_path` will be used to manage temporary audio and MIDI files during testing.
