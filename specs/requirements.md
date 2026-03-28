# Requirements - Polyphonic Guitar-to-MIDI Converter

## Project Overview
This software aims to provide high-fidelity, polyphonic transcription of acoustic guitar audio files (.wav) into Standard MIDI Files (.mid). It leverages state-of-the-art Deep Learning models to handle the complexities of overlapping notes and string harmonics.

## Functional Requirements
- **Audio Input:** Support mono and stereo .wav files.
  - **Sample Rates:** 44.1kHz, 48kHz, 96kHz.
  - **Bit Depths:** 16-bit and 24-bit resolution.
- **Polyphonic Transcription:** Accurately identify multiple simultaneous notes, including chords and overlapping melodies.
- **Instrument-Specific Optimization:** Optimized for acoustic guitar characteristics (decay, pluck onsets, string resonance).
- **MIDI Output:** Generate a .mid file containing:
  - Note pitches (MIDI numbers).
  - Accurate onsets and offsets (timing).
  - Velocity estimation (mapping audio intensity to MIDI velocity).
- **Tempo Calibration (Mandatory):**
  - Input parameter for BPM (Beats Per Minute).
  - Use the specified BPM for rendering the output .mid file.
  - Assume the guitarist is performing at this constant tempo for timing calculations.
- **Quantization (Optional):**
  - Optional CLI flag to snap note onsets and offsets to a musical grid (e.g., 1/16th, 1/8th notes).
  - User-configurable grid resolution.
- **Note Cleaning & Heuristics:**
  - Minimum Duration Threshold: Filter out notes shorter than a user-definable limit (default 30ms) to ignore artifacts.
  - Velocity Threshold: Filter out notes below a specific amplitude to reduce background noise/fret buzz.
- **Audio Pre-processing:**
  - Automatic Normalization: Scale input audio to optimal levels for the ML model.
  - High-Pass Filtering: Filter frequencies below 80Hz to remove non-musical rumble.
- **MIDI Metadata:**
  - Include Tempo Meta-Events and Track Name metadata in the generated .mid file.
- **CLI Interface:**
  - Command-line tool for single/batch processing.
  - Required BPM argument.
  - Real-time progress bar for long audio files.

## Non-Functional Requirements
- **Accuracy:** Minimize false positives (phantom notes) and false negatives (missed notes), especially in complex chords.
- **Latency:** Transcription should ideally take less than 1.5x the duration of the audio on a modern CPU.
- **Extensibility:** The architecture should allow for swapping transcription engines (e.g., Omnizart vs. Basic Pitch).
- **Stability:** Handle large audio files without memory overflows.

## Success Criteria
- **High Recall/Precision:** Correctly identify at least 90% of notes when validated against the **GuitarSet** dataset or similar ground truth recordings.
- **Timing Precision:** MIDI events should align with audio onsets within a 20ms tolerance (pre-quantization).
- **Polyphony Support:** Successfully transcribe at least 6 simultaneous notes.

## Target Users
- Musicians and composers needing to digitize guitar performances.
- Music researchers analyzing guitar techniques.
- Sound designers creating MIDI-based layers from acoustic recordings.

## Use Cases
- **Scenario 1:** A user records a fingerstyle piece and wants to import the MIDI into a DAW for further arrangement.
- **Scenario 2:** A researcher needs to analyze the exact timing and velocity of a professional guitar recording.

## Constraints
- **Format Support:** Limited to .wav input for the initial prototype to ensure highest quality.
- **Environment:** Requires Python 3.8+ and compatible ML libraries (TensorFlow/PyTorch).
