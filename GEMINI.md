# Project Context: audio2midi

This project is a high-fidelity, polyphonic acoustic guitar-to-MIDI transcriber.

## ⚠️ Critical Environment Rules
- **Python Version:** MUST use **Python 3.10.x**. (Python 3.14+ is currently incompatible with TensorFlow/Basic-Pitch).
- **Virtual Environment:** Located at E:\sw_ws\repo1\audio2midi\venv. Always use the interpreter at .\venv\Scripts\python.exe.
- **Primary Dependencies:** asic-pitch (Spotify), 	ensorflow, librosa, pretty_midi.

## Architectural Overview
The system follows a pipe-and-filter pattern:
1. **Processor (processor.py):** Loads 16/24-bit WAV, normalizes, and applies an 80Hz high-pass filter.
2. **Engine (asic_pitch_engine.py):** Wraps Spotify's Basic Pitch ML model for polyphonic inference.
3. **Post-Processor (post_process.py):** Filters short/quiet notes and applies optional BPM-based quantization.
4. **Generator (midi_gen.py):** Produces the final Standard MIDI File (SMF).
5. **CLI (udio2midi.py):** The entry point coordinating the pipeline.

## Standard Commands
- **Run Transcription:**
  .\venv\Scripts\python.exe src\audio2midi.py <input.wav> <output.mid> --bpm <value>
- **Optional Quantization:**
  Add --quantize 1/16 (or 1/4, 1/8, 1/32) to the command above.

## Workspace Mandates
- All file operations should be performed on the **E:** drive path: E:\sw_ws\repo1\audio2midi.
- Specifications are maintained in the specs/ directory.
- Verification must be done against the GuitarSet dataset for accuracy benchmarks.