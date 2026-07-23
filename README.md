# Polyphonic Guitar-to-MIDI Converter 🎸➡️🎹

This tool converts polyphonic acoustic guitar recordings (`.wav`) into high-fidelity Standard MIDI Files (`.mid`). Powered by Spotify's **Basic Pitch** deep learning model, it accurately transcribes complex fingerstyle performances, chord progressions, and strumming patterns.

## Features
- **Polyphonic Detection:** Accurately transcribes multiple simultaneous notes and overlapping melodies.
- **High-Resolution Audio:** Supports **16-bit** and **24-bit** WAV files at sample rates of 44.1kHz, 48kHz, and 96kHz.
- **Harmonic/Percussive Source Separation (HPSS):** Optional `--hpss` pre-filtering to eliminate pick attack transients and prevent false-positive phantom notes.
- **Frequency Bounding:** Restrict detection to standard acoustic guitar frequencies (80 Hz to 1,400 Hz, E2–E6) using `--freq-bounds`.
- **Expressive Pitch Bends:** Support for bends, vibrato, and slides via `--pitch-bend` written to output MIDI tracks.
- **Logarithmic Perceptual Velocity:** Optional `--velocity-curve` for dynamic loudness response matching human hearing.
- **Viterbi Tablature & MPE Support:** Global dynamic programming solver for string assignment (Strings 1–6) with strict polyphonic chord string collision guards using `--tab`.
- **Quantization & Humanization:** Grid quantization with partial `--quantize-strength` snapping (e.g. 80% strength).
- **Tempo Calibration:** Mandatory `--bpm` parameter for accurate MIDI timing.

---

## Installation

### Prerequisites
- **Python 3.10.x** (Required for TensorFlow compatibility).

### Setup
1. Open a terminal in the project directory.
2. Create and activate the virtual environment:
   ```powershell
   py -3.10 -m venv venv
   .\venv\Scripts\pip install -r requirements.txt
   ```

---

## How to Use

Run the converter using the virtual environment's Python interpreter:

```powershell
.\venv\Scripts\python.exe src\audio2midi.py "recording.wav" "output.mid" --bpm 120 --tab --hpss --freq-bounds --pitch-bend
```

### CLI Command Options

| Argument / Option | Description | Example |
| :--- | :--- | :--- |
| `input_wav` | **(Required)** Path to your source `.wav` file. | `"recording.wav"` |
| `output_midi` | **(Required)** Path where the `.mid` file will be saved. | `"output.mid"` |
| `--bpm` | **(Required)** Beats Per Minute tempo. | `--bpm 120` |
| `--hpss` | Enable Harmonic-Percussive Source Separation to filter pick clicks. | `--hpss` |
| `--freq-bounds` | Restrict pitch detection to acoustic guitar range (80–1400 Hz). | `--freq-bounds` |
| `--pitch-bend` | Extract and write pitch bend events (slides, vibrato). | `--pitch-bend` |
| `--tab` | Assign notes to strings 1–6 using Viterbi solver (MPE output). | `--tab` |
| `--velocity-curve` | Logarithmic dynamic velocity curve factor (e.g. 5.0). | `--velocity-curve 5.0` |
| `--quantize` | Snap notes to grid (`1/4`, `1/8`, `1/16`, `1/32`). | `--quantize 1/16` |
| `--quantize-strength` | Quantization snapping strength (`0.0` to `1.0`). | `--quantize-strength 0.8` |
| `--min-duration` | Filter notes shorter than X seconds. | `--min-duration 0.03` |
| `--velocity-threshold` | Ignore notes quieter than X (`0-127`). | `--velocity-threshold 10` |
| `--noise-threshold` | Amplitude threshold for noise gate (`0.0` to `1.0`). | `--noise-threshold 0.005` |
| `--no-merge` | Disable merging of overlapping identical pitches. | `--no-merge` |
| `--instrument` | Set the name of the MIDI instrument track. | `--instrument "Acoustic Guitar"` |

---

## Technical Details & Architecture

- **Pre-processing:** `librosa` (HPSS & normalization) and `scipy.signal` (80Hz Butterworth HPF).
- **Engine:** Spotify's `basic-pitch` ML inference model with configurable frequency bounds.
- **Tablature Solver:** Viterbi dynamic programming path solver with polyphonic string collision prevention.
- **MIDI Generation:** `pretty_midi` with multi-channel MPE track separation and pitch bend event writing.