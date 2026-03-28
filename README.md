# Polyphonic Guitar-to-MIDI Converter 🎸➡️🎹

This tool converts polyphonic acoustic guitar recordings (.wav) into high-fidelity MIDI files (.mid). It uses Spotify's **Basic Pitch** deep learning model to accurately detect multiple simultaneous notes, chords, and melodies.

## Features
- **Polyphonic Detection:** Transcribes multiple notes at once (ideal for fingerstyle and strumming).
- **High-Resolution Support:** Handles both **16-bit** and **24-bit** WAV files.
- **Audio Cleaning:** Automatic peak normalization and 80Hz high-pass filtering to remove non-musical rumble.
- **Tempo Calibration:** Mandatory BPM setting to ensure correct MIDI timing.
- **Optional Quantization:** Snap notes to a musical grid (1/4, 1/8, 1/16, or 1/32 notes).
- **Note Filtering:** Remove "phantom notes" by setting minimum duration and velocity thresholds.

---

## Installation

### Prerequisites
- **Python 3.10** (Required for TensorFlow compatibility).
- **Windows Package Manager (winget)** or a manual Python 3.10 installation.

### Setup
1. Clone or copy this folder to your machine.
2. Open a terminal (PowerShell recommended) in the project folder.
3. Create and activate the virtual environment:
   `powershell
   py -3.10 -m venv venv
   .\venv\Scripts\pip install -r requirements.txt
   `

---

## How to Use

Run the tool using the virtual environment's Python interpreter:
- **Note Filtering:** Remove "phantom notes" by setting minimum duration and velocity thresholds.
- **String Assignment (Tablature):** Heuristic-based string/fret assignment (E2-E4 standard tuning) to split notes into 6 separate MIDI tracks (Strings 1-6).
- **Intelligent Note Merging:** Automatically merges re-triggered notes of the same pitch during chords or overlapping melodies.
- **Configurable Noise Gate:** Filter out low-level background noise before transcription.

---

## Installation
...
### Command Options

| Argument / Option | Description | Example |
| :--- | :--- | :--- |
| input_wav | Path to your source .wav file. | "recording.wav" |
| output_midi | Path where the .mid file will be saved. | "output.mid" |
| --bpm | **(Required)** The tempo of the performance. | --bpm 120 |
| --tab | Enable string/fret assignment (6 MIDI tracks). | --tab |
| --no-merge | Disable merging of overlapping identical pitches. | --no-merge |
| --noise-threshold | Amplitude threshold for noise gate (0.0 to 1.0). | --noise-threshold 0.01 |
| --quantize | Snap notes to a grid (1/4, 1/8, 1/16, 1/32). | --quantize 1/16 |
| --min-duration | Filter notes shorter than X seconds. | --min-duration 0.05 |
| --velocity-threshold | Ignore notes quieter than X (0-127). | --velocity-threshold 20 |
| --instrument | Set the name of the MIDI track. | --instrument "My Guitar" |

### Example with Tablature and Merging
To transcribe a guitar piece at 100 BPM with string assignment and noise filtering:
`powershell
.\venv\Scripts\python.exe src\audio2midi.py "solo.wav" "solo.mid" --bpm 100 --tab --noise-threshold 0.01
`

---

## Technical Details
- **Engine:** Based on Spotify's asic-pitch.
- **Pre-processing:** Uses librosa and scipy for audio conditioning.
- **MIDI Generation:** Powered by pretty_midi for precise event timing.