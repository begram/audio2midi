# Polyphonic Guitar-to-MIDI Converter 🎸➡️🎹

This tool converts polyphonic acoustic guitar recordings (`.wav`) into high-fidelity Standard MIDI Files (`.mid`). Powered by Spotify's **Basic Pitch** deep learning model, it accurately transcribes complex fingerstyle performances, chord progressions, and strumming patterns.

## Features
- **Polyphonic Detection:** Accurately transcribes multiple simultaneous notes and overlapping melodies.
- **High-Resolution Audio:** Supports **16-bit** and **24-bit** WAV files at sample rates of 44.1kHz, 48kHz, and 96kHz.
- **Harmonic/Percussive Source Separation (HPSS):** Optional `--hpss` pre-filtering to eliminate pick attack transients and prevent false-positive phantom notes.
- **Frequency Bounding:** Restrict detection to standard acoustic guitar frequencies (80 Hz to 1,400 Hz, E2–E6) using `--freq-bounds`.
- **Expressive Pitch Bends:** Per-note bend contours (bends, vibrato, slides) via `--pitch-bend`, taken from each note's own contour so one string's bend is never copied onto its neighbours.
- **Logarithmic Perceptual Velocity:** Optional `--velocity-curve` for dynamic loudness response matching human hearing.
- **Viterbi Tablature & Per-String Channels:** Global dynamic programming solver for string assignment (Strings 1–6) with strict polyphonic chord string collision guards using `--tab`. Each string becomes its own MIDI track/channel, so per-note bends do not collide.
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

For the test and lint toolchain, install the dev requirements instead:
```powershell
.\venv\Scripts\pip install -r requirements-dev.txt
```

### Running the tests

```powershell
.\venv\Scripts\python.exe -m pytest -m "not slow"         # fast unit suite
.\venv\Scripts\python.exe -m pytest                       # includes end-to-end regressions
.\venv\Scripts\python.exe -m pytest -m "not slow" --cov   # with coverage report
.\venv\Scripts\python.exe -m ruff check .                 # lint
```

The `slow` marker covers the end-to-end regressions, which run full Basic Pitch
inference over the audio fixtures in `tests/`.

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
| `--min-duration` | Filter notes shorter than X seconds (post-filter). | `--min-duration 0.03` |
| `--min-note-length` | Basic Pitch minimum note length in **milliseconds** (engine-level). | `--min-note-length 127.7` |
| `--velocity-threshold` | Ignore notes quieter than X (`0-127`). | `--velocity-threshold 10` |
| `--noise-threshold` | RMS threshold for the noise gate (`0.0` to `1.0`, `0.0` disables). | `--noise-threshold 0.005` |
| `--min-freq` | Explicit minimum frequency bound in Hz (overrides `--freq-bounds`). | `--min-freq 80` |
| `--max-freq` | Explicit maximum frequency bound in Hz (overrides `--freq-bounds`). | `--max-freq 1400` |
| `--onset-threshold` | Basic Pitch onset detection threshold (`0.0` to `1.0`). | `--onset-threshold 0.5` |
| `--frame-threshold` | Basic Pitch frame confidence threshold (`0.0` to `1.0`). | `--frame-threshold 0.3` |
| `--no-merge` | Disable merging of overlapping identical pitches. | `--no-merge` |
| `--instrument` | Set the name of the MIDI instrument track. | `--instrument "Acoustic Guitar"` |

> `--min-duration` (seconds, applied after transcription) and `--min-note-length`
> (milliseconds, applied inside the engine) are two independent gates. The
> engine-level default of 127.7 ms is the stricter of the two.

---

## Technical Details & Architecture

- **Pre-processing:** `librosa` (HPSS, normalization, RMS envelope gate) and a vectorized one-pole 80 Hz high-pass IIR via `scipy.signal.lfilter`.
- **Engine:** Spotify's `basic-pitch` ML inference model with configurable frequency bounds.
- **Tablature Solver:** Viterbi dynamic programming path solver with a per-frame neck-position emission cost, physical travel transition cost, and polyphonic string collision prevention.
- **MIDI Generation:** `pretty_midi`, one track (and channel) per assigned string, with per-note pitch bend contours recentred at each note's end.

### Noise gate

The gate works on a smoothed short-term RMS envelope, not on individual
samples. Zeroing samples below a threshold clips the waveform at every zero
crossing, and the resulting discontinuities are broadband distortion that
Basic Pitch reports as phantom notes.