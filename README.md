# Polyphonic Guitar-to-MIDI Converter 🎸➡️🎹

This tool converts polyphonic acoustic guitar recordings (`.wav`) into high-fidelity Standard MIDI Files (`.mid`). Powered by Spotify's **Basic Pitch** deep learning model, it accurately transcribes complex fingerstyle performances, chord progressions, and strumming patterns.

## Features
- **Polyphonic Detection:** Accurately transcribes multiple simultaneous notes and overlapping melodies.
- **High-Resolution Audio:** Supports **16-bit** and **24-bit** WAV files at sample rates of 44.1kHz, 48kHz, and 96kHz.
- **Harmonic/Percussive Source Separation (HPSS):** Optional `--hpss` pre-filtering to eliminate pick attack transients and prevent false-positive phantom notes.
- **Frequency Bounding:** `--freq-bounds` restricts detection to 80–1,400 Hz. Basic Pitch rounds those to pitch bins, so this keeps MIDI 39–88: D#2 (a semitone of slack under the low E) up to E6.
- **Expressive Pitch Bends:** Per-note bend contours (bends, vibrato, slides) via `--pitch-bend`, taken from each note's own contour. Combine with `--tab`: pitch bend is a channel-wide MIDI message, so only per-string channels keep one string's bend off its neighbours. Without `--tab` all notes share a channel and the CLI warns about it.
- **Logarithmic Perceptual Velocity:** Optional `--velocity-curve` for dynamic loudness response matching human hearing.
- **Viterbi Tablature & Per-String Channels:** Global dynamic programming solver for string assignment (Strings 1–6) with strict polyphonic chord string collision guards using `--tab`. Each string becomes its own MIDI track/channel, so per-note bends do not collide.
- **Quantization:** Grid quantization with partial `--quantize-strength` snapping (e.g. 80% strength keeps some of the original timing).
- **Tempo Calibration:** Mandatory `--bpm` parameter for accurate MIDI timing.

---

## Installation

### Prerequisites
- **Python 3.10.x** (`basic-pitch` needs TensorFlow on 3.11+; on 3.10 it runs on the much lighter ONNX runtime).

### Setup
1. Open a terminal in the project directory.
2. Create the virtual environment and install the dependencies (no activation needed; the commands call the venv's interpreter directly):
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

| Argument / Option | Description | Default |
| :--- | :--- | :--- |
| `input_wav` | **(Required)** Path to your source `.wav` file. | — |
| `output_midi` | **(Required)** Path where the `.mid` file will be saved. | — |
| `--bpm` | **(Required)** Tempo written to the MIDI file, and the grid `--quantize` snaps to (1–400). | — |
| `--hpss` | Harmonic-Percussive Source Separation, to suppress pick clicks. | off |
| `--freq-bounds` | Restrict detection to 80–1400 Hz, i.e. MIDI 39–88 (D#2–E6). | off (unbounded) |
| `--min-freq` | Explicit minimum frequency in Hz; overrides the `--freq-bounds` minimum. | none |
| `--max-freq` | Explicit maximum frequency in Hz; overrides the `--freq-bounds` maximum. | none |
| `--pitch-bend` | Write each note's pitch bend contour (bends, slides, vibrato). Use with `--tab`; see below. | off |
| `--tab` | Assign notes to strings 1–6 with the Viterbi solver; each string gets its own track and channel. Not full MPE (no channel rotation). | off |
| `--velocity-curve` | Logarithmic velocity curve factor `k` (e.g. `5.0`); `0.0` disables. | `0.0` |
| `--velocity-scale` | Multiply output velocities by this, after `--velocity-curve`. The default matches hand-corrected recordings; `1.0` keeps the engine's level. | `0.8` |
| `--quantize` | Snap notes to a grid: `1/4`, `1/8`, `1/16`, `1/32`. | off |
| `--quantize-strength` | How far notes move toward the grid, `0.0`–`1.0`. | `1.0` |
| `--min-duration` | Drop notes shorter than this, in **seconds** (post-filter). | `0.030` |
| `--min-note-length` | Basic Pitch minimum note length, in **milliseconds** (engine-level). | `127.7` |
| `--velocity-threshold` | Drop notes below this raw velocity (`0`–`127`), judged before `--velocity-curve`. | `10` |
| `--noise-threshold` | RMS threshold for the noise gate (`0.0`–`1.0`); `0.0` disables. | `0.005` |
| `--onset-threshold` | Basic Pitch onset detection threshold (`0.0`–`1.0`). | `0.5` |
| `--frame-threshold` | Basic Pitch frame confidence threshold (`0.0`–`1.0`). | `0.3` |
| `--absorb-retriggers` / `--keep-retriggers` | When a new string is plucked, the model often restarts notes that are still ringing. Absorption folds those false re-onsets back into the ringing note, unless the audio shows that pitch was actually plucked again. May occasionally absorb a genuine re-strum of the same chord. | `--absorb-retriggers` |
| `--merge` / `--no-merge` | Merge consecutive same-pitch notes. Off by default: it absorbs genuine repeated notes, costing more recall than the precision it gains (see below). | `--no-merge` |
| `--instrument` | Track name; string tracks become `"<name> - String N"`. Names only: every track uses MIDI program 24 (nylon acoustic guitar). | `"Acoustic Guitar"` |

> `--min-duration` (seconds, applied after transcription) and `--min-note-length`
> (milliseconds, applied inside the engine) are two independent gates. The
> engine-level default of 127.7 ms is the stricter of the two.

---

## Technical Details & Architecture

- **Pre-processing:** `librosa` (HPSS, normalization, RMS envelope gate) and a vectorized one-pole 80 Hz high-pass IIR via `scipy.signal.lfilter`.
- **Engine:** Spotify's `basic-pitch` model, run as its ONNX export on `onnxruntime` (no TensorFlow), with configurable frequency bounds.
- **Tablature Solver:** Viterbi dynamic programming path solver with a per-frame neck-position emission cost, physical travel transition cost, and polyphonic string collision prevention.
- **MIDI Generation:** `pretty_midi`, one track (and channel) per assigned string, with per-note pitch bend contours recentred at each note's end. Where two bend events fall on one MIDI tick, only the one meant to be in force is kept: `pretty_midi` would otherwise order them by value.

### Why note merging is off by default

`merge_notes` collapses consecutive same-pitch notes, and was written to remove duplicate
detections. Measured against GuitarSet (20 tracks, ground-truth annotations), enabling it
costs 17 points of onset recall to gain 3 of precision — onset F1 falls from 0.742 to
0.646, and on strummed chord material from 0.704 to 0.524. The held-out test players
confirm it: 0.775 → 0.693 overall, 0.711 → 0.554 on chords.

The rule tests the gap between one note's *end* and the next's *start*. That gap stays
small for a note which sustained until the next time it was played, however far apart the
two onsets are, so repeated notes get absorbed. Rules keyed on onset separation instead
score identically to no merging at all, because Basic Pitch's own `--min-note-length`
(127.7 ms) already keeps same-pitch onsets further apart than that — the duplicates this
was meant to catch never reach it.

It remains available via `--merge` for sparser output, since trading recall for precision
is a reasonable preference.

### Measuring accuracy

`tools/evaluate_guitarset.py` scores the pipeline against [GuitarSet](https://zenodo.org/records/3371780)
ground truth (download `annotation.zip` and `audio_mono-mic.zip` into `data/guitarset/`). Tracks
are split by player: tune on the `dev` split (players 00–03, the default) and confirm on `test`
(players 04–05). On the 120 held-out test tracks the defaults reach onset F1 0.775, and `--tab`
places 66% of matched notes on the right string, against 64% for simply putting each note
nearest fret 5 (72% vs 68% on chords; level on single-note lines). `tools/analyze_transcription.py` compares presets on the recordings in `tests/`, but those
have no annotations, so its scores rank configurations rather than measure accuracy.

```powershell
.\venv\Scripts\python.exe tools\evaluate_guitarset.py --limit 20               # tune: dev split
.\venv\Scripts\python.exe tools\evaluate_guitarset.py --split test --limit 0   # confirm only
.\venv\Scripts\python.exe tools\evaluate_guitarset.py --engine-presets baseline tuned_clean   # defaults vs clean-recording thresholds
```

### Measuring on your own recordings

GuitarSet is a controlled recording, so its scores may not carry over to yours.
`tools/evaluate_recordings.py` scores the pipeline against hand-corrected references of your
own takes:

```powershell
.\venv\Scripts\python.exe tools\evaluate_recordings.py --write-drafts   # 1. drafts to correct
# 2. In a DAW, correct tests\references\<name>.draft.mid BY EAR; save as tests\references\<name>.mid
.\venv\Scripts\python.exe tools\evaluate_recordings.py                  # 3. score defaults vs clean
```

Correct the drafts by listening, not by looking at them: a draft comes from the pipeline
itself, so a reference corrected by eye leans toward the pipeline's own mistakes. References
(`tests/references/*.mid`) are tracked in git; drafts are not. Five to ten takes covering
fingerpicked, strummed and picked playing, including a noisy one, are enough to decide which
detection thresholds suit your recordings. **The full step-by-step checklist, including how to
keep the draft aligned with the audio in a DAW, is in
[`docs/ANNOTATING.md`](docs/ANNOTATING.md).**

### Clean recordings

The detection defaults (`--onset-threshold 0.5 --frame-threshold 0.3 --min-note-length 127.7`)
are Basic Pitch's own, and are kept on purpose because real recordings are rarely perfect. For
**clean** recordings (a good microphone or direct input, a quiet room, a steady level), stricter
values are measurably more accurate:

```powershell
.\venv\Scripts\python.exe src\audio2midi.py "in.wav" "out.mid" --bpm 120 --onset-threshold 0.6 --frame-threshold 0.5 --min-note-length 58
```

On held-out GuitarSet players these raise onset F1 from 0.775 to 0.822 (chords 0.711 → 0.752,
single-note lines 0.839 → 0.893), at a small cost to note-end accuracy and to `--tab` string
accuracy (66.4% → 65.7%). They are not the default because they mainly drop quiet and short
notes. On the less controlled recordings in `tests/` they found fewer of the note events audible
in the audio (event recall 0.69 → 0.61), and a missed note is more work to fix than an extra one.
If notes go missing, use the defaults; if you get spurious notes on a clean take, use these.

### Noise gate

The gate works on a smoothed short-term RMS envelope, not on individual
samples. Zeroing samples below a threshold clips the waveform at every zero
crossing, and the resulting discontinuities are broadband distortion that
Basic Pitch reports as phantom notes.