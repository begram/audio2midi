# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python 3.10.x is mandatory.** Inference runs on ONNX (`basic_pitch_engine.MODEL_PATH`), and
  TensorFlow is not installed. But `basic-pitch` 0.4 makes TensorFlow a hard dependency on
  Python 3.11+, so moving off 3.10 would pull it back in.
- The venv lives at `.\venv\`. Always invoke `.\venv\Scripts\python.exe` rather than a bare `python`;
  the system interpreter is a different (incompatible) version. Never edit anything under `venv/`.
- `numpy` is pinned `<2.0.0`. The pin dates from TensorFlow, which is gone; lifting it is untested
  (librosa/numba/onnxruntime under numpy 2), so treat that as its own change.
- Don't reinstall TensorFlow: `basic-pitch` prefers it whenever it can be imported, and even an
  empty leftover `site-packages/tensorflow` folder makes `import basic_pitch.inference` fail
  looking for `tensorflow.lite`.

## Commands

```powershell
# Transcribe
.\venv\Scripts\python.exe src\audio2midi.py "in.wav" "out.mid" --bpm 120 --tab --hpss --freq-bounds --pitch-bend

# Fast unit suite (what CI gates on)
.\venv\Scripts\python.exe -m pytest -m "not slow"

# Full suite, including end-to-end Basic Pitch regressions (~15 s more on ONNX)
.\venv\Scripts\python.exe -m pytest

# Single test / single case
.\venv\Scripts\python.exe -m pytest tests\test_post_process.py::test_merge_notes
.\venv\Scripts\python.exe -m pytest -k "pitch_bend"

# Coverage (gate: fail_under = 95, branch mode; see .coveragerc)
.\venv\Scripts\python.exe -m pytest -m "not slow" --cov

# Lint (line-length 110; E,F,W,I,UP,B)
.\venv\Scripts\python.exe -m ruff check .
```

Dev toolchain: `.\venv\Scripts\pip install -r requirements-dev.txt`.

```powershell
# Transcription quality analysis: sweeps presets over the fixtures, writes analysis/metrics.csv
.\venv\Scripts\python.exe tools\analyze_transcription.py
.\venv\Scripts\python.exe tools\analyze_transcription.py --fixtures Fingerpick_mono_44-16.wav
```

```powershell
# Ground-truth accuracy against GuitarSet, writes analysis/guitarset_<split>.csv
.\venv\Scripts\python.exe tools\evaluate_guitarset.py --limit 20               # dev split (tuning)
.\venv\Scripts\python.exe tools\evaluate_guitarset.py --split test --limit 0   # held-out confirmation
```

GuitarSet is **split by player**: `dev` is players 00–03, `test` is players 04–05 (120 tracks). The
default is `dev`. Tune and compare settings on `dev` only, and run `test` to confirm a decision,
never to make one: a setting chosen by looking at test scores makes those scores optimistic.

`tools/evaluate_guitarset.py` is the only thing here that measures **accuracy**. It needs the
corpus in place (~664 MB, both archives from https://zenodo.org/records/3371780, extracted to
`data/guitarset/annotation/` and `data/guitarset/audio_mono-mic/`; `data/` is gitignored).
GuitarSet was recorded with a hexaphonic pickup, so it is also the only way to score
`tab_engine` — plain audio carries no signal saying which string was played. Metrics follow
MIREX conventions via `mir_eval` (50 ms onset window, 50 cent pitch tolerance). Note the
annotation `data_source` runs 0 (low E) to 5 (high e), the opposite of `tab_engine`'s 6-to-1
numbering, and reference pitches are continuous f0 estimates rather than integers.

`tools/analyze_transcription.py` caches raw engine output under `analysis/cache/` (via
`tools/engine_cache.py`, whose key also hashes `processor.py`, `basic_pitch_engine.py` and the
`basic-pitch` version, so editing a default invalidates it), so only preprocessing/engine changes
pay for inference; post-processing presets re-sweep for free.
Its metrics are plausibility and self-consistency scores, **not** accuracy — there are no ground-truth
annotations for the fixtures, so the numbers only rank configurations against each other. Real
precision/recall needs annotated data (GuitarSet, which also carries the per-string labels required
to score `tab_engine`). All of `analysis/` is generated and gitignored.

## Architecture

Pipe-and-filter. `src/audio2midi.py` (Click CLI) is the only orchestrator; every other module is a
pure stage that takes and returns note data.

```
wav ──> processor ──> basic_pitch_engine ──> post_process ──> tab_engine ──> midi_gen ──> .mid
        (DSP prep)    (ML inference)         (clean/merge/    (string       (SMF write)
                                              quantize)        assignment)
```

**The note dict is the contract between every stage.** Defined in `src/engine_base.py`
(`TranscriptionEngine.transcribe` docstring):

- `pitch` (int MIDI number), `start` / `end` (float seconds), `velocity` (int 1–127)
- `pitch_bends` — optional `[{'pitch': int MIDI ticks, 'time': float seconds}]`
- `string` / `fret` — added only by `TabMapper.assign_strings`
- `start_ticks` / `end_ticks` — added only by `quantize_notes`, at 480 ticks per quarter. They record
  the quantized grid position and are not updated later: when `assign_strings` truncates a note,
  `end` changes and `end_ticks` does not. Nothing downstream reads them; `start`/`end` are the truth.

New stages should read and return this shape rather than introducing a parallel representation.

### Stage ordering constraints in `main()`

These are load-bearing; reordering silently corrupts output:

1. `quantize_notes` writes `start_ticks`/`end_ticks` but does **not** touch `start`/`end`.
   `ticks_to_seconds` must be called after it, or quantization is a no-op.
2. `merge_notes` (opt-in via `--merge`) runs before `assign_strings`. It is **off by default**:
   measured against GuitarSet it trades 17 points of onset recall for 3 of precision, because it
   absorbs notes that merely sustained until the next time the same pitch was played. See its
   docstring for the full numbers before considering re-enabling it. Since it normally does not
   run, `_resolve_string_overlaps` must handle duplicate same-pitch detections itself rather than
   assuming they were merged away.
3. `clean_notes` runs before `apply_logarithmic_velocity`, so the velocity threshold applies to raw
   engine confidence, not to remapped values.

### Design decisions that look like candidates for "simplification" but aren't

- **Noise gate operates on a smoothed RMS envelope, not per-sample** (`processor.noise_gate`).
  Sample-wise gating clips at every zero crossing; the resulting discontinuities are broadband
  distortion that Basic Pitch reports as phantom notes. The Hann smoothing is edge-padded so the
  gate doesn't fade in over the first note's attack.
- **Pitch bends come from each note's own contour** (`note_events[i][4]`), never from time-window
  matching. Window matching copies one string's bend onto every simultaneous note.
- **One MIDI track/channel per string** (`midi_gen`). Pitch bend is a channel-level message, so
  sharing a channel across strings makes per-note bends collide. Each bent note is recentred to 0
  at its end so bends don't bleed forward. Without `--tab` that sharing is unavoidable (everything
  lands on the unassigned track), so the CLI warns when `--pitch-bend` is used alone.
- **Exactly one pitch-wheel event per MIDI tick** (`midi_gen._channel_bends`). `pretty_midi`
  writes same-tick pitch-wheel events sorted by *value*, so insertion order means nothing there.
  The survivor is the event meant to be in force: the later-starting note's, and within a note the
  later in time (so a recentre beats its note's own last bend). Shared ticks are routine: Basic
  Pitch puts the last bend exactly at the note's end, truncation and quantization make notes touch,
  and at low tempos a tick is longer than the ~11.6 ms between contour points.
- **Tab emission cost is the mean of per-note distances from `PREFERRED_FRET`**, not the distance of
  the mean fret. Averaging first scores frets 1 + 9 the same as 5 + 5, letting chords stretch beyond
  any hand; measured on GuitarSet it was the main reason chord string accuracy was ~0.52.
- **Open strings carry a penalty (`OPEN_STRING_PENALTY = 8`), not a bonus.** Without it an open
  string is free twice over: no position cost (no fretted note), and `_transition_cost` is 0 into
  or out of a frame without fretted notes, so the hand moves through it for free. The old bonus made
  the solver take open strings wherever a melody allowed. The value was picked on the dev split
  (gains plateau from ~8) and confirmed on held-out players: string accuracy 0.610 → 0.664.
- **`basic_pitch_engine` filters pitches exactly as Basic Pitch bins them**: `round(min)` to
  `round(max)` exclusive, so `--freq-bounds` (80–1400 Hz) keeps MIDI 39–88 (D#2–E6). Don't
  "tighten" it to E2 by switching to floor/ceil; that only makes the filter disagree with the engine.
- **`merge_notes` concatenates the absorbed note's bend contour** rather than dropping it, and
  `ticks_to_seconds` retimes bends onto the quantized span — bends left at pre-quantization absolute
  times land outside their own note.
- **Emission cost is applied at every Viterbi frame** in `tab_engine`, not just frame 0; otherwise
  absolute neck position is anchored only at the start and the preference decays away.
  `max_candidates` caps the Cartesian product so a dense strum can't blow up the `|prev|×|curr|` step.
- **`_resolve_string_overlaps` runs after the solver, not inside it.** The candidate guard only
  prevents collisions within one 15 ms chord frame; the Viterbi state is a single frame, so a sustain
  spanning frames is invisible to it, and making it sustain-aware would mean carrying the set of
  ringing strings in the state space. A later pluck therefore truncates the earlier note — which is
  what a real string does. Conflicts closer than `MIN_REPLUCK_SECONDS` are not re-plucks, so there
  the note keeps its full length and gives up its string instead, routing to the unassigned track;
  truncating those would emit millisecond clicks.

  The invariant to preserve: **no two simultaneous notes may share a string.** Violating it also
  breaks the per-string channel design, since one channel's bend would apply to both notes.

### Two independent duration gates

`--min-duration` (seconds, applied in `post_process.clean_notes`) and `--min-note-length`
(milliseconds, passed into Basic Pitch itself) are separate. The engine-level default of 127.7 ms is
the stricter of the two — raising only `--min-duration` often has no visible effect.

## Layout and imports

`src/` holds **flat top-level modules, not a package** — imports are `from processor import ...`,
with no package prefix. This is wired up in three places that must stay in sync when adding a module:
`pytest.ini` (`pythonpath = src tools`), `setup.py` (`py_modules`, since `find_packages` would
return nothing), and the imports themselves. `tools/` is flat the same way but deliberately not
installed: its scripts put `src` on `sys.path` themselves, and only `pytest.ini` lists it.

## Tests

- `tests/test_cli.py` spawns the CLI as a subprocess (verifies real argument parsing);
  `tests/test_cli_wiring.py` drives it in-process via `CliRunner` with the heavy stages patched, to
  assert that options actually reach the pipeline components. Add option-plumbing tests to the latter.
  Its `real_writer` fixture keeps the real `generate_midi`, for invariants only visible in the
  written file (string exclusivity, bends inside their notes, wheel back at 0).
- `tests/test_tools.py` covers `tools/` bookkeeping (GuitarSet string numbering, `string_accuracy`,
  track selection, cache invalidation); `pytest.ini` puts `tools` on the path for it.
- Unit tests patch `basic_pitch_engine.predict` with mock `note_events` tuples — no model inference.
- `tests/test_regression.py` is the only `@pytest.mark.slow` module: full end-to-end inference over
  the `.wav` fixtures in `tests/`, run with CLI defaults only, asserting note counts against
  **tolerance bands** (about ±8% around the measured count). Changing DSP or threshold defaults will
  move these numbers; re-measure and update the bands deliberately, and mention it.
- Use the session fixtures in `tests/conftest.py` (`python_exe`, `cli_script`, `fixture_wav`) instead
  of hardcoding paths — `fixture_wav` skips cleanly when an audio fixture is absent.

## Project conventions

- Specifications live in `specs/` (`requirements.md`, `design.md`, `tasks.md`, `test-plan.md`); keep
  them current alongside behavioural changes.
- Comments in this codebase explain *why* a non-obvious approach was chosen (see the examples above).
  Match that: prefer a rationale comment over restating the code.
- GuitarSet benchmarking (`tools/evaluate_guitarset.py`) is a manual tool, not part of the test
  suite — the recordings in `tests/` remain the automated regression coverage.
