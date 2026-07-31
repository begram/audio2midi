# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python 3.10.x is mandatory** — TensorFlow / `basic-pitch` do not support 3.11+.
- The venv lives at `.\venv\`. Always invoke `.\venv\Scripts\python.exe` rather than a bare `python`;
  the system interpreter is a different (incompatible) version. Never edit anything under `venv/`.
- `numpy` is pinned `<2.0.0` for TensorFlow compatibility.

## Commands

```powershell
# Transcribe
.\venv\Scripts\python.exe src\audio2midi.py "in.wav" "out.mid" --bpm 120 --tab --hpss --freq-bounds --pitch-bend

# Fast unit suite (what CI gates on)
.\venv\Scripts\python.exe -m pytest -m "not slow"

# Full suite, including end-to-end Basic Pitch regressions (minutes per fixture)
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
- `start_ticks` / `end_ticks` — added only by `quantize_notes`

New stages should read and return this shape rather than introducing a parallel representation.

### Stage ordering constraints in `main()`

These are load-bearing; reordering silently corrupts output:

1. `quantize_notes` writes `start_ticks`/`end_ticks` but does **not** touch `start`/`end`.
   `ticks_to_seconds` must be called after it, or quantization is a no-op.
2. `merge_notes` runs before `assign_strings` — the tab solver's collision guard assumes duplicate
   same-pitch detections have already been merged away.
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
  sharing a channel across strings makes per-note bends collide. `_clean_bends` dedupes repeats and
  each bent note is recentred to 0 at its end so bends don't bleed forward.
- **`merge_notes` concatenates the absorbed note's bend contour** rather than dropping it, and
  `ticks_to_seconds` retimes bends onto the quantized span — bends left at pre-quantization absolute
  times land outside their own note.
- **Emission cost is applied at every Viterbi frame** in `tab_engine`, not just frame 0; otherwise
  absolute neck position is anchored only at the start and the preference decays away.
  `max_candidates` caps the Cartesian product so a dense strum can't blow up the `|prev|×|curr|` step.

### Two independent duration gates

`--min-duration` (seconds, applied in `post_process.clean_notes`) and `--min-note-length`
(milliseconds, passed into Basic Pitch itself) are separate. The engine-level default of 127.7 ms is
the stricter of the two — raising only `--min-duration` often has no visible effect.

## Layout and imports

`src/` holds **flat top-level modules, not a package** — imports are `from processor import ...`,
with no package prefix. This is wired up in three places that must stay in sync when adding a module:
`pytest.ini` (`pythonpath = src`), `setup.py` (`py_modules`, since `find_packages` would return
nothing), and the imports themselves.

## Tests

- `tests/test_cli.py` spawns the CLI as a subprocess (verifies real argument parsing);
  `tests/test_cli_wiring.py` drives it in-process via `CliRunner` with the heavy stages patched, to
  assert that options actually reach the pipeline components. Add option-plumbing tests to the latter.
- Unit tests patch `basic_pitch_engine.predict` with mock `note_events` tuples — no model inference.
- `tests/test_regression.py` is the only `@pytest.mark.slow` module: full end-to-end inference over
  the `.wav` fixtures in `tests/`, asserting note counts against **tolerance bands**. Changing DSP or
  threshold defaults will move these numbers; update the bands deliberately, and mention it.
- Use the session fixtures in `tests/conftest.py` (`python_exe`, `cli_script`, `fixture_wav`) instead
  of hardcoding paths — `fixture_wav` skips cleanly when an audio fixture is absent.

## Project conventions

- Specifications live in `specs/` (`requirements.md`, `design.md`, `tasks.md`, `test-plan.md`); keep
  them current alongside behavioural changes.
- Comments in this codebase explain *why* a non-obvious approach was chosen (see the examples above).
  Match that: prefer a rationale comment over restating the code.
- Benchmarking against the GuitarSet dataset is a stated goal but is **not implemented** — the
  recordings in `tests/` are the current regression coverage.
