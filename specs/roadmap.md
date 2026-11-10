# Roadmap — Recommended Continuation

> Forward-looking plan, written 2026-11-06 after the GuitarSet evaluation work. It
> supersedes nothing: `tasks.md` is the historical plan that got the project here, and
> §6 of `requirements.md` records where the implementation stands today. Every number
> below comes from `tools/evaluate_guitarset.py` (GuitarSet, MIREX scoring) or
> `tools/analyze_transcription.py` (the recordings in `tests/`, no ground truth).

## How to read this

**Importance levels**

| Level | Meaning |
| :--- | :--- |
| **Critical** | Blocks trustworthy decisions or leaves output wrong. Do first. |
| **High** | A large measured gap, or a correctness risk users will hit. |
| **Medium** | Clear user or maintenance value; no measured problem forcing it. |
| **Low** | Nice to have; do when convenient or on request. |

**Effort**: **S** = a focused change with tests, **M** = several components or a new tool,
**L** = new infrastructure, research, or model work. Sizes compare items with each other;
they are not time estimates.

**Rule that applies to every tuning item**: choose settings on the GuitarSet `dev` split
(players 00–03) and confirm *once* on `test` (players 04–05). A setting chosen by looking at
test scores makes those scores optimistic. Record each decision in the appendix.

## 1. Where the project stands

| Measure (held-out GuitarSet players, 120 tracks) | Value | Target |
| :--- | :--- | :--- |
| Onset F1, CLI defaults | 0.784 (chords 0.724, single notes 0.843) | 0.92 |
| Onset F1, clean-recording thresholds (opt-in) | 0.822 (chords 0.755, single notes 0.890) | 0.92 |
| Offset F1, CLI defaults | 0.552 | — |
| `--tab` string accuracy | 0.661 (naive nearest-to-fret-5 baseline: 0.643) | — |
| `--tab` string accuracy, single-note lines | 0.612 (naive: 0.611) | — |
| Test suite / coverage | all passing, slow regressions included; 99.8% branch coverage of `src/` | ≥ 95% coverage |

Open requirement gaps (`requirements.md` §6): the accuracy target, latency and memory
budgets (no harness), `--tuning`, and type annotations.

The single biggest unknown: **the right detection thresholds for real, imperfect
recordings.** GuitarSet favours stricter thresholds; the project's own recordings point the
other way, but they have no ground truth, so the question could not be settled.

## 2. Summary

| ID | Item | Importance | Effort | Phase |
| :--- | :--- | :--- | :--- | :--- |
| A1 | Confirm CI is green on Linux — ✅ done | **Critical** | S | 0 |
| A2 | Ground truth for your own recordings — 🟡 3 of 5 distinct takes annotated | **Critical** | M | 1 |
| A3 | Confidence intervals on every comparison | **High** | S | 1 |
| D1 | Write the pitch-bend range into the file | **High** | S | 1 |
| B1 | Measure preprocessing against ground truth | **High** | S | 1 |
| B2 | Chord-recall error analysis | **High** | M | 2 |
| B3 | Choose thresholds per recording | **High** | M | 2 |
| C1 | Tune hand-movement weights for single-note lines | **High** | S | 2 |
| B4 | Revise the 92% accuracy target | **High** | S | 2 |
| C2 | Tablature export (ASCII tab, MusicXML) | **High** | M | 3 |
| F1 | `--preset clean` instead of three flags | Medium | S | 3 |
| C3 | Alternate tunings (`--tuning`) and capo | Medium | M | 3 |
| C4 | Hand-span limit for chord candidates | Medium | S | 2 |
| B5 | Note-end (offset) accuracy | Medium | M | 3 |
| A4 | Latency and memory harness | Medium | S | 3 |
| E1 | Run inference without the `basic-pitch` package | Medium | M | 4 |
| E2 | CI: Windows runner and a scheduled slow job | Medium | S | 3 |
| F2 | Batch processing of a folder | Medium | S | 3 |
| E3 | `.gitattributes` for line endings | Low | S | 0 |
| C5 | Sustain-aware tab state | Low | L | 4 |
| B6 | Fine-tune or replace the model | Low | L | 4 |
| D2 | `--program` (steel vs nylon) | Low | S | 3 |
| D3 | Optional MPE output | Low | M | 4 |
| E4 | Type annotations, or revise that requirement | Low | M | 4 |
| F3 | More input formats (FLAC, MP3) | Low | S | 4 |

## 3. Phases

- **Phase 0 — housekeeping (now):** A1, E3. Small, and A1 must not stay unknown.
- **Phase 1 — measure what matters:** A2, A3, B1, D1. Better measurement first, because
  every later decision depends on it; D1 is a cheap correctness fix that rides along.
- **Phase 2 — accuracy:** B2, B3, C1, C4, B4. Uses the Phase 1 measurements.
- **Phase 3 — features and polish:** C2, F1, C3, B5, A4, E2, F2, D2.
- **Phase 4 — longer term:** E1, C5, B6, D3, E4, F3.

## 4. Items in detail

### A — Measurement

#### A1 · Confirm CI is green on Linux — **Critical** · S · ✅ Done 2026-11-07
- **Why:** several pushes ran CI for the first time with the ONNX backend and the new
  `tests/test_tools.py`, and the result had not been checked.
- **Result:** every `main` run since the ONNX switch passed (`450fba6`, `be29a30`,
  `260df2a`, `e9689ff`), checked through the GitHub API. The ONNX backend and the tool tests
  work on Linux.

#### A2 · Ground truth for your own recordings — **Critical** · M · 🟡 3 of 5 distinct takes annotated
- **Status (2026-11-10):** references are committed for four of the five recordings in
  `tests/`, which are three distinct takes: fingerpicked (in two audio formats, identical
  references), picked and strummed. `ample_stereo_44-16` is not annotated. The threshold
  decision needs **two more distinct takes**, ideally a noisy or quiet one. The references
  already decided two other defaults: re-onset absorption and the velocity scale (appendix).
  The step-by-step checklist is [`docs/ANNOTATING.md`](../docs/ANNOTATING.md).
- **Why:** the threshold decision split by recording type. GuitarSet (controlled recording)
  says stricter thresholds gain 4.7 points of onset F1; the recordings in `tests/` suggest
  they lose audible notes (event recall 0.692 → 0.612), but those scores are self-consistency,
  not accuracy. Only annotated recordings like the ones users actually make can settle it.
- **What:**
  1. Pick 5–10 short takes representative of real use: fingerpicked, strummed, picked
     lines, at least one noisy or quiet recording.
  2. Correct the tool's draft for each by ear in a DAW (delete, add, move notes) and save it
     as `tests/references/<name>.mid`. The draft is the union of both settings being
     compared, so it gives neither a head start.
  3. Score the pipeline against those references with `tools/evaluate_recordings.py`, which
     uses the same `mir_eval` settings as the GuitarSet tool.
  4. Decide the defaults with the rule fixed in advance in `docs/ANNOTATING.md` (last section).
- **Done when:** the default thresholds are chosen from measured accuracy on representative
  recordings, and the decision is recorded in `basic_pitch_engine.py` and the appendix.
- **Note:** the drafts come from the tool itself, so a reference can still lean toward what
  both settings found (a note both missed may stay missed). Correct by listening, not by
  looking.

#### A3 · Confidence intervals on every comparison — **High** · S
- **Why:** several decisions turned on small differences (`max_frets` 12 vs 15: 0.002; the
  threshold change's cost to string accuracy: 0.007). Without an interval there is no way to
  tell a real difference from track-to-track noise.
- **What:** add a paired bootstrap over tracks to `evaluate_guitarset.py`: for two presets,
  report the mean difference and a 95% interval. Unit-test it on synthetic data with a known
  difference.
- **Done when:** comparisons print `Δ = x [lo, hi]`, and docs quote intervals for decisions.

#### A4 · Latency and memory harness — **Medium** · S
- **Why:** two requirements (under 1.5× real time; under 2 GB for 10 minutes of audio) are
  "Unverified". Inference on ONNX is fast now, so they are probably met, but that is a guess.
- **What:** `tools/benchmark.py` timing each stage and recording peak memory on a 10-minute
  file (concatenate fixtures). Optionally a slow test asserting the budgets.
- **Done when:** both rows in `requirements.md` §6 read "Met" or "Not met" with numbers.

### B — Transcription accuracy

#### B1 · Measure preprocessing against ground truth — **High** · S
- **Why:** the noise gate threshold (0.005), the 80 Hz high-pass and `--hpss` were designed by
  reasoning about artefacts, never measured against ground truth in this work. The GuitarSet
  tool already has an `hpss` preset; the gate has none.
- **What:** add presets (`gate_off`, gate thresholds 0.002 / 0.01 / 0.02, `hpss`) and compare on
  `dev`; confirm any winner on `test`; cross-check on A2's recordings, since gate and
  recording noise interact.
- **Done when:** each preprocessing default is either justified by a measurement or changed.

#### B2 · Chord-recall error analysis — **High** · M
- **Why:** chords are the weakest material (onset F1 0.711 vs 0.839 for single notes), and
  nobody has looked at *which* chord notes go missing.
- **What:** break misses and false notes down by pitch register, polyphony, position in the
  strum, and octave relation (an octave error looks like one miss plus one false note).
  The data is all in `mir_eval`'s matching; this is analysis, then targeted fixes.
- **Done when:** the main error classes are quantified, each with a candidate fix or a reason
  it is the model's limit.

#### B3 · Choose thresholds per recording — **High** · M · depends on A2
- **Why:** the stricter thresholds help clean recordings and hurt imperfect ones. Picking
  them automatically from the recording would give each user the right setting without
  having to know about thresholds.
- **What:** estimate recording quality (noise floor from the gate's RMS envelope, or SNR
  between notes and silence) and select defaults vs clean values above a threshold. Tune the
  rule on GuitarSet `dev` plus half of A2's recordings; confirm on the rest.
- **Done when:** the automatic choice matches or beats both fixed settings on A2's
  recordings and on GuitarSet `test`. If it doesn't, document that and drop the idea.

#### B4 · Revise the 92% accuracy target — **High** · S
- **Why:** threshold tuning plateaued at 0.822 on held-out players. Reaching 0.92 would take a
  different or retrained model (B6), not more tuning. A target nobody is working toward makes
  the requirements less useful, not more ambitious.
- **What:** restate the requirement with what is measured and achievable (for example a
  floor on held-out onset F1 that guards against regressions), and move 0.92 to B6 as that
  item's goal.
- **Done when:** `requirements.md` has a target the current pipeline is measured against.

#### B5 · Note-end (offset) accuracy — **Medium** · M
- **Why:** offset F1 is 0.540: note *ends* are much less reliable than starts. This matters
  for how notes sound when played back, less for notation.
- **What:** measure where ends go wrong (too early, too late, by note length); try
  end-of-note rules such as extending to the next onset on the same string under `--tab`.
- **Done when:** offset F1 improves on `test` without hurting onset F1, or the cause is
  documented as a model limit.

#### B6 · Fine-tune or replace the model — **Low** · L
- **Why:** the remaining onset gap is mostly the model's. This is the only route toward 0.92.
- **What:** research first: guitar-specific transcription models, their licences, and
  whether fine-tuning Basic Pitch on GuitarSet `dev` is feasible (it would need training
  infrastructure and must never touch `test`).
- **Done when:** a written recommendation with expected gain and cost exists. Implementation
  is a separate decision.

### C — Tablature

#### C1 · Tune hand-movement weights for single-note lines — **High** · S
- **Why:** after the open-string fix, the solver is clearly better than the naive baseline on
  chords (0.715 vs 0.678) but only level on single-note lines (0.612 vs 0.610). The fret- and
  string-travel weights (1.5 / 0.5) were never tuned since the cost changes.
- **What:** sweep `FRET_TRAVEL_WEIGHT`, `STRING_TRAVEL_WEIGHT` and `PREFERRED_FRET` on `dev`
  (solving is milliseconds), scoring chords and single notes separately; confirm on `test`.
  Use A3's intervals to decide.
- **Done when:** single-note accuracy beats the naive baseline with an interval excluding
  zero, without losing chord accuracy.

#### C2 · Tablature export — **High** · M
- **Why:** the solver computes string and fret for every note, but they only decide MIDI
  channels. Players who use `--tab` almost certainly want to *read* tab.
- **What:** `--tab-out file.txt` for ASCII tab (six lines, bar lines from `--bpm`), then
  MusicXML with string/fret technical notation, which notation software and Guitar Pro
  import. Golden-file tests.
- **Done when:** a fixture's tab opens in MuseScore with the correct strings and frets.

#### C3 · Alternate tunings and capo — **Medium** · M
- **Why:** `--tuning` is a stated requirement that does not exist; `TabMapper.TUNING` is
  hardcoded to standard tuning, so drop-D or open tunings get wrong fingerings.
- **What:** `--tuning E2,A2,D3,G3,B3,E4` (and named shortcuts like `drop-d`), `--capo N`.
  The pitch-to-placement table, `max_frets` and the frequency bounds must follow the tuning.
  Unit tests only (GuitarSet is all standard tuning).
- **Done when:** the requirement row reads "Met"; drop-D low D (MIDI 38) gets string 6, fret 0.

#### C4 · Hand-span limit for chord candidates — **Medium** · S
- **Why:** the per-note cost discourages stretched chords but does not forbid them; no hand
  spans, say, seven frets.
- **What:** drop candidates whose fretted span exceeds a limit (try 4–5) before the
  candidate cap, falling back to the current behaviour if nothing remains. Tune on `dev`.
- **Done when:** measured on `dev`/`test`; keep only if it helps.

#### C5 · Sustain-aware tab state — **Low** · L
- **Why:** the solver cannot see a sustain spanning frames; overlap resolution fixes the
  collisions afterwards. Carrying the ringing strings in the state would let the solver
  avoid them in the first place, at a large cost in state size.
- **What:** prototype only if C1 and C4 leave a measurable gap that overlap resolution causes.

### D — MIDI output

#### D1 · Write the pitch-bend range into the file — **High** · S
- **Why:** bend values assume the receiver's default ±2-semitone range, so any wider bend or
  slide is silently clipped, and a receiver set to another range plays every bend wrong.
- **What:** write the standard registered-parameter messages (RPN 0) setting the range on
  each track that carries bends, and scale values to it. A `--bend-range` option (default
  12, so slides fit). Tests read the controller events back.
- **Done when:** a 5-semitone slide survives unclipped and plays at the right pitch in a DAW.

#### D2 · `--program` (steel vs nylon) — **Low** · S
- **Why:** every track uses General MIDI program 24 (nylon), while the default track name
  says only "Acoustic Guitar"; a steel-string guitar is program 25, and there is no way to
  choose it.
- **What:** a `--program` option, default unchanged.

#### D3 · Optional MPE output — **Low** · M
- **Why:** the requirements describe MPE; the implementation is per-string channels, which
  gives the same benefit for guitar. Real MPE helps only MPE-aware synths.
- **What:** only on request.

### E — Engineering

#### E1 · Run inference without the `basic-pitch` package — **Medium** · M
- **Why:** `basic-pitch` 0.4 pins the Python version in practice: on 3.11+ it requires
  TensorFlow. The project only needs its ONNX model file and its note-creation code.
  Separately, the project's own `numpy<2` pin is left over from TensorFlow (basic-pitch
  itself only asks for `numpy>=1.18`) and has never been tested without.
- **What:** load the ONNX model with `onnxruntime` directly, depend on (or vendor, licence
  permitting) the note-creation step, and drop the package. Then test Python 3.11/3.12, and
  numpy 2 with the pin lifted.
- **Done when:** the suite passes on a newer Python with identical regression and GuitarSet
  numbers.

#### E2 · CI: Windows runner and a scheduled slow job — **Medium** · S
- **Why:** development happens on Windows, but CI runs only Linux; the slow regressions and
  the tool tests never run automatically.
- **What:** a Windows job in the matrix; a nightly or weekly workflow running `pytest`
  (all markers).
- **Done when:** both run green.

#### E3 · `.gitattributes` for line endings — **Low** · S
- **Why:** git warns "LF will be replaced by CRLF" on several files at every commit.
- **What:** add `.gitattributes` with `* text=auto eol=lf` (or the team's preference) and
  renormalise once.

#### E4 · Type annotations, or revise that requirement — **Low** · M
- **Why:** listed as "Not met". The codebase documents itself well in prose; annotations
  would add tooling checks but are a mechanical, repo-wide change.
- **What:** decide first; if yes, annotate `src/` and add a type checker to CI.

### F — Usability

#### F1 · `--preset clean` instead of three flags — **Medium** · S
- **Why:** the clean-recording setting needs three exact flags today.
- **What:** `--preset default|clean` mapping to the `DEFAULT_*` / `CLEAN_RECORDING_*`
  constants, with explicit flags still overriding. Superseded if B3 succeeds.

#### F2 · Batch processing of a folder — **Medium** · S
- **What:** accept a directory of `.wav` files and an output directory; skip on error with
  a summary at the end.

#### F3 · More input formats — **Low** · S
- **What:** FLAC should already load (librosa reads it through `soundfile`), but that is
  untested; add a test and document it. MP3 depends on which decoder librosa finds on the
  machine and needs care.

## Appendix — Decision record

Decisions made so far, with the evidence. Keep adding to this. Figures are as measured when
each decision was made; later defaults move the baselines (re-onset absorption, 2026-11-10, took
the held-out default from 0.775 to 0.784 onset F1), so compare within a row, not across rows.

| Decision | Chosen on | Confirmed on held-out players | Where documented |
| :--- | :--- | :--- | :--- |
| Merging off by default | 20 tracks, all players | onset F1 0.775 vs 0.693 with `--merge` | `post_process.merge_notes` |
| Per-note position cost | 20 tracks, all players | string accuracy 0.516 → 0.610 | `tab_engine._frame_position_cost` |
| Open-string penalty 8 (was bonus 3) | `dev`, 240 tracks | string accuracy 0.610 → 0.664 | `tab_engine.OPEN_STRING_PENALTY` |
| `max_frets` stays 15 | `dev`, 240 tracks (current costs) | only under the earlier cost: 0.610 at 15, 0.604 at 19, 0.589 at 24 | `tab_engine.TabMapper.__init__` |
| Thresholds stay at Basic Pitch defaults | `dev` sweep + fixtures | clean values 0.822 vs 0.775, but lose audible notes on fixtures | `basic_pitch_engine.DEFAULT_*` |
| Absorb phantom re-onsets of ringing notes, 0 dB fundamental rise (`--keep-retriggers` to disable) | 3 distinct annotated takes: onset F1 0.707 → 0.754 (P 0.594 → 0.682, R 0.928 → 0.867), offset F1 0.523 → 0.617; strum take 0.916 → 0.874 (R 0.954 → 0.856). GuitarSet `dev`, 240 tracks: 0.772 → 0.769 | onset F1 0.775 → 0.784 (P 0.776 → 0.812, R 0.801 → 0.775); string accuracy 0.664 → 0.661 | `retrigger.py` |
| Velocity scale 0.8 by default | 3 distinct annotated takes, 318 matched notes: r = 0.86 with engine velocity at 0.81×; mean error 14.6 → 6.9 | n/a (GuitarSet has no velocity labels) | `post_process.DEFAULT_VELOCITY_SCALE` |
| ONNX instead of TensorFlow | all fixtures, note-for-note | not run on held-out players; identical scores on 20 mixed GuitarSet tracks | `basic_pitch_engine.MODEL_PATH` |
