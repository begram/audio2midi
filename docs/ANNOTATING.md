# Annotating Your Own Recordings

A step-by-step checklist for making the hand-corrected reference transcriptions that
roadmap item **A2** needs. You can stop after any recording and continue another day; every
step can be repeated safely.

## Why this matters

The detection thresholds (how readily a sound counts as a note) were tuned on GuitarSet, a
dataset recorded under controlled conditions. There, stricter thresholds are clearly better.
On the recordings in `tests/`, they seemed to lose real notes, but that could not be
measured, because nobody had written down which notes are really there. Your references make
it measurable, and they decide which defaults suit real recordings.

## What you need

- The repository, with the venv set up (see the README).
- A DAW or MIDI editor that can show a `.wav` and a `.mid` together (Reaper, Ableton,
  Logic, Cubase, MuseScore with audio, etc.).
- Headphones, ideally. You are correcting by ear.
- About 15–30 minutes per recording, depending on its length and density.

## Files

| File | What it is | In git? |
| :--- | :--- | :--- |
| `tests/<name>.wav` | The recording | yes |
| `tests/references/<name>.draft.mid` | The tool's transcription, a starting point | **no** (regenerate any time) |
| `tests/references/<name>.mid` | **Your corrected reference**, the ground truth | **yes** — commit it |

`<name>` must match exactly: `tests/Fingerpick_mono_44-16.wav` needs
`tests/references/Fingerpick_mono_44-16.mid`.

## Step 1 — Get the drafts

```powershell
.\venv\Scripts\python.exe tools\evaluate_recordings.py --write-drafts
```

This writes a draft for every `.wav` in `tests/` that has no reference yet. It never touches
a finished reference, and it never replaces a draft you have already started on (pass
`--overwrite-drafts` only if you want a fresh draft).

Each draft contains every note found by *either* of the two settings being compared, so it
deliberately has some extra notes. That keeps the draft from favouring one setting, and it
means most of your corrections will be deletions.

**Adding your own takes** is welcome and makes the result more trustworthy. Copy the `.wav`
into `tests/` and rerun the command above. Aim for 5–10 recordings in total, covering
fingerpicked, strummed and picked playing, including at least one noisy or quiet take.
Short is fine: 15–30 seconds each.

## Step 2 — Line the draft up with the audio (do not skip)

Drafts are written at **100 BPM**. Many DAWs import a MIDI file into the project's current
tempo and silently stretch it, which would shift every note away from the audio.

1. Create a new project and set its tempo to **100 BPM** (or choose "import tempo map from
   MIDI file" if your DAW offers it).
2. Import the `.wav` at the very start (time 0).
3. Import the `.draft.mid` at the very start (time 0).
4. Play a few seconds. The draft's notes should start exactly with the plucks you hear,
   all the way through the recording. If they drift apart over time, the tempo is wrong; if
   they are shifted by a constant amount, one of the two did not start at time 0.

## Step 3 — Correct by ear

Go through the recording slowly, a few seconds at a time, and listen.

- **Delete** every note you cannot actually hear being played: echoes, harmonics, noise.
- **Add** every note that is played but missing from the draft, including quiet ones and the
  inner notes of chords.
- **Fix wrong pitches**, octave errors especially (a note an octave too high or too low).
- **Move onsets** so each note starts where the pluck is. This matters most: notes are
  matched within 50 ms.
- **Note ends** (where each note stops sounding) are scored separately and are less
  important. Fix them if you can; don't agonise over them.
- **Velocity, track names and channels don't matter.** You can merge everything onto one
  track or leave it split.

The important rule: **trust your ears, not the draft.** The draft comes from the very tool
being measured. If you only fix what looks wrong, its mistakes become part of the "truth" and
the measurement says it is better than it is. Listen especially for notes the draft has
*missed*: those are the ones that are easy to overlook, because nothing on screen marks them. When unsure whether a note is really there,
loop the passage, slow it down if your DAW can, and decide by listening.

## Step 4 — Save the reference

Export as a standard MIDI file (type 0 or 1 are both fine), at the **same tempo** you
worked at, to:

```
tests/references/<name>.mid
```

Make sure the export starts at time 0 and does not trim leading silence.

## Step 5 — Check it

```powershell
.\venv\Scripts\python.exe tools\evaluate_recordings.py
```

This lists which recordings still lack a reference, then scores the ones that have one. For a
quick sanity check, look at `onset_f` for your new recording. Since the reference started
from the tool's own draft, both settings should score fairly high:

- **Above about 0.5 for both settings:** plausible.
- **Below about 0.3 for both:** almost certainly an alignment problem, not a bad
  transcription. Redo step 2 and check that the export kept the tempo and started at time 0.

## Step 6 — Commit your references

```powershell
git add tests/references
git commit -m "test: add hand-corrected reference for <name>"
```

Adding the folder picks up only the references; git ignores the drafts in it automatically.
Check with `git status` that no `.draft.mid` file is listed.

## When you are done

Once there are at least 5 references, ask Claude to "score the recordings and decide the
detection defaults" (roadmap A2). It will run:

```powershell
.\venv\Scripts\python.exe tools\evaluate_recordings.py
```

and compare the two settings per recording and on average:

- `baseline`: the current defaults (onset 0.5, frame 0.3, 127.7 ms)
- `tuned_clean`: the clean-recording values (onset 0.6, frame 0.5, 58 ms)

The rule, fixed in advance so the result cannot be argued into a preferred answer:

1. If one setting's mean `onset_f` is higher by **more than 0.02**, it wins.
2. Otherwise they count as tied, and the one with the higher mean `onset_recall` wins, since
   a missed note is more work to fix than an extra one.
3. If there are fewer than 5 references, no decision is made yet.

The result and the decision go into the code (`basic_pitch_engine.py`) and the roadmap's
decision record.
