"""Absorbs Basic Pitch's phantom re-onsets of strings that are still ringing.

When a new string is plucked, the broadband attack often makes Basic Pitch cut every
note that is still sounding and restart it at the same pitch. In the MIDI file that is
a re-pluck that never happened. On the hand-corrected references of the project's own
recordings (`tests/references/`), these phantom re-onsets were about 60% of all spurious
notes under the default thresholds.

Nothing in the note list tells a phantom from a genuine re-pluck: both follow their
predecessor with no gap, and their velocities overlap almost completely (a rule on the
velocity ratio gained at most 0.003 onset F1). The audio does tell them apart: a real
pluck adds energy at the note's own fundamental, while a phantom sits on a fundamental
that keeps decaying while a *different* string supplies the attack. Measured over the
candidate re-onsets in those recordings, the fundamental's level rose by a median 5.4 dB
across real re-plucks and -0.2 dB across phantoms.

The cost is recall: a genuine re-pluck that lands no louder than the still-ringing note
(typically a re-strum of the same chord) is absorbed too. Measurements and the decision
are in `specs/roadmap.md`, appendix.
"""

import librosa
import numpy as np

# A same-pitch note starting within this long of its predecessor's end is a continuation
# candidate. Basic Pitch's split leaves the two touching (gap 0) or one ~11.6 ms frame
# apart; 15 ms covers both without reaching notes that genuinely stopped.
MAX_GAP_SECONDS = 0.015

# Absorb a candidate whose fundamental rises by less than this. Swept from -2 to +3 dB on
# the three distinct annotated takes and 20 GuitarSet dev tracks: 0 dB scored best on the
# takes (mean onset F1 0.707 -> 0.754) and was neutral on GuitarSet. Higher values absorb
# more real re-plucks than phantoms.
MIN_RISE_DB = 0.0

# Level windows around the onset frame f, in analysis frames (256 samples, ~11.6 ms at
# 22.05 kHz): the mean over frames f-6..f-2 (before) against the peak over f+1..f+5
# (after), about 58 ms each. Frames f-1 and f are skipped: the other string's attack
# transient smears into every bin there.
_HOP = 256
_PRE_FRAMES = (6, 1)   # slice [f-6, f-1)
_POST_FRAMES = (1, 6)  # slice [f+1, f+6)
# Three bins per semitone, with the peak taken over the nearest bin and its neighbours,
# so a slightly sharp or flat string still reads at its own pitch.
_BINS_PER_SEMITONE = 3
_LOWEST_MIDI = 24
_N_SEMITONES = 96
# CQT magnitude below which a band counts as silent (normalised audio reads ~1e-2 and up).
_SILENT = 1e-6


def _fundamental_spectrogram(audio, sr):
    return np.abs(librosa.cqt(
        audio, sr=sr, hop_length=_HOP, fmin=librosa.midi_to_hz(_LOWEST_MIDI),
        n_bins=_N_SEMITONES * _BINS_PER_SEMITONE, bins_per_octave=12 * _BINS_PER_SEMITONE,
    ))


def fundamental_rise_db(spec, sr, pitch, time):
    """How much the level at `pitch`'s fundamental rises across `time`, in dB."""
    centre = int(round((pitch - _LOWEST_MIDI) * _BINS_PER_SEMITONE))
    if not 1 <= centre < spec.shape[0] - 1:
        # Out of the analysed range: report a rise, so the note is kept as detected.
        return np.inf
    band = spec[centre - 1:centre + 2].max(axis=0)
    frame = int(librosa.time_to_frames(time, sr=sr, hop_length=_HOP))
    pre = band[max(0, frame - _PRE_FRAMES[0]):max(0, frame - _PRE_FRAMES[1])]
    post = band[frame + _POST_FRAMES[0]:frame + _POST_FRAMES[1]]
    if pre.size == 0 or post.size == 0 or pre.mean() < _SILENT:
        # No level to compare against (too close to either end of the audio), or nothing
        # was ringing at this pitch: no evidence of a phantom, so the note is kept.
        return np.inf
    return float(20.0 * np.log10((post.max() + 1e-9) / pre.mean()))


def absorb_ringing_retriggers(notes, audio, sr, min_rise_db=MIN_RISE_DB,
                              max_gap=MAX_GAP_SECONDS):
    """Folds phantom same-pitch re-onsets back into the note that was still ringing.

    The surviving note keeps its own onset and velocity (those are the real pluck's) and
    takes the later end. Its bend contour is extended by the absorbed note's, as in
    `merge_notes`, so expression on the continuation is not dropped. Returns new dicts
    sorted by onset; the input is not modified.
    """
    ordered = sorted((dict(n) for n in notes), key=lambda n: n['start'])
    candidates = []
    last_by_pitch = {}
    for i, n in enumerate(ordered):
        prev = last_by_pitch.get(n['pitch'])
        if prev is not None and n['start'] - ordered[prev]['end'] <= max_gap:
            candidates.append(i)
        last_by_pitch[n['pitch']] = i
    if not candidates:
        return ordered

    spec = _fundamental_spectrogram(np.asarray(audio, dtype=np.float32), sr)
    phantom = {i for i in candidates
               if fundamental_rise_db(spec, sr, ordered[i]['pitch'], ordered[i]['start'])
               < min_rise_db}

    kept = []
    owner = {}  # pitch -> the kept note the next phantom of that pitch extends
    for i, n in enumerate(ordered):
        if i in phantom and n['pitch'] in owner:
            target = owner[n['pitch']]
            target['end'] = max(target['end'], n['end'])
            if n.get('pitch_bends'):
                target['pitch_bends'] = list(target.get('pitch_bends', [])) + list(n['pitch_bends'])
            continue
        kept.append(n)
        owner[n['pitch']] = n
    return kept
