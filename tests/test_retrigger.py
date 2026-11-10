"""Tests for absorbing phantom re-onsets of still-ringing notes.

The audio is synthesised so the deciding signal is known: a decaying sine at the note's
pitch, with or without a fresh attack at the re-onset, plus a second string starting there.
"""

import librosa
import numpy as np
import pytest

from retrigger import absorb_ringing_retriggers, fundamental_rise_db

SR = 22050
A3, E4 = 57, 64


def _string(pitch, start, dur, sr=SR, amp=0.5, decay=1.5):
    t = np.arange(int(dur * sr)) / sr
    tone = amp * np.exp(-decay * t) * np.sin(2 * np.pi * librosa.midi_to_hz(pitch) * t)
    out = np.zeros(int((start + dur) * sr) + 1, dtype=np.float32)
    out[int(start * sr):int(start * sr) + len(tone)] = tone
    return out


def _mix(*parts):
    out = np.zeros(max(len(p) for p in parts), dtype=np.float32)
    for p in parts:
        out[:len(p)] += p
    return out


def _note(pitch, start, end, velocity=80, **extra):
    return {'pitch': pitch, 'start': start, 'end': end, 'velocity': velocity, **extra}


@pytest.fixture
def ringing_then_other_string():
    """A3 plucked at 0 s and left ringing; E4 plucked at 1 s. A3 is not re-plucked."""
    return _mix(_string(A3, 0.0, 2.0), _string(E4, 1.0, 1.0))


@pytest.fixture
def replucked():
    """A3 plucked at 0 s and plucked again, harder, at 1 s, with E4 alongside."""
    return _mix(_string(A3, 0.0, 1.0, amp=0.2), _string(A3, 1.0, 1.0, amp=0.6), _string(E4, 1.0, 1.0))


def test_phantom_reonset_is_absorbed_into_the_ringing_note(ringing_then_other_string):
    notes = [_note(A3, 0.0, 1.0, 90), _note(A3, 1.0, 2.0, 70), _note(E4, 1.0, 2.0)]
    out = absorb_ringing_retriggers(notes, ringing_then_other_string, SR)

    a3 = [n for n in out if n['pitch'] == A3]
    assert len(a3) == 1
    # The real pluck's onset and velocity survive; the end is the phantom's.
    assert a3[0]['start'] == 0.0 and a3[0]['end'] == 2.0 and a3[0]['velocity'] == 90
    assert any(n['pitch'] == E4 for n in out)


def test_real_repluck_is_kept(replucked):
    notes = [_note(A3, 0.0, 1.0), _note(A3, 1.0, 2.0), _note(E4, 1.0, 2.0)]
    out = absorb_ringing_retriggers(notes, replucked, SR)
    assert [n['start'] for n in out if n['pitch'] == A3] == [0.0, 1.0]


def test_rise_measure_separates_the_two_cases(ringing_then_other_string, replucked):
    def rise(audio):
        spec = np.abs(librosa.cqt(audio, sr=SR, hop_length=256, fmin=librosa.midi_to_hz(24),
                                  n_bins=288, bins_per_octave=36))
        return fundamental_rise_db(spec, SR, A3, 1.0)
    assert rise(ringing_then_other_string) < 0.0
    assert rise(replucked) > 3.0


def test_same_pitch_after_a_gap_is_never_a_candidate(ringing_then_other_string):
    # The note stopped 200 ms earlier: a genuine new note whatever the audio says.
    notes = [_note(A3, 0.0, 0.8), _note(A3, 1.0, 2.0)]
    out = absorb_ringing_retriggers(notes, ringing_then_other_string, SR)
    assert len(out) == 2


def test_silent_fundamental_keeps_the_note():
    # No evidence of a ringing note is not evidence of a phantom.
    silent = np.zeros(SR * 2, dtype=np.float32)
    notes = [_note(A3, 0.0, 1.0), _note(A3, 1.0, 2.0)]
    assert len(absorb_ringing_retriggers(notes, silent, SR)) == 2


def test_chain_of_phantoms_collapses_into_one_note():
    audio = _mix(_string(A3, 0.0, 3.0, decay=0.8), _string(E4, 1.0, 0.5), _string(E4, 2.0, 1.0))
    notes = [_note(A3, 0.0, 1.0), _note(A3, 1.0, 2.0), _note(A3, 2.0, 3.0),
             _note(E4, 1.0, 1.5), _note(E4, 2.0, 3.0)]
    a3 = [n for n in absorb_ringing_retriggers(notes, audio, SR) if n['pitch'] == A3]
    assert [(n['start'], n['end']) for n in a3] == [(0.0, 3.0)]


def test_absorbed_bend_contour_is_kept(ringing_then_other_string):
    first = _note(A3, 0.0, 1.0, pitch_bends=[{'pitch': 100, 'time': 0.5}])
    second = _note(A3, 1.0, 2.0, pitch_bends=[{'pitch': 200, 'time': 1.5}])
    out = absorb_ringing_retriggers([first, second], ringing_then_other_string, SR)
    assert out[0]['pitch_bends'] == [{'pitch': 100, 'time': 0.5}, {'pitch': 200, 'time': 1.5}]


def test_input_is_not_modified(ringing_then_other_string):
    notes = [_note(A3, 0.0, 1.0), _note(A3, 1.0, 2.0)]
    absorb_ringing_retriggers(notes, ringing_then_other_string, SR)
    assert notes[0]['end'] == 1.0


def test_no_candidates_skips_the_analysis():
    # Empty audio would fail the CQT; with nothing to check it must not be computed.
    notes = [_note(A3, 0.0, 1.0), _note(E4, 0.5, 1.0)]
    out = absorb_ringing_retriggers(notes, np.zeros(0, dtype=np.float32), SR)
    assert [n['pitch'] for n in out] == [A3, E4]


def test_out_of_range_pitch_is_kept(ringing_then_other_string):
    notes = [_note(127, 0.0, 1.0), _note(127, 1.0, 2.0)]
    assert len(absorb_ringing_retriggers(notes, ringing_then_other_string, SR)) == 2


def test_onset_too_close_to_the_start_has_no_before_window():
    # At 10 ms there is no frame before the skipped ones to compare against: keep the note.
    audio = _string(A3, 0.0, 1.0)
    spec = np.abs(librosa.cqt(audio, sr=SR, hop_length=256, fmin=librosa.midi_to_hz(24),
                              n_bins=288, bins_per_octave=36))
    assert fundamental_rise_db(spec, SR, A3, 0.01) == np.inf
