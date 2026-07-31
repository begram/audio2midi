import pytest

from post_process import (
    apply_logarithmic_velocity,
    merge_notes,
    quantize_notes,
    ticks_to_seconds,
)


def test_apply_logarithmic_velocity():
    """Test logarithmic transformation of MIDI velocities."""
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 0},
        {'pitch': 62, 'start': 0.0, 'end': 1.0, 'velocity': 64},
        {'pitch': 64, 'start': 0.0, 'end': 1.0, 'velocity': 127},
    ]

    transformed = apply_logarithmic_velocity(notes, curvature=5.0)

    # 0 maps to 0 (or min 1 if non-zero input)
    assert transformed[0]['velocity'] == 0
    # 127 maps to 127
    assert transformed[2]['velocity'] == 127
    # Mid-range velocity 64 should be boosted logarithmically (> 64)
    assert transformed[1]['velocity'] > 64

def test_quantize_notes_partial_strength():
    """Test quantization with partial strength (e.g. 50%)."""
    # 120 BPM: 1 beat = 0.5s = 480 ticks
    # onset at 0.1s => raw ticks = 96. 1/4 grid snaps to 0 ticks.
    notes = [
        {'pitch': 60, 'start': 0.1, 'end': 0.6, 'velocity': 100},
    ]

    # Full strength (1.0) -> onset ticks = 0
    q_full = quantize_notes(notes.copy(), bpm=120, grid_resolution='1/4', strength=1.0)
    assert q_full[0]['start_ticks'] == 0

    # 50% strength (0.5) -> onset ticks = 96 + 0.5 * (0 - 96) = 48
    q_half = quantize_notes(notes.copy(), bpm=120, grid_resolution='1/4', strength=0.5)
    assert q_half[0]['start_ticks'] == 48

def test_quantize_notes_zero_strength():
    """Test quantization with 0% strength leaves original raw ticks intact."""
    notes = [{'pitch': 60, 'start': 0.1, 'end': 0.6, 'velocity': 100}]
    q_zero = quantize_notes(notes.copy(), bpm=120, grid_resolution='1/4', strength=0.0)
    assert q_zero[0]['start_ticks'] == 96

def test_apply_logarithmic_velocity_disabled():
    """Test disabling logarithmic velocity transformation (curvature=0)."""
    notes = [{'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 64}]
    result = apply_logarithmic_velocity(notes.copy(), curvature=0.0)
    assert result[0]['velocity'] == 64


def test_ticks_to_seconds_applies_quantized_grid():
    """start/end are rewritten from the tick values quantize_notes produced."""
    notes = [{'pitch': 60, 'start': 0.1, 'end': 0.6, 'velocity': 100}]

    quantize_notes(notes, bpm=120, grid_resolution='1/4', strength=1.0)
    ticks_to_seconds(notes, bpm=120)

    # 120 BPM: 480 ticks = 1 quarter = 0.5s
    assert notes[0]['start'] == pytest.approx(0.0)
    assert notes[0]['end'] == pytest.approx(0.5)


def test_ticks_to_seconds_retimes_pitch_bends():
    """Bends follow their note onto the quantized grid instead of drifting off it."""
    notes = [{
        'pitch': 60, 'start': 0.1, 'end': 0.6, 'velocity': 100,
        'pitch_bends': [
            {'pitch': 0, 'time': 0.1},
            {'pitch': 2048, 'time': 0.35},
            {'pitch': 4096, 'time': 0.6},
        ],
    }]

    quantize_notes(notes, bpm=120, grid_resolution='1/4', strength=1.0)
    ticks_to_seconds(notes, bpm=120)

    note = notes[0]
    times = [b['time'] for b in note['pitch_bends']]
    # Endpoints land on the new span, midpoint stays proportional.
    assert times[0] == pytest.approx(note['start'])
    assert times[-1] == pytest.approx(note['end'])
    assert times[1] == pytest.approx(0.25)
    # Every bend stays inside its own note.
    assert all(note['start'] <= t <= note['end'] for t in times)


def test_merge_notes_keeps_absorbed_pitch_bends():
    """Merging must not silently discard the absorbed note's expression."""
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 100,
         'pitch_bends': [{'pitch': 0, 'time': 0.1}]},
        {'pitch': 60, 'start': 0.4, 'end': 1.0, 'velocity': 80,
         'pitch_bends': [{'pitch': 4096, 'time': 0.7}]},
        {'pitch': 62, 'start': 0.1, 'end': 0.6, 'velocity': 100},
    ]

    merged = merge_notes(notes)

    note_60 = [n for n in merged if n['pitch'] == 60][0]
    assert [b['pitch'] for b in note_60['pitch_bends']] == [0, 4096]


def test_merge_notes_empty_input():
    assert merge_notes([]) == []


def test_quantize_notes_empty_input():
    assert quantize_notes([], bpm=120) == []


def test_apply_logarithmic_velocity_empty_input():
    assert apply_logarithmic_velocity([], curvature=5.0) == []


@pytest.mark.parametrize("grid,expected_end_ticks", [
    ('1/4', 480),
    ('1/8', 240),
    ('1/16', 120),
    ('1/32', 60),
])
def test_quantize_notes_grid_resolutions(grid, expected_end_ticks):
    """Each grid choice snaps to its own tick multiple at 120 BPM."""
    notes = [{'pitch': 60, 'start': 0.0, 'end': 0.02, 'velocity': 100}]

    quantize_notes(notes, bpm=120, grid_resolution=grid, strength=1.0)

    # A 20ms note rounds to zero length, so the guard extends it by one grid step.
    assert notes[0]['start_ticks'] == 0
    assert notes[0]['end_ticks'] == expected_end_ticks


def test_quantize_notes_never_produces_zero_length_notes():
    """Notes shorter than the grid must not collapse to zero (or negative) length."""
    notes = [
        {'pitch': 60, 'start': 0.001, 'end': 0.002, 'velocity': 100},
        {'pitch': 62, 'start': 0.51, 'end': 0.52, 'velocity': 100},
    ]

    quantize_notes(notes, bpm=120, grid_resolution='1/4', strength=1.0)

    for n in notes:
        assert n['end_ticks'] > n['start_ticks']


def test_quantize_notes_unknown_grid_defaults_to_sixteenth():
    notes = [{'pitch': 60, 'start': 0.0, 'end': 0.02, 'velocity': 100}]

    quantize_notes(notes, bpm=120, grid_resolution='1/7', strength=1.0)

    assert notes[0]['end_ticks'] == 120


def test_ticks_to_seconds_skips_unquantized_notes():
    """A note without tick fields is left exactly as it was."""
    notes = [{'pitch': 60, 'start': 0.3, 'end': 0.8, 'velocity': 100}]

    ticks_to_seconds(notes, bpm=120)

    assert notes[0]['start'] == 0.3
    assert notes[0]['end'] == 0.8


def test_ticks_to_seconds_handles_zero_span_source_note():
    """A zero-length input span must not divide by zero when re-timing bends."""
    notes = [{
        'pitch': 60, 'start': 0.25, 'end': 0.25, 'velocity': 100,
        'start_ticks': 240, 'end_ticks': 480,
        'pitch_bends': [{'pitch': 1024, 'time': 0.25}],
    }]

    ticks_to_seconds(notes, bpm=120)

    note = notes[0]
    assert note['start'] == pytest.approx(0.25)
    assert note['end'] == pytest.approx(0.5)
    assert note['start'] <= note['pitch_bends'][0]['time'] <= note['end']


# --- merge window boundaries ------------------------------------------------

def _other_pitch_sustaining(start=0.0, end=3.0):
    """A different pitch held across the whole span, enabling merges."""
    return {'pitch': 62, 'start': start, 'end': end, 'velocity': 100}


def test_merge_notes_merges_within_gap_threshold():
    """A same-pitch note starting inside the 50ms gap is absorbed."""
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.50, 'velocity': 100},
        {'pitch': 60, 'start': 0.54, 'end': 1.00, 'velocity': 80},
        _other_pitch_sustaining(),
    ]

    merged = merge_notes(notes)
    pitch_60 = [n for n in merged if n['pitch'] == 60]

    assert len(pitch_60) == 1
    assert pitch_60[0]['end'] == 1.00


def test_merge_notes_respects_gap_threshold():
    """Beyond the 50ms gap it is a genuine re-pluck, not engine chatter."""
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.50, 'velocity': 100},
        {'pitch': 60, 'start': 0.60, 'end': 1.00, 'velocity': 80},
        _other_pitch_sustaining(),
    ]

    merged = merge_notes(notes)

    assert len([n for n in merged if n['pitch'] == 60]) == 2


def test_merge_notes_takes_max_velocity_of_merged_pair():
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 70},
        {'pitch': 60, 'start': 0.4, 'end': 1.0, 'velocity': 110},
        _other_pitch_sustaining(),
    ]

    merged = merge_notes(notes)
    pitch_60 = [n for n in merged if n['pitch'] == 60][0]

    assert pitch_60['velocity'] == 110


def test_merge_notes_requires_other_pitch_to_span_the_onset():
    """The other pitch must still be sounding *at* the second onset.

    Here pitch 62 has already ended before the repeat starts, so the notes are
    left separate. This pins the concurrency window that the merge check
    binary-searches.
    """
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.50, 'velocity': 100},
        {'pitch': 60, 'start': 0.45, 'end': 1.00, 'velocity': 80},
        {'pitch': 62, 'start': 0.0, 'end': 0.20, 'velocity': 100},  # ends too early
    ]

    merged = merge_notes(notes)

    assert len([n for n in merged if n['pitch'] == 60]) == 2


def test_merge_notes_finds_concurrent_pitch_starting_far_earlier():
    """A long sustain starting well before the repeat still counts.

    The lookback window is bounded, so a sustain that began ~1.9s earlier must
    still be found -- this guards the lower bisect bound.
    """
    notes = [
        {'pitch': 60, 'start': 1.90, 'end': 2.40, 'velocity': 100},
        {'pitch': 60, 'start': 2.35, 'end': 2.90, 'velocity': 80},
        {'pitch': 62, 'start': 0.50, 'end': 3.00, 'velocity': 100},
    ]

    merged = merge_notes(notes)

    assert len([n for n in merged if n['pitch'] == 60]) == 1


def test_merge_notes_does_not_mutate_source_bend_lists():
    """The merged note owns its bend list; the input notes are left alone."""
    first_bends = [{'pitch': 0, 'time': 0.1}]
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 100, 'pitch_bends': first_bends},
        {'pitch': 60, 'start': 0.4, 'end': 1.0, 'velocity': 80,
         'pitch_bends': [{'pitch': 4096, 'time': 0.7}]},
        {'pitch': 62, 'start': 0.1, 'end': 0.6, 'velocity': 100},
    ]

    merge_notes(notes)

    assert len(first_bends) == 1

