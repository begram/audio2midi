import pytest

from tab_engine import TabMapper


def test_tab_mapper_chord():
    """Test that a C Major chord is assigned to unique strings."""
    # C Major: C3(48), E3(52), G3(55), C4(60), E4(64)
    notes = [
        {'pitch': 48, 'start': 0.0, 'end': 1.0, 'velocity': 100},
        {'pitch': 52, 'start': 0.0, 'end': 1.0, 'velocity': 100},
        {'pitch': 55, 'start': 0.0, 'end': 1.0, 'velocity': 100},
        {'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 100},
        {'pitch': 64, 'start': 0.0, 'end': 1.0, 'velocity': 100},
    ]

    mapper = TabMapper()
    assigned = mapper.assign_strings(notes)

    strings = [n['string'] for n in assigned]
    # Check that all strings are unique (polyphony constraint)
    assert len(set(strings)) == len(notes)
    # All strings should be between 1 and 6
    for s in strings:
        assert 1 <= s <= 6

def test_tab_mapper_box_constraint():
    """Test that notes are kept within a reasonable fret range."""
    # Play a scale or sequence that could be played in different positions
    # A4(69), B4(71), C#5(73)
    # Position 1: String 1, frets 5, 7, 9 (Span 4)
    # Position 2: String 2, frets 10, 12, 14 (Span 4)
    notes = [
        {'pitch': 69, 'start': 0.0, 'end': 0.5, 'velocity': 100},
        {'pitch': 71, 'start': 0.6, 'end': 1.1, 'velocity': 100},
        {'pitch': 73, 'start': 1.2, 'end': 1.7, 'velocity': 100},
    ]

    mapper = TabMapper()
    assigned = mapper.assign_strings(notes)

    frets = [n['fret'] for n in assigned]
    fret_min = min(frets)
    fret_max = max(frets)

    # Span should be small
    assert (fret_max - fret_min) <= 5

def test_tab_mapper_out_of_range():
    """Test handling of notes out of guitar range."""
    notes = [{'pitch': 20, 'start': 0.0, 'end': 1.0, 'velocity': 100}]
    mapper = TabMapper()
    assigned = mapper.assign_strings(notes)
    assert assigned[0]['string'] == 0
    assert assigned[0]['fret'] == -1


def test_position_preference_does_not_decay_over_time():
    """Emission cost applies at every frame, not just the first.

    A long run of A4s is playable at fret 5 (string 1), 10 (string 2) or 14
    (string 3). With position cost only anchoring frame 0, later frames drift;
    with per-frame emission cost every one stays near the preferred position.
    """
    notes = [
        {'pitch': 69, 'start': i * 0.5, 'end': i * 0.5 + 0.4, 'velocity': 100}
        for i in range(40)
    ]

    assigned = TabMapper().assign_strings(notes)

    assert all(n['fret'] == 5 for n in assigned)
    assert all(n['string'] == 1 for n in assigned)


def test_open_string_bonus_does_not_scale_with_polyphony():
    """A six-note open chord must not out-score everything via summed bonuses."""
    mapper = TabMapper()

    single_open = ((6, 0),)
    six_open = ((6, 0), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0))

    # Averaged, not summed: both frames get the same per-note bonus.
    assert mapper._frame_position_cost(single_open) == pytest.approx(
        mapper._frame_position_cost(six_open)
    )


def test_frame_candidates_are_capped():
    """Dense frames cannot blow up the |prev| x |curr| Viterbi step."""
    mapper = TabMapper(max_candidates=8)
    frame = [
        {'pitch': p, 'start': 0.0, 'end': 1.0, 'velocity': 100}
        for p in [55, 57, 59, 60, 62, 64]
    ]

    assert len(mapper._generate_frame_candidates(frame)) <= 8


def test_assign_strings_empty_input():
    assert TabMapper().assign_strings([]) == []


# --- fallback when no collision-free assignment exists ----------------------

def _collisions(notes):
    used = [n['string'] for n in notes if n['string'] > 0]
    return len(used) - len(set(used))


def test_fallback_minimizes_collisions_for_oversized_frame():
    """Seven simultaneous notes cannot fit six strings, but only one may collide.

    Taking each note's first placement (the previous fallback) piled four notes
    onto string 6 while strings 1-5 sat free.
    """
    notes = [
        {'pitch': p, 'start': 0.0, 'end': 1.0, 'velocity': 100}
        for p in [40, 45, 50, 55, 59, 64, 67]
    ]

    assigned = TabMapper().assign_strings(notes)

    used = {n['string'] for n in assigned if n['string'] > 0}
    assert len(used) == 6, f"all six strings should be used, got {sorted(used)}"
    assert _collisions(assigned) == 1, "exactly one collision is unavoidable here"


def test_fallback_collides_only_when_physically_forced():
    """Duplicate low E: pitch 40 has a single placement, so a collision is forced."""
    mapper = TabMapper()
    assert mapper.get_possible_placements(40) == [(6, 0)]

    notes = [
        {'pitch': 40, 'start': 0.000, 'end': 1.0, 'velocity': 100},
        {'pitch': 40, 'start': 0.001, 'end': 0.9, 'velocity': 80},
    ]

    assigned = mapper.assign_strings(notes)

    assert [n['string'] for n in assigned] == [6, 6]
    assert [n['fret'] for n in assigned] == [0, 0]


def test_fallback_does_not_displace_a_single_option_note():
    """A note with one placement keeps it; the flexible note moves instead."""
    mapper = TabMapper()
    # 40 -> only (6, 0). 45 -> (6, 5) and (5, 0).
    assert mapper.get_possible_placements(40) == [(6, 0)]
    assert (5, 0) in mapper.get_possible_placements(45)

    notes = [
        {'pitch': 45, 'start': 0.000, 'end': 1.0, 'velocity': 100},
        {'pitch': 40, 'start': 0.001, 'end': 1.0, 'velocity': 100},
    ]

    assigned = mapper.assign_strings(notes)
    by_pitch = {n['pitch']: n for n in assigned}

    assert by_pitch[40]['string'] == 6
    assert by_pitch[45]['string'] == 5
    assert _collisions(assigned) == 0


def test_six_note_chord_has_no_collisions():
    """A full open-position six-note voicing still resolves cleanly."""
    notes = [
        {'pitch': p, 'start': 0.0, 'end': 1.0, 'velocity': 100}
        for p in [40, 45, 50, 55, 59, 64]
    ]

    assigned = TabMapper().assign_strings(notes)

    assert sorted(n['string'] for n in assigned) == [1, 2, 3, 4, 5, 6]
    assert _collisions(assigned) == 0


def test_transition_cost_is_zero_when_a_frame_is_all_open():
    """No fretted note on either side means no hand travel to charge for."""
    mapper = TabMapper()
    all_open = ((6, 0), (5, 0))
    fretted = ((6, 3), (5, 2))

    assert mapper._transition_cost(all_open, fretted) == 0.0
    assert mapper._transition_cost(fretted, all_open) == 0.0
    assert mapper._transition_cost(fretted, fretted) == 0.0


def test_out_of_range_pitch_costs_nothing():
    """An unplayable (0, -1) placement must not skew position or bonus terms."""
    assert TabMapper()._frame_position_cost(((0, -1),)) == 0.0


def test_empty_combo_costs_nothing():
    """The guard against dividing by an empty frame size."""
    assert TabMapper()._frame_position_cost(()) == 0.0
