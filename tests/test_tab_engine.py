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
    """Seven simultaneous notes cannot fit six strings, so exactly one goes unassigned.

    Taking each note's first placement (the previous fallback) piled four notes
    onto string 6 while strings 1-5 sat free, so the matching still has to fill all
    six. The seventh voice cannot be placed and no earlier note can be truncated to
    make room, so it gives up its string rather than colliding on one.
    """
    notes = [
        {'pitch': p, 'start': 0.0, 'end': 1.0, 'velocity': 100}
        for p in [40, 45, 50, 55, 59, 64, 67]
    ]

    assigned = TabMapper().assign_strings(notes)

    used = {n['string'] for n in assigned if n['string'] > 0}
    assert len(used) == 6, f"all six strings should be used, got {sorted(used)}"
    assert _collisions(assigned) == 0, "no string may carry two simultaneous notes"
    assert sum(n['string'] == 0 for n in assigned) == 1
    assert len(assigned) == 7, "the unplaceable note is kept, only its string is dropped"


def test_forced_duplicate_placement_gives_the_string_to_one_note():
    """Duplicate low E: pitch 40 has a single placement, so the two notes conflict.

    The placement is forced, but both notes cannot hold string 6 at once. They are
    1ms apart -- far too close to be a re-pluck -- so the earlier one keeps its pitch
    and length while giving up the string.
    """
    mapper = TabMapper()
    assert mapper.get_possible_placements(40) == [(6, 0)]

    notes = [
        {'pitch': 40, 'start': 0.000, 'end': 1.0, 'velocity': 100},
        {'pitch': 40, 'start': 0.001, 'end': 0.9, 'velocity': 80},
    ]

    first, second = mapper.assign_strings(notes)

    assert (first['string'], first['fret']) == (0, -1)
    assert (second['string'], second['fret']) == (6, 0)
    assert first['end'] == pytest.approx(1.0), "length is kept; only the string is given up"
    assert _string_overlaps([first, second]) == 0


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


def _string_overlaps(notes):
    """Same-string notes that sound at the same time -- physically impossible.

    `_collisions` above ignores time, so it only catches notes within one chord
    frame; a sustain running underneath a later note on its own string needs this.
    """
    by_string = {}
    for note in notes:
        if note['string'] > 0:
            by_string.setdefault(note['string'], []).append(note)

    overlaps = 0
    for string_notes in by_string.values():
        string_notes.sort(key=lambda n: n['start'])
        for prev, nxt in zip(string_notes, string_notes[1:], strict=False):
            if nxt['start'] < prev['end'] - 1e-9:
                overlaps += 1
    return overlaps


def test_sustained_note_is_truncated_by_a_later_pluck_on_the_same_string():
    """E2 has a single placement, so both notes must take string 6.

    A string sounds one note at a time: the second pluck stops the first note. The
    two onsets are a second apart, so they land in different chord frames and the
    candidate collision guard never sees the pair.
    """
    notes = [
        {'pitch': 40, 'start': 0.0, 'end': 2.0, 'velocity': 100},
        {'pitch': 40, 'start': 1.0, 'end': 3.0, 'velocity': 100},
    ]

    first, second = TabMapper().assign_strings(notes)

    assert [first['string'], second['string']] == [6, 6]
    assert first['end'] == pytest.approx(1.0), "the sustain must stop at the new pluck"
    assert second['end'] == pytest.approx(3.0), "the later note keeps its full length"
    assert _string_overlaps([first, second]) == 0


def test_sustained_chord_under_a_melody_leaves_no_string_overlaps():
    """The case that occurs in real transcriptions: a held voicing plus a melody."""
    chord = [
        {'pitch': p, 'start': 0.0, 'end': 4.0, 'velocity': 100}
        for p in [40, 45, 50, 55, 59, 64]
    ]
    melody = [
        {'pitch': p, 'start': s, 'end': s + 2.0, 'velocity': 90}
        for p, s in [(52, 1.0), (57, 1.5), (62, 2.0), (67, 2.5)]
    ]

    assigned = TabMapper().assign_strings(chord + melody)

    assert _string_overlaps(assigned) == 0
    assert len(assigned) == len(chord) + len(melody), "no note may be discarded"


def test_notes_on_distinct_strings_keep_their_full_duration():
    """Truncation must only fire on a genuine same-string conflict."""
    notes = [
        {'pitch': 40, 'start': 0.0, 'end': 2.0, 'velocity': 100},
        {'pitch': 64, 'start': 1.0, 'end': 2.0, 'velocity': 100},
    ]

    low, high = TabMapper().assign_strings(notes)

    assert low['string'] != high['string']
    assert low['end'] == pytest.approx(2.0)
    assert high['end'] == pytest.approx(2.0)


def test_truncation_drops_pitch_bends_past_the_new_note_end():
    """A bend surviving past the truncation point would bleed into the next note.

    generate_midi recentres the channel at each note's end, so a later bend event
    would be written after that reset.
    """
    notes = [
        {
            'pitch': 40, 'start': 0.0, 'end': 2.0, 'velocity': 100,
            'pitch_bends': [
                {'pitch': 0, 'time': 0.0},
                {'pitch': 400, 'time': 0.5},
                {'pitch': 900, 'time': 1.5},
            ],
        },
        {'pitch': 40, 'start': 1.0, 'end': 3.0, 'velocity': 100},
    ]

    first, _second = TabMapper().assign_strings(notes)

    assert [b['time'] for b in first['pitch_bends']] == [0.0, 0.5]


def test_near_simultaneous_conflict_is_not_truncated_to_a_sliver():
    """A 20ms gap is below any physical re-pluck, so truncating would emit a click.

    Both notes need string 6, and they fall in different chord frames (the grouping
    window is 15ms), so this is the pair the frame guard cannot see. Shortening the
    first to 20ms would produce an unplayable sliver rather than a note.
    """
    notes = [
        {'pitch': 40, 'start': 0.00, 'end': 2.0, 'velocity': 100},
        {'pitch': 40, 'start': 0.02, 'end': 1.0, 'velocity': 100},
    ]

    first, second = TabMapper().assign_strings(notes)

    assert first['string'] == 0, "the earlier note gives up the string instead"
    assert first['end'] == pytest.approx(2.0), "and keeps its full length"
    assert second['string'] == 6
    assert _string_overlaps([first, second]) == 0


def test_unplaceable_simultaneous_voice_gives_up_its_string():
    """Two E2s at one instant cannot both sound, and neither can be truncated.

    The note keeps its pitch and velocity but loses its string assignment, so
    midi_gen routes it to the unassigned track rather than colliding on a channel.
    """
    notes = [
        {'pitch': 40, 'start': 0.0, 'end': 1.0, 'velocity': 100},
        {'pitch': 40, 'start': 0.0, 'end': 1.0, 'velocity': 90},
    ]

    assigned = TabMapper().assign_strings(notes)

    assert len(assigned) == 2, "the note is kept, only its string is given up"
    assert sorted(n['string'] for n in assigned) == [0, 6]
    unassigned = next(n for n in assigned if n['string'] == 0)
    assert unassigned['fret'] == -1
    assert unassigned['end'] == pytest.approx(1.0), "duration must not be zeroed"
    assert _string_overlaps(assigned) == 0
