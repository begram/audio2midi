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
