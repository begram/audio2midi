import pytest
import numpy as np
from post_process import apply_logarithmic_velocity, quantize_notes, clean_notes, merge_notes

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

