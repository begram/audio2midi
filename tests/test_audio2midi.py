import pytest
import numpy as np
import os
import pretty_midi
from processor import normalize_audio, high_pass_filter, noise_gate
from post_process import clean_notes, quantize_notes, merge_notes
from midi_gen import generate_midi

def test_noise_gate():
    """Test that samples below the threshold are zeroed out."""
    audio = np.array([0.001, 0.01, -0.002, -0.05], dtype=np.float32)
    gated = noise_gate(audio, threshold=0.005)
    expected = np.array([0.0, 0.01, 0.0, -0.05], dtype=np.float32)
    np.testing.assert_array_equal(gated, expected)

def test_merge_notes():
    """Test that overlapping notes of the same pitch are merged ONLY if another pitch is active."""
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 100},
        {'pitch': 60, 'start': 0.4, 'end': 1.0, 'velocity': 80}, # Overlaps with previous
        {'pitch': 62, 'start': 0.1, 'end': 0.6, 'velocity': 100}, # Active during overlap
    ]
    merged = merge_notes(notes)
    # Pitch 60 (2 notes merged) + Pitch 62 = 2 notes total
    assert len(merged) == 2

    note_60 = [n for n in merged if n['pitch'] == 60][0]
    assert note_60['start'] == 0.0
    assert note_60['end'] == 1.0

def test_no_merge_without_other_pitch():
    """Test that overlapping notes are NOT merged if no other pitch is active."""
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 100},
        {'pitch': 60, 'start': 0.4, 'end': 1.0, 'velocity': 80}, # Overlaps, but no other pitch active
    ]
    merged = merge_notes(notes)
    # Should NOT merge, so 2 notes total
    assert len(merged) == 2


def test_normalize_audio():
    """Test that normalization scales audio to -1.0 dB peak."""
    audio = np.array([0.1, 0.5, -0.2], dtype=np.float32)
    normalized = normalize_audio(audio, target_db=-1.0)
    max_val = np.max(np.abs(normalized))
    assert np.isclose(max_val, 10**(-1.0 / 20), atol=1e-5)

def test_clean_notes():
    """Test filtering of short and quiet notes."""
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.1, 'velocity': 100}, 
        {'pitch': 62, 'start': 0.2, 'end': 0.21, 'velocity': 100},
        {'pitch': 64, 'start': 0.4, 'end': 0.5, 'velocity': 5}
    ]
    cleaned = clean_notes(notes, min_duration=0.030, velocity_threshold=10)
    assert len(cleaned) == 1
    assert cleaned[0]['pitch'] == 60

def test_quantize_notes():
    """Test snapping to a 1/4 note grid at 120 BPM."""
    notes = [
        {'pitch': 60, 'start': 0.1, 'end': 0.6, 'velocity': 100},
    ]
    quantized = quantize_notes(notes, bpm=120, grid_resolution='1/4')
    assert quantized[0]['start_ticks'] == 0
    assert quantized[0]['end_ticks'] == 480

def test_midi_generation(tmp_path):
    """Test that a valid MIDI file is generated with correct BPM."""
    midi_path = os.path.join(tmp_path, "test.mid")
    notes = [{'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 100}]
    generate_midi(notes, bpm=140, output_path=midi_path)
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    assert len(pm.instruments) == 1
    # Use approx with larger tolerance
    assert pm.get_tempo_changes()[1][0] == pytest.approx(140, rel=1e-3)
    assert len(pm.instruments[0].notes) == 1
    assert pm.instruments[0].notes[0].pitch == 60

def test_synthetic_audio_processing():
    """Test high-pass filter on synthetic audio."""
    sr = 22050
    t = np.linspace(0, 1, sr)
    audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    filtered = high_pass_filter(audio, sr, cutoff=80)
    assert filtered.shape == audio.shape