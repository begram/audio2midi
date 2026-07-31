import os

import numpy as np
import pretty_midi
import pytest

from midi_gen import generate_midi
from post_process import clean_notes, merge_notes, quantize_notes
from processor import high_pass_filter, noise_gate, normalize_audio


def test_noise_gate_silences_quiet_passages():
    """The gate attenuates passages whose RMS falls below the threshold."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    signal[sr // 2:] *= 0.0005  # second half well below the threshold

    gated = noise_gate(signal, threshold=0.01)

    loud_rms = np.sqrt(np.mean(gated[: sr // 4] ** 2))
    quiet_rms = np.sqrt(np.mean(gated[-sr // 4:] ** 2))
    assert loud_rms > 0.3
    assert quiet_rms < 1e-4


def test_noise_gate_does_not_clip_waveform():
    """A passage the gate lets through must pass without sample-wise zeroing.

    Zeroing individual samples clips the waveform at every zero crossing; the
    resulting discontinuities are what generate phantom notes downstream.
    """
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    gated = noise_gate(tone, threshold=0.01)

    # The tone has its own exact-zero samples; the gate must not add any.
    assert np.sum(np.abs(gated) < 1e-6) == np.sum(np.abs(tone) < 1e-6)
    # The passage is above threshold, so it should survive essentially intact.
    assert np.corrcoef(tone, gated)[0, 1] > 0.999


def test_noise_gate_preserves_leading_attack():
    """Envelope smoothing must not fade in the start of the file."""
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    gated = noise_gate(tone, threshold=0.01)

    head_rms = np.sqrt(np.mean(gated[:512] ** 2))
    body_rms = np.sqrt(np.mean(gated[sr // 8: sr // 8 + 512] ** 2))
    assert head_rms == pytest.approx(body_rms, rel=0.05)


def test_noise_gate_disabled():
    """A threshold of 0.0 passes audio through untouched."""
    audio = np.array([0.001, 0.01, -0.002, -0.05], dtype=np.float32)
    np.testing.assert_array_equal(noise_gate(audio, threshold=0.0), audio)

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
