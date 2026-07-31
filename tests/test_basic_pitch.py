import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from basic_pitch_engine import BasicPitchEngine


def test_basic_pitch_frequency_bounding():
    """Test that notes outside min_freq and max_freq are filtered out."""
    engine = BasicPitchEngine()

    # Mock note events: pitch 30 (out of bounds low), pitch 60 (in bounds), pitch 95 (out of bounds high)
    mock_note_events = [
        (0.0, 0.5, 30, 0.8),  # F#1 (~46 Hz) - Out of bounds
        (0.5, 1.0, 60, 0.9),  # C4 (~261 Hz) - In bounds
        (1.0, 1.5, 95, 0.7),  # B6 (~1975 Hz) - Out of bounds
    ]

    mock_predict = MagicMock(return_value=({}, None, mock_note_events))

    with patch('basic_pitch_engine.predict', mock_predict):
        dummy_audio = np.zeros(22050, dtype=np.float32)
        notes = engine.transcribe(
            dummy_audio, sr=22050,
            min_freq=80.0, max_freq=1400.0
        )

    # Only pitch 60 should remain
    assert len(notes) == 1
    assert notes[0]['pitch'] == 60

def test_basic_pitch_pitch_bends():
    """Bends come from the note's own contour (note_events[i][4]).

    Values are in 1/3-semitone units and scale by 4096/3 into MIDI ticks; the
    times are spread evenly across the owning note's span.
    """
    engine = BasicPitchEngine()

    # A single note carrying a 3-point bend contour: 0, +1/3 st, +1 st.
    mock_note_events = [
        (0.0, 1.0, 60, 0.8, [0, 1, 3]),
    ]

    mock_predict = MagicMock(return_value=({}, None, mock_note_events))

    with patch('basic_pitch_engine.predict', mock_predict):
        dummy_audio = np.zeros(22050, dtype=np.float32)
        notes = engine.transcribe(
            dummy_audio, sr=22050,
            include_pitch_bends=True
        )

    assert len(notes) == 1
    bends = notes[0]['pitch_bends']
    assert [b['pitch'] for b in bends] == [0, 1365, 4096]
    assert [b['time'] for b in bends] == [0.0, 0.5, 1.0]


def test_basic_pitch_bends_are_not_shared_between_simultaneous_notes():
    """Two notes sounding together must not inherit each other's bends."""
    engine = BasicPitchEngine()

    mock_note_events = [
        (0.0, 1.0, 60, 0.8, [3, 3]),
        (0.0, 1.0, 64, 0.8, None),
    ]

    mock_predict = MagicMock(return_value=({}, None, mock_note_events))

    with patch('basic_pitch_engine.predict', mock_predict):
        dummy_audio = np.zeros(22050, dtype=np.float32)
        notes = engine.transcribe(dummy_audio, sr=22050, include_pitch_bends=True)

    bent = [n for n in notes if n['pitch'] == 60][0]
    unbent = [n for n in notes if n['pitch'] == 64][0]
    assert len(bent['pitch_bends']) == 2
    assert unbent['pitch_bends'] == []


def test_basic_pitch_velocity_never_zero():
    """A near-silent onset must not become a velocity-0 (note-off) event."""
    engine = BasicPitchEngine()

    mock_predict = MagicMock(return_value=({}, None, [(0.0, 1.0, 60, 0.001)]))

    with patch('basic_pitch_engine.predict', mock_predict):
        dummy_audio = np.zeros(22050, dtype=np.float32)
        notes = engine.transcribe(dummy_audio, sr=22050)

    assert notes[0]['velocity'] == 1


def test_basic_pitch_velocity_never_exceeds_127():
    """An amplitude above 1.0 must clamp rather than overflow the MIDI range."""
    mock_predict = MagicMock(return_value=({}, None, [(0.0, 1.0, 60, 1.5)]))

    with patch('basic_pitch_engine.predict', mock_predict):
        notes = BasicPitchEngine().transcribe(np.zeros(1024, dtype=np.float32), sr=22050)

    assert notes[0]['velocity'] == 127


def test_basic_pitch_bends_clamped_to_14_bit_range():
    """Large contour excursions clamp instead of wrapping around."""
    # Units are 1/3 semitone and scale by 4096/3, so +-10 far exceeds the range.
    mock_predict = MagicMock(return_value=({}, None, [(0.0, 1.0, 60, 0.8, [10, -10])]))

    with patch('basic_pitch_engine.predict', mock_predict):
        notes = BasicPitchEngine().transcribe(
            np.zeros(1024, dtype=np.float32), sr=22050, include_pitch_bends=True
        )

    assert [b['pitch'] for b in notes[0]['pitch_bends']] == [8191, -8192]


def test_basic_pitch_handles_note_events_without_bend_field():
    """Four-element note events (no contour field) must not raise."""
    mock_predict = MagicMock(return_value=({}, None, [(0.0, 1.0, 60, 0.8)]))

    with patch('basic_pitch_engine.predict', mock_predict):
        notes = BasicPitchEngine().transcribe(
            np.zeros(1024, dtype=np.float32), sr=22050, include_pitch_bends=True
        )

    assert notes[0]['pitch_bends'] == []


def test_basic_pitch_omits_bend_key_when_disabled():
    mock_predict = MagicMock(return_value=({}, None, [(0.0, 1.0, 60, 0.8, [1, 2])]))

    with patch('basic_pitch_engine.predict', mock_predict):
        notes = BasicPitchEngine().transcribe(
            np.zeros(1024, dtype=np.float32), sr=22050, include_pitch_bends=False
        )

    assert 'pitch_bends' not in notes[0]


def test_basic_pitch_unbounded_range_keeps_extreme_pitches():
    """With no frequency bounds the full 0-127 MIDI range is admitted."""
    mock_predict = MagicMock(
        return_value=({}, None, [(0.0, 0.5, 20, 0.8), (0.5, 1.0, 100, 0.8)])
    )

    with patch('basic_pitch_engine.predict', mock_predict):
        notes = BasicPitchEngine().transcribe(np.zeros(1024, dtype=np.float32), sr=22050)

    assert sorted(n['pitch'] for n in notes) == [20, 100]


# --- temp file lifecycle ----------------------------------------------------

def test_temp_wav_is_removed_after_success():
    """The scratch WAV handed to the model must not survive the call."""
    seen = {}

    def fake_predict(audio_path, **kwargs):
        seen['path'] = audio_path
        assert os.path.exists(audio_path), "temp wav should exist during inference"
        return ({}, None, [(0.0, 1.0, 60, 0.8)])

    with patch('basic_pitch_engine.predict', side_effect=fake_predict):
        BasicPitchEngine().transcribe(np.zeros(1024, dtype=np.float32), sr=22050)

    assert not os.path.exists(seen['path']), "temp wav leaked after a successful run"


def test_temp_wav_is_removed_when_inference_raises():
    """A failing model must not leak a full-length WAV into the temp directory."""
    seen = {}

    def boom(audio_path, **kwargs):
        seen['path'] = audio_path
        raise RuntimeError("inference exploded")

    with patch('basic_pitch_engine.predict', side_effect=boom):
        with pytest.raises(RuntimeError, match="inference exploded"):
            BasicPitchEngine().transcribe(np.zeros(1024, dtype=np.float32), sr=22050)

    assert not os.path.exists(seen['path']), "temp wav leaked after a failed run"
