import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from basic_pitch_engine import BasicPitchEngine
import pretty_midi

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
    """Test pitch bend extraction from prediction output."""
    engine = BasicPitchEngine()

    mock_note_events = [
        (0.0, 1.0, 60, 0.8),
    ]

    # Create mock pretty_midi object with pitch bend events
    pm_mock = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=1024, time=0.2))
    pm_mock.instruments.append(inst)

    mock_predict = MagicMock(return_value=({}, pm_mock, mock_note_events))

    with patch('basic_pitch_engine.predict', mock_predict):
        dummy_audio = np.zeros(22050, dtype=np.float32)
        notes = engine.transcribe(
            dummy_audio, sr=22050,
            include_pitch_bends=True
        )

    assert len(notes) == 1
    assert 'pitch_bends' in notes[0]
    assert len(notes[0]['pitch_bends']) == 1
    assert notes[0]['pitch_bends'][0]['pitch'] == 1024
