import pytest
import os
import pretty_midi
from midi_gen import generate_midi

def test_generate_midi_mpe_channels(tmp_path):
    """Test generating MIDI with string assignments and pitch bends."""
    midi_path = os.path.join(tmp_path, "mpe_test.mid")

    notes = [
        {
            'pitch': 40, 'start': 0.0, 'end': 1.0, 'velocity': 100, 'string': 6,
            'pitch_bends': [{'pitch': 512, 'time': 0.2}, {'pitch': 1024, 'time': 0.5}]
        },
        {
            'pitch': 64, 'start': 0.0, 'end': 1.0, 'velocity': 90, 'string': 1,
            'pitch_bends': [{'pitch': -256, 'time': 0.3}]
        }
    ]

    generate_midi(notes, bpm=120, output_path=midi_path, instrument_name="Guitar MPE")

    assert os.path.exists(midi_path)
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Should have 2 instruments corresponding to String 6 and String 1
    assert len(pm.instruments) == 2

    string_6_inst = [inst for inst in pm.instruments if "String 6" in inst.name][0]
    string_1_inst = [inst for inst in pm.instruments if "String 1" in inst.name][0]

    assert len(string_6_inst.notes) == 1
    assert len(string_6_inst.pitch_bends) == 2
    assert string_6_inst.pitch_bends[0].pitch == 512

    assert len(string_1_inst.notes) == 1
    assert len(string_1_inst.pitch_bends) == 1
    assert string_1_inst.pitch_bends[0].pitch == -256
