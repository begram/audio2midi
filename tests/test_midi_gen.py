import os

import pretty_midi
import pytest

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
    # Two authored bends plus a recentre-to-zero at the note's end, so the bend
    # does not bleed into whatever plays next on this channel.
    assert [pb.pitch for pb in string_6_inst.pitch_bends] == [512, 1024, 0]

    assert len(string_1_inst.notes) == 1
    assert [pb.pitch for pb in string_1_inst.pitch_bends] == [-256, 0]


def test_generate_midi_deduplicates_redundant_bends(tmp_path):
    """Consecutive bends holding the same value collapse to one event."""
    midi_path = os.path.join(tmp_path, "dedupe_test.mid")

    notes = [
        {
            'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 100, 'string': 3,
            'pitch_bends': [
                {'pitch': 512, 'time': 0.1},
                {'pitch': 512, 'time': 0.2},
                {'pitch': 512, 'time': 0.3},
                {'pitch': 900, 'time': 0.4},
            ],
        }
    ]

    generate_midi(notes, bpm=120, output_path=midi_path)

    pm = pretty_midi.PrettyMIDI(midi_path)
    inst = pm.instruments[0]
    assert [pb.pitch for pb in inst.pitch_bends] == [512, 900, 0]


def test_generate_midi_accepts_tuple_form_bends(tmp_path):
    """Bends may also arrive as (time, value) tuples rather than dicts."""
    midi_path = os.path.join(tmp_path, "tuple_bends.mid")

    notes = [{
        'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 100, 'string': 4,
        'pitch_bends': [(0.2, 512), (0.6, 1024)],
    }]

    generate_midi(notes, bpm=120, output_path=midi_path)

    pm = pretty_midi.PrettyMIDI(midi_path)
    inst = pm.instruments[0]
    assert [pb.pitch for pb in inst.pitch_bends] == [512, 1024, 0]
    assert inst.pitch_bends[0].time == pytest.approx(0.2, abs=1e-3)


def test_generate_midi_splits_string_and_unassigned_notes(tmp_path):
    """Notes with and without a string assignment land on separate tracks."""
    midi_path = os.path.join(tmp_path, "mixed.mid")

    notes = [
        {'pitch': 40, 'start': 0.0, 'end': 1.0, 'velocity': 100, 'string': 6},
        {'pitch': 64, 'start': 0.0, 'end': 1.0, 'velocity': 90},              # no string
        {'pitch': 20, 'start': 1.0, 'end': 2.0, 'velocity': 90, 'string': 0},  # unplayable
    ]

    generate_midi(notes, bpm=120, output_path=midi_path, instrument_name="Guitar")

    pm = pretty_midi.PrettyMIDI(midi_path)
    by_name = {inst.name: inst for inst in pm.instruments}

    assert "Guitar - String 6" in by_name
    assert len(by_name["Guitar - String 6"].notes) == 1
    # The unassigned and unplayable notes share the default track.
    assert sorted(n.pitch for n in by_name["Guitar"].notes) == [20, 64]


def test_generate_midi_routes_out_of_range_string_to_default(tmp_path):
    """A string index outside 1-6 must not raise a KeyError."""
    midi_path = os.path.join(tmp_path, "bad_string.mid")

    notes = [{'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 100, 'string': 9}]

    generate_midi(notes, bpm=120, output_path=midi_path, instrument_name="Guitar")

    pm = pretty_midi.PrettyMIDI(midi_path)
    assert [inst.name for inst in pm.instruments] == ["Guitar"]
    assert len(pm.instruments[0].notes) == 1


def test_generate_midi_with_no_notes(tmp_path):
    """An empty transcription still produces a readable file."""
    midi_path = os.path.join(tmp_path, "empty.mid")

    generate_midi([], bpm=120, output_path=midi_path)

    pm = pretty_midi.PrettyMIDI(midi_path)
    assert sum(len(inst.notes) for inst in pm.instruments) == 0


def test_generate_midi_clamps_out_of_range_bends(tmp_path):
    """Bend values outside the 14-bit signed range are clamped, not wrapped."""
    midi_path = os.path.join(tmp_path, "clamp_test.mid")

    notes = [
        {
            'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 100, 'string': 2,
            'pitch_bends': [{'pitch': 20000, 'time': 0.1}, {'pitch': -20000, 'time': 0.5}],
        }
    ]

    generate_midi(notes, bpm=120, output_path=midi_path)

    pm = pretty_midi.PrettyMIDI(midi_path)
    values = [pb.pitch for pb in pm.instruments[0].pitch_bends]
    assert values[:2] == [8191, -8192]
