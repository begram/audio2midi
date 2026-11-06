import os

import pretty_midi
import pytest

from midi_gen import generate_midi


def test_generate_midi_writes_one_track_per_string(tmp_path):
    """String-assigned notes land on their own string's track, each with its own bends."""
    midi_path = os.path.join(tmp_path, "per_string.mid")

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

    generate_midi(notes, bpm=120, output_path=midi_path, instrument_name="Guitar")

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


def _wheel_at(inst, time):
    """Effective pitch-wheel value at `time`: the last event at or before it, in file order.

    File order matters, not time order: pretty_midi writes same-tick pitch-wheel events
    sorted by value, so that is the order a synth applies them in.
    """
    value = 0
    for pb in inst.pitch_bends:
        if pb.time <= time + 1e-9:
            value = pb.pitch
    return value


def test_next_notes_downward_bend_survives_a_shared_boundary(tmp_path):
    """A note truncated to the next pluck ends on the tick where the next note's bend starts.

    The previous note's 0 recentre and the next note's first bend then share a tick, and
    pretty_midi orders those by value: a negative bend sorts first and the 0 overwrites it.
    """
    midi_path = os.path.join(tmp_path, "boundary.mid")
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 100, 'string': 2,
         'pitch_bends': [{'pitch': 300, 'time': 0.2}]},
        {'pitch': 62, 'start': 0.5, 'end': 1.0, 'velocity': 100, 'string': 2,
         'pitch_bends': [{'pitch': -700, 'time': 0.5}, {'pitch': 900, 'time': 0.8}]},
    ]

    generate_midi(notes, bpm=120, output_path=midi_path)

    inst = pretty_midi.PrettyMIDI(midi_path).instruments[0]
    assert _wheel_at(inst, 0.5) == -700
    assert _wheel_at(inst, 1.0) == 0


def test_upward_last_bend_at_note_end_does_not_bleed(tmp_path):
    """Basic Pitch puts a note's last bend exactly at its end, on the recentre's tick.

    Sorted by value, a positive bend would land after the 0 and hold the channel bent
    through the silence that follows.
    """
    midi_path = os.path.join(tmp_path, "bleed.mid")
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 100, 'string': 2,
         'pitch_bends': [{'pitch': 200, 'time': 0.0}, {'pitch': 800, 'time': 0.5}]},
    ]

    generate_midi(notes, bpm=120, output_path=midi_path)

    inst = pretty_midi.PrettyMIDI(midi_path).instruments[0]
    assert _wheel_at(inst, 0.25) == 200
    assert _wheel_at(inst, 0.5) == 0


def test_bends_inside_one_note_on_one_tick_keep_the_later_value(tmp_path):
    """Two contour points 0.1 ms apart share a tick; the later one must be in force.

    At 120 BPM a tick is ~2.3 ms; at low tempos it exceeds Basic Pitch's ~11.6 ms contour
    spacing, so this happens on real material. Sorted by value, 500 would beat 200.
    """
    midi_path = os.path.join(tmp_path, "intra_tick.mid")
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 100, 'string': 2,
         'pitch_bends': [{'pitch': 500, 'time': 0.1000}, {'pitch': 200, 'time': 0.1001}]},
    ]

    generate_midi(notes, bpm=120, output_path=midi_path)

    inst = pretty_midi.PrettyMIDI(midi_path).instruments[0]
    assert _wheel_at(inst, 0.5) == 200


def test_bend_past_its_note_end_is_dropped(tmp_path):
    """Otherwise it would follow the recentre and hold the channel bent after the note."""
    midi_path = os.path.join(tmp_path, "past_end.mid")
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 100, 'string': 2,
         'pitch_bends': [{'pitch': 300, 'time': 0.2}, {'pitch': 900, 'time': 0.8}]},
    ]

    generate_midi(notes, bpm=120, output_path=midi_path)

    inst = pretty_midi.PrettyMIDI(midi_path).instruments[0]
    assert [pb.pitch for pb in inst.pitch_bends] == [300, 0]


def test_bend_output_does_not_depend_on_note_order(tmp_path):
    results = []
    for order in ('forward', 'reversed'):
        notes = [
            {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 100, 'string': 2,
             'pitch_bends': [{'pitch': 300, 'time': 0.2}]},
            {'pitch': 62, 'start': 0.5, 'end': 1.0, 'velocity': 100, 'string': 2,
             'pitch_bends': [{'pitch': -700, 'time': 0.5}]},
        ]
        if order == 'reversed':
            notes.reverse()
        midi_path = os.path.join(tmp_path, f"{order}.mid")
        generate_midi(notes, bpm=120, output_path=midi_path)
        inst = pretty_midi.PrettyMIDI(midi_path).instruments[0]
        results.append([(round(pb.time, 4), pb.pitch) for pb in inst.pitch_bends])

    assert results[0] == results[1]


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


def test_generate_midi_writes_contract_form_bends(tmp_path):
    """Bends use the engine_base contract: {'pitch': pitch-wheel units, 'time': seconds}."""
    midi_path = os.path.join(tmp_path, "dict_bends.mid")

    notes = [{
        'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 100, 'string': 4,
        'pitch_bends': [{'pitch': 512, 'time': 0.2}, {'pitch': 1024, 'time': 0.6}],
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
