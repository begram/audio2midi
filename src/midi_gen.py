import pretty_midi

PITCH_BEND_MIN = -8192
PITCH_BEND_MAX = 8191


def _clean_bends(bends):
    """Sorts bend events by time and drops redundant repeats.

    Pitch bend is a channel-level message, so duplicates accumulated from
    several notes sharing a track bloat the file without changing playback.
    """
    bends.sort(key=lambda b: b.time)

    cleaned = []
    for b in bends:
        if cleaned and cleaned[-1].pitch == b.pitch:
            continue
        cleaned.append(b)
    return cleaned


def generate_midi(notes, bpm, output_path, instrument_name='Acoustic Guitar'):
    """Generates a Standard MIDI File from a list of notes.

    Splits notes onto per-string tracks where a string assignment is available
    and writes each note's pitch bend events.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # One track (and therefore one MIDI channel) per string. A physical string
    # sounds one note at a time, so per-string channels give each voice its own
    # bend stream -- which is what makes the bends meaningful.
    string_instruments = {}
    for i in range(1, 7):
        inst = pretty_midi.Instrument(program=24) # Acoustic Guitar (nylon)
        inst.name = f"{instrument_name} - String {i}"
        string_instruments[i] = inst

    # Default instrument for notes without string info
    default_inst = pretty_midi.Instrument(program=24)
    default_inst.name = instrument_name

    for n in notes:
        note = pretty_midi.Note(
            velocity=int(n['velocity']),
            pitch=int(n['pitch']),
            start=float(n['start']),
            end=float(n['end'])
        )

        s = n.get('string', 0)
        target_inst = string_instruments[s] if 1 <= s <= 6 else default_inst
        target_inst.notes.append(note)

        # Write pitch bend events if present
        pitch_bends = n.get('pitch_bends', [])
        for pb in pitch_bends:
            if isinstance(pb, dict):
                pb_val = int(pb['pitch'])
                pb_time = float(pb['time'])
            else:
                pb_time, pb_val = float(pb[0]), int(pb[1])
            pb_val = max(PITCH_BEND_MIN, min(PITCH_BEND_MAX, pb_val))
            target_inst.pitch_bends.append(pretty_midi.PitchBend(pitch=pb_val, time=pb_time))

        if pitch_bends:
            # Recentre at the note's end so the bend does not bleed into the
            # next note on this channel.
            target_inst.pitch_bends.append(
                pretty_midi.PitchBend(pitch=0, time=float(n['end']))
            )

    # Add instruments that have notes or pitch bends to the MIDI object
    for i in range(1, 7):
        inst = string_instruments[i]
        if inst.notes or inst.pitch_bends:
            inst.pitch_bends = _clean_bends(inst.pitch_bends)
            pm.instruments.append(inst)

    if default_inst.notes or default_inst.pitch_bends:
        default_inst.pitch_bends = _clean_bends(default_inst.pitch_bends)
        pm.instruments.append(default_inst)

    pm.write(output_path)
