import pretty_midi

def generate_midi(notes, bpm, output_path, instrument_name='Acoustic Guitar'):
    """Generates a Standard MIDI File from a list of notes, splitting by string if available and writing pitch bends."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Create 6 instruments (one for each string) for better DAW and MPE support
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
            target_inst.pitch_bends.append(pretty_midi.PitchBend(pitch=pb_val, time=pb_time))

    # Add instruments that have notes or pitch bends to the MIDI object
    for i in range(1, 7):
        if string_instruments[i].notes or string_instruments[i].pitch_bends:
            pm.instruments.append(string_instruments[i])

    if default_inst.notes or default_inst.pitch_bends:
        pm.instruments.append(default_inst)

    pm.write(output_path)