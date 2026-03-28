import pretty_midi

def generate_midi(notes, bpm, output_path, instrument_name='Acoustic Guitar'):
    """Generates a Standard MIDI File from a list of notes, splitting by string if available."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Create 6 instruments (one for each string) for better DAW support
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
        if 1 <= s <= 6:
            string_instruments[s].notes.append(note)
        else:
            default_inst.notes.append(note)

    # Add instruments that have notes to the MIDI object
    for i in range(1, 7):
        if string_instruments[i].notes:
            pm.instruments.append(string_instruments[i])

    if default_inst.notes:
        pm.instruments.append(default_inst)

    pm.write(output_path)