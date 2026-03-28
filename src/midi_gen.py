import pretty_midi

def generate_midi(notes, bpm, output_path, instrument_name='Acoustic Guitar'):
    """Generates a Standard MIDI File from a list of notes."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=24) 
    instrument.name = instrument_name

    for n in notes:
        note = pretty_midi.Note(
            velocity=int(n['velocity']),
            pitch=int(n['pitch']),
            start=float(n['start']),
            end=float(n['end'])
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(output_path)