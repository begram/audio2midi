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


def _write_bends(pm, inst, inst_notes):
    """Writes each note's bends to its channel, recentring to 0 at the note's end.

    Collisions are judged in MIDI ticks, because that is where they bite: pretty_midi
    orders pitch-wheel events sharing a tick by *value*, not by insertion, so whichever
    of two same-tick events is larger wins regardless of intent. Two events land on one
    tick routinely -- Basic Pitch places a note's last bend exactly at its end, and
    overlap resolution and quantization make one note end exactly where the next begins.
    Hence:

    * a note's own bends at or after its end tick are dropped, since the recentre is
      meant to supersede them (a positive one would otherwise outrank the 0 and bleed);
    * the recentre is skipped when the next note on the channel already bends at or
      before that tick, since its value is the correct one from there on (a negative one
      would otherwise be outranked by the 0 and lost).
    """
    inst_notes = sorted(inst_notes, key=lambda n: n['start'])
    for i, n in enumerate(inst_notes):
        bends = n.get('pitch_bends') or []
        if not bends:
            continue

        end_tick = pm.time_to_tick(float(n['end']))
        for pb in bends:
            if pm.time_to_tick(float(pb['time'])) >= end_tick:
                continue
            value = max(PITCH_BEND_MIN, min(PITCH_BEND_MAX, int(pb['pitch'])))
            inst.pitch_bends.append(pretty_midi.PitchBend(pitch=value, time=float(pb['time'])))

        nxt_bends = inst_notes[i + 1].get('pitch_bends') if i + 1 < len(inst_notes) else None
        if nxt_bends and pm.time_to_tick(min(float(b['time']) for b in nxt_bends)) <= end_tick:
            continue
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=float(n['end'])))


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

    notes_by_inst = {}
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
        notes_by_inst.setdefault(id(target_inst), (target_inst, []))[1].append(n)

    for inst, inst_notes in notes_by_inst.values():
        _write_bends(pm, inst, inst_notes)

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
