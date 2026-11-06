import pretty_midi

PITCH_BEND_MIN = -8192
PITCH_BEND_MAX = 8191


def _channel_bends(pm, inst_notes):
    """Builds one channel's pitch-wheel events: each note's bends, then a recentre to 0.

    The recentre at each bent note's end stops its bend bleeding into whatever plays next
    on the channel.

    Exactly one event survives per MIDI tick, and it is the one that should be in force
    from that tick on: the later-starting note's, and within a note the later one in time
    (so a note's recentre beats its own last bend). This matters because pretty_midi writes
    pitch-wheel events sharing a tick sorted by *value*, not by intent -- left alone, the
    larger value simply wins. Shared ticks are routine, not edge cases: Basic Pitch puts a
    note's last bend exactly at its end, overlap resolution and quantization make one note
    end exactly where the next begins, and at low tempos a tick is longer than the ~11.6 ms
    between contour points. Consecutive repeats of one value are then dropped, as they
    change nothing on a channel-wide message.
    """
    events = []  # (tick, note rank, seq within note, value, time)
    for rank, n in enumerate(sorted(inst_notes, key=lambda n: n['start'])):
        bends = sorted(n.get('pitch_bends') or [], key=lambda b: b['time'])
        if not bends:
            continue
        end = float(n['end'])
        # A bend past the note's end would follow its recentre and hold the channel bent.
        timeline = [(float(b['time']), int(b['pitch'])) for b in bends if b['time'] <= end]
        timeline.append((end, 0))
        for seq, (time, value) in enumerate(timeline):
            value = max(PITCH_BEND_MIN, min(PITCH_BEND_MAX, value))
            events.append((pm.time_to_tick(time), rank, seq, value, time))

    events.sort()
    per_tick = {}
    for event in events:
        per_tick[event[0]] = event  # sorted, so the last one per tick is the one in force

    bends = []
    for _tick, _rank, _seq, value, time in sorted(per_tick.values()):
        if bends and bends[-1].pitch == value:
            continue
        bends.append(pretty_midi.PitchBend(pitch=value, time=time))
    return bends


def generate_midi(notes, bpm, output_path, instrument_name='Acoustic Guitar'):
    """Writes `notes` (the note-dict contract) to a Standard MIDI File at `output_path`.

    Notes with a `string` of 1-6 go to that string's own track, hence its own channel;
    everything else shares one unassigned track. `instrument_name` only names the tracks
    -- every track uses General MIDI program 24 (acoustic guitar, nylon) regardless.
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
        inst.pitch_bends = _channel_bends(pm, inst_notes)

    # Only tracks that received notes are written.
    for i in range(1, 7):
        if string_instruments[i].notes:
            pm.instruments.append(string_instruments[i])
    if default_inst.notes:
        pm.instruments.append(default_inst)

    pm.write(output_path)
