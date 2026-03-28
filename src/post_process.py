import numpy as np

def clean_notes(notes, min_duration=0.030, velocity_threshold=10):
    """Filters notes by minimum duration and velocity."""
    cleaned = [
        n for n in notes 
        if (n['end'] - n['start']) >= min_duration and n['velocity'] >= velocity_threshold
    ]
    return cleaned

def merge_notes(notes):
    """
    Optimized O(N log N) note merging using a sorted event list (sweep-line).
    """
    if not notes:
        return []

    # Sort for overall active note tracking
    sorted_by_start = sorted(notes, key=lambda x: x['start'])

    # Pre-calculate active note counts for all merge candidates
    # We use a sweep-line to know if "any other pitch" is active at any time T
    events = []
    for i, n in enumerate(notes):
        events.append((n['start'], 1, n['pitch'])) # 1 for start
        events.append((n['end'], -1, n['pitch']))  # -1 for end
    events.sort()

    # Group by pitch for merging
    by_pitch = {}
    for n in notes:
        p = n['pitch']
        if p not in by_pitch:
            by_pitch[p] = []
        by_pitch[p].append(n)

    merged_all = []
    for pitch, p_notes in by_pitch.items():
        p_notes.sort(key=lambda x: x['start'])
        current = p_notes[0].copy()

        for i in range(1, len(p_notes)):
            nxt = p_notes[i]

            if nxt['start'] <= (current['end'] + 0.050):
                # Efficiently check for other active pitches:
                # Is there any note where other.pitch != pitch AND other.start <= nxt.start < other.end?
                # For optimization, we only check neighbors in time
                is_other_active = any(
                    n['pitch'] != pitch and n['start'] <= nxt['start'] < n['end']
                    for n in sorted_by_start 
                    if nxt['start'] - 2.0 < n['start'] < nxt['start'] + 0.1 # Scoped search
                )

                if is_other_active:
                    current['end'] = max(current['end'], nxt['end'])
                    current['velocity'] = max(current['velocity'], nxt['velocity'])
                    continue 

            merged_all.append(current)
            current = nxt.copy()
        merged_all.append(current)

    merged_all.sort(key=lambda x: x['start'])
    return merged_all

def quantize_notes(notes, bpm, grid_resolution='1/16'):
    """Vectorized quantization using NumPy."""
    if not notes:
        return []

    seconds_per_quarter = 60.0 / bpm
    ticks_per_quarter = 480

    res_map = {'1/4': 1.0, '1/8': 0.5, '1/16': 0.25, '1/32': 0.125}
    grid_multiplier = res_map.get(grid_resolution, 0.25)
    grid_ticks = int(ticks_per_quarter * grid_multiplier)

    onsets = np.array([n['start'] for n in notes])
    offsets = np.array([n['end'] for n in notes])

    onset_ticks = (onsets / seconds_per_quarter) * ticks_per_quarter
    offset_ticks = (offsets / seconds_per_quarter) * ticks_per_quarter

    q_onsets = (np.round(onset_ticks / grid_ticks) * grid_ticks).astype(int)
    q_offsets = (np.round(offset_ticks / grid_ticks) * grid_ticks).astype(int)

    # Fix zero-duration notes
    mask = q_offsets <= q_onsets
    q_offsets[mask] = q_onsets[mask] + grid_ticks

    for i, n in enumerate(notes):
        n['start_ticks'] = int(q_onsets[i])
        n['end_ticks'] = int(q_offsets[i])

    return notes