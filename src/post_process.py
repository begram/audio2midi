import numpy as np

def clean_notes(notes, min_duration=0.030, velocity_threshold=10):
    """Filters notes by minimum duration and velocity."""
    cleaned = [
        n for n in notes 
        if (n['end'] - n['start']) >= min_duration and n['velocity'] >= velocity_threshold
    ]
    return cleaned

def apply_logarithmic_velocity(notes, curvature=5.0):
    """
    Applies a logarithmic perceptual dynamic velocity mapping:
    V_out = round(127 * ln(1 + k * a) / ln(1 + k)) where a = V_in / 127.
    """
    if not notes or curvature <= 0:
        return notes

    denom = np.log(1.0 + curvature)
    for n in notes:
        v_in = float(n['velocity'])
        if v_in <= 0:
            continue
        a = v_in / 127.0
        v_out = int(np.round(127.0 * (np.log(1.0 + curvature * a) / denom)))
        n['velocity'] = int(np.clip(v_out, 1, 127))

    return notes

def merge_notes(notes):
    """
    Optimized O(N log N) note merging using a sorted event list (sweep-line).
    """
    if not notes:
        return []

    sorted_by_start = sorted(notes, key=lambda x: x['start'])

    events = []
    for i, n in enumerate(notes):
        events.append((n['start'], 1, n['pitch'])) # 1 for start
        events.append((n['end'], -1, n['pitch']))  # -1 for end
    events.sort()

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
                is_other_active = any(
                    n['pitch'] != pitch and n['start'] <= nxt['start'] < n['end']
                    for n in sorted_by_start 
                    if nxt['start'] - 2.0 < n['start'] < nxt['start'] + 0.1
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

def quantize_notes(notes, bpm, grid_resolution='1/16', strength=1.0):
    """Vectorized quantization using NumPy with optional partial strength."""
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

    q_onsets = np.round(onset_ticks / grid_ticks) * grid_ticks
    q_offsets = np.round(offset_ticks / grid_ticks) * grid_ticks

    if strength < 1.0:
        eff_onsets = np.round(onset_ticks + strength * (q_onsets - onset_ticks)).astype(int)
        eff_offsets = np.round(offset_ticks + strength * (q_offsets - offset_ticks)).astype(int)
    else:
        eff_onsets = q_onsets.astype(int)
        eff_offsets = q_offsets.astype(int)

    mask = eff_offsets <= eff_onsets
    eff_offsets[mask] = eff_onsets[mask] + grid_ticks

    for i, n in enumerate(notes):
        n['start_ticks'] = int(eff_onsets[i])
        n['end_ticks'] = int(eff_offsets[i])

    return notes