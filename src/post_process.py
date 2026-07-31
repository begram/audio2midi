import bisect

import numpy as np

TICKS_PER_QUARTER = 480

# Same-pitch notes closer than this are merge candidates.
MERGE_GAP_SECONDS = 0.050
# How far back to look for a concurrently sounding different pitch. A repeated
# note is only treated as engine chatter (rather than a genuine re-pluck) when
# another voice is sustaining through it, so the lookback spans a plausible
# sustain rather than the whole piece.
POLYPHONY_LOOKBACK_SECONDS = 2.0
POLYPHONY_LOOKAHEAD_SECONDS = 0.1


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
    Merges consecutive same-pitch notes that overlap (or nearly touch) while
    another pitch is sounding through them.

    O(N log N): the concurrency check binary-searches the start-sorted note
    list for the relevant time window instead of scanning every note.
    """
    if not notes:
        return []

    sorted_by_start = sorted(notes, key=lambda x: x['start'])
    starts = [n['start'] for n in sorted_by_start]

    by_pitch = {}
    for n in notes:
        by_pitch.setdefault(n['pitch'], []).append(n)

    merged_all = []
    for pitch, p_notes in by_pitch.items():
        p_notes.sort(key=lambda x: x['start'])
        current = p_notes[0].copy()

        for i in range(1, len(p_notes)):
            nxt = p_notes[i]

            if nxt['start'] <= (current['end'] + MERGE_GAP_SECONDS):
                # Window bounds are exclusive on both sides.
                lo = bisect.bisect_right(starts, nxt['start'] - POLYPHONY_LOOKBACK_SECONDS)
                hi = bisect.bisect_left(starts, nxt['start'] + POLYPHONY_LOOKAHEAD_SECONDS)
                is_other_active = any(
                    n['pitch'] != pitch and n['start'] <= nxt['start'] < n['end']
                    for n in sorted_by_start[lo:hi]
                )

                if is_other_active:
                    current['end'] = max(current['end'], nxt['end'])
                    current['velocity'] = max(current['velocity'], nxt['velocity'])
                    # Keep the absorbed note's bend contour; dropping it would
                    # silently discard expression on merged notes.
                    if nxt.get('pitch_bends'):
                        current['pitch_bends'] = (
                            list(current.get('pitch_bends', [])) + list(nxt['pitch_bends'])
                        )
                    continue

            merged_all.append(current)
            current = nxt.copy()
        merged_all.append(current)

    merged_all.sort(key=lambda x: x['start'])
    return merged_all

def quantize_notes(notes, bpm, grid_resolution='1/16', strength=1.0):
    """Vectorized quantization using NumPy with optional partial strength.

    Writes `start_ticks`/`end_ticks`; call `ticks_to_seconds` to apply them.
    """
    if not notes:
        return []

    seconds_per_quarter = 60.0 / bpm

    res_map = {'1/4': 1.0, '1/8': 0.5, '1/16': 0.25, '1/32': 0.125}
    grid_multiplier = res_map.get(grid_resolution, 0.25)
    grid_ticks = int(TICKS_PER_QUARTER * grid_multiplier)

    onsets = np.array([n['start'] for n in notes])
    offsets = np.array([n['end'] for n in notes])

    onset_ticks = (onsets / seconds_per_quarter) * TICKS_PER_QUARTER
    offset_ticks = (offsets / seconds_per_quarter) * TICKS_PER_QUARTER

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


def _retime_bends(bends, old_start, old_end, new_start, new_end):
    """Maps bend times from a note's original span onto its quantized span."""
    old_span = old_end - old_start
    new_span = new_end - new_start
    scale = (new_span / old_span) if old_span > 0 else 1.0

    retimed = []
    for b in bends:
        t = new_start + (b['time'] - old_start) * scale
        retimed.append({
            'pitch': b['pitch'],
            'time': float(min(max(t, new_start), new_end)),
        })
    return retimed


def ticks_to_seconds(notes, bpm):
    """Rewrites `start`/`end` from quantized tick values.

    Pitch bend times are mapped onto the new span too -- leaving them at their
    pre-quantization absolute times would push them outside their own note.
    """
    seconds_per_quarter = 60.0 / bpm

    for n in notes:
        if 'start_ticks' not in n or 'end_ticks' not in n:
            continue

        new_start = (n['start_ticks'] / TICKS_PER_QUARTER) * seconds_per_quarter
        new_end = (n['end_ticks'] / TICKS_PER_QUARTER) * seconds_per_quarter

        if n.get('pitch_bends'):
            n['pitch_bends'] = _retime_bends(
                n['pitch_bends'], n['start'], n['end'], new_start, new_end
            )

        n['start'] = new_start
        n['end'] = new_end

    return notes
