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
    Merges notes of the same pitch if they overlap (the second note starts 
    while the first is still ringing), but ONLY if at least one other 
    note of a different pitch is active at the start of the second note. 
    This suppresses re-triggering artifacts during chords and strums.
    """
    if not notes:
        return []
        
    # Group by pitch
    by_pitch = {}
    for n in notes:
        p = n['pitch']
        if p not in by_pitch:
            by_pitch[p] = []
        by_pitch[p].append(n)
        
    merged_all = []
    
    for pitch, p_notes in by_pitch.items():
        # Sort by start time
        p_notes.sort(key=lambda x: x['start'])
        
        if not p_notes:
            continue
            
        current = p_notes[0].copy()
        
        for i in range(1, len(p_notes)):
            nxt = p_notes[i]
            
            # Check for overlap (nxt starts before current ends)
            # We add a tiny 50ms grace period for "near-overlaps"
            if nxt['start'] <= (current['end'] + 0.050):
                # Check if another pitch is active at nxt['start']
                is_other_active = any(
                    other['pitch'] != pitch and 
                    other['start'] <= nxt['start'] < other['end']
                    for other in notes
                )
                
                if is_other_active:
                    # Merge: Extend the end time if the next note ends later
                    current['end'] = max(current['end'], nxt['end'])
                    current['velocity'] = max(current['velocity'], nxt['velocity'])
                    continue 

            # If we didn't merge, save current and move to next
            merged_all.append(current)
            current = nxt.copy()
        
        merged_all.append(current)
        
    # Re-sort by start time across all pitches
    merged_all.sort(key=lambda x: x['start'])
    return merged_all

def quantize_notes(notes, bpm, grid_resolution='1/16'):
    """Snaps note onsets and offsets to the musical grid based on BPM."""
    seconds_per_quarter = 60.0 / bpm
    ticks_per_quarter = 480
    
    res_map = {'1/4': 1.0, '1/8': 0.5, '1/16': 0.25, '1/32': 0.125}
    grid_multiplier = res_map.get(grid_resolution, 0.25)
    grid_ticks = int(ticks_per_quarter * grid_multiplier)

    for n in notes:
        onset_ticks = (n['start'] / seconds_per_quarter) * ticks_per_quarter
        offset_ticks = (n['end'] / seconds_per_quarter) * ticks_per_quarter
        
        n['start_ticks'] = int(round(onset_ticks / grid_ticks) * grid_ticks)
        n['end_ticks'] = int(round(offset_ticks / grid_ticks) * grid_ticks)
        
        if n['end_ticks'] <= n['start_ticks']:
            n['end_ticks'] = n['start_ticks'] + grid_ticks
            
    return notes