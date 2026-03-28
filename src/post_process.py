import numpy as np

def clean_notes(notes, min_duration=0.030, velocity_threshold=10):
    """Filters notes by minimum duration and velocity."""
    cleaned = [
        n for n in notes 
        if (n['end'] - n['start']) >= min_duration and n['velocity'] >= velocity_threshold
    ]
    return cleaned

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