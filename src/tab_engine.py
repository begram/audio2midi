import numpy as np

class TabMapper:
    """
    Optimized string assignment using pre-calculated mapping.
    """
    TUNING = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

    def __init__(self, max_frets=15):
        self.max_frets = max_frets
        # Pre-calculate pitch to (string, fret) mapping
        self._pitch_map = {}
        for pitch in range(40, 40 + max_frets + 25): # Cover all guitar notes
            placements = []
            for string, open_pitch in self.TUNING.items():
                fret = pitch - open_pitch
                if 0 <= fret <= self.max_frets:
                    placements.append((string, fret))
            self._pitch_map[pitch] = placements

    def get_possible_placements(self, pitch):
        return self._pitch_map.get(pitch, [])

    def assign_strings(self, notes):
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda x: x['start'])
        active_strings = {s: 0.0 for s in self.TUNING.keys()}
        current_box_fret = 0 

        for note in sorted_notes:
            pitch = note['pitch']
            start = note['start']
            candidates = self.get_possible_placements(pitch)
            
            if not candidates:
                note['string'], note['fret'] = 0, -1
                continue

            available = [c for c in candidates if active_strings[c[0]] <= start]
            if not available:
                available = sorted(candidates, key=lambda c: active_strings[c[0]])
            
            # Simplified scoring
            def score_candidate(c):
                string, fret = c
                # Heuristic: open strings are -5, otherwise distance from box
                return (abs(fret - current_box_fret) if fret > 0 else -5) + (string * 0.1)

            best_placement = min(available, key=score_candidate)
            note['string'], note['fret'] = best_placement
            active_strings[note['string']] = note['end']
            
            if note['fret'] > 0:
                current_box_fret = (current_box_fret * 0.7) + (note['fret'] * 0.3) if current_box_fret > 0 else note['fret']

        return sorted_notes
