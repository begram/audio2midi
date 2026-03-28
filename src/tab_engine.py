import numpy as np

class TabMapper:
    """
    Assigns MIDI notes to guitar strings and frets using a heuristic-based approach.
    """
    # String -> Open MIDI Pitch (Standard Tuning)
    TUNING = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

    def __init__(self, max_frets=15):
        self.max_frets = max_frets

    def get_possible_placements(self, pitch):
        """Returns a list of (string, fret) for a given pitch."""
        placements = []
        for string, open_pitch in self.TUNING.items():
            fret = pitch - open_pitch
            if 0 <= fret <= self.max_frets:
                placements.append((string, fret))
        return placements

    def assign_strings(self, notes):
        """
        Processes a list of notes and adds 'string' and 'fret' keys.
        """
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda x: x['start'])
        
        # Track when each string becomes available: {string: end_time}
        active_strings = {s: 0.0 for s in self.TUNING.keys()}
        
        # Track current "box" center (weighted average fret of active notes)
        current_box_fret = 0 

        for note in sorted_notes:
            pitch = note['pitch']
            start = note['start']
            
            candidates = self.get_possible_placements(pitch)
            
            if not candidates:
                # Pitch out of range for standard guitar
                note['string'] = 0
                note['fret'] = -1
                continue

            # Filter for strings that are not currently ringing
            available = [c for c in candidates if active_strings[c[0]] <= start]
            
            if not available:
                # If all strings for this pitch are busy, pick the one that ends soonest
                available = sorted(candidates, key=lambda c: active_strings[c[0]])
            
            # Score candidates based on "Box" and "Open String" heuristics
            def score_candidate(c):
                string, fret = c
                score = 0
                
                # Favor open strings (easier to play)
                if fret == 0:
                    score -= 5
                
                # Distance from current hand position (box)
                score += abs(fret - current_box_fret)
                
                # Slight tie-breaker for lower strings
                score += string * 0.1
                
                return score

            best_placement = min(available, key=score_candidate)
            
            note['string'] = best_placement[0]
            note['fret'] = best_placement[1]
            
            # Update state
            active_strings[note['string']] = note['end']
            
            # Update box center if it was a fretted note
            if note['fret'] > 0:
                if current_box_fret == 0:
                    current_box_fret = note['fret']
                else:
                    # Nudge box towards new note
                    current_box_fret = (current_box_fret * 0.7) + (note['fret'] * 0.3)

        return sorted_notes
