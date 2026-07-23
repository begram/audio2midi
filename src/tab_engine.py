import numpy as np
import itertools

class TabMapper:
    """
    Optimized string assignment using dynamic programming (Viterbi solver)
    and strict polyphonic chord string collision guards.
    """
    TUNING = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

    def __init__(self, max_frets=15):
        self.max_frets = max_frets
        self._pitch_map = {}
        for pitch in range(40, 40 + max_frets + 25):
            placements = []
            for string, open_pitch in self.TUNING.items():
                fret = pitch - open_pitch
                if 0 <= fret <= self.max_frets:
                    placements.append((string, fret))
            self._pitch_map[pitch] = placements

    def get_possible_placements(self, pitch):
        return self._pitch_map.get(pitch, [])

    def assign_strings(self, notes):
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda x: x['start'])

        # Group simultaneous or near-simultaneous notes into chord frames (within 15ms)
        frames = []
        current_frame = [sorted_notes[0]]
        for n in sorted_notes[1:]:
            if abs(n['start'] - current_frame[0]['start']) < 0.015:
                current_frame.append(n)
            else:
                frames.append(current_frame)
                current_frame = [n]
        if current_frame:
            frames.append(current_frame)

        # For each frame, generate valid non-colliding (string, fret) tuples
        frame_candidates = []
        for frame in frames:
            cands = self._generate_frame_candidates(frame)
            frame_candidates.append(cands)

        # Viterbi Solver across time frames
        num_frames = len(frames)
        dp = []
        backpointers = []

        # Step 0
        first_cands = frame_candidates[0]
        first_costs = [self._frame_initial_cost(c) for c in first_cands]
        dp.append(first_costs)
        backpointers.append([-1] * len(first_cands))

        for t in range(1, num_frames):
            prev_cands = frame_candidates[t - 1]
            curr_cands = frame_candidates[t]
            curr_costs = []
            curr_bp = []

            for curr_idx, curr_cand in enumerate(curr_cands):
                min_cost = float('inf')
                best_prev = 0

                for prev_idx, prev_cand in enumerate(prev_cands):
                    t_cost = self._transition_cost(prev_cand, curr_cand)
                    cost = dp[t - 1][prev_idx] + t_cost
                    if cost < min_cost:
                        min_cost = cost
                        best_prev = prev_idx

                curr_costs.append(min_cost)
                curr_bp.append(best_prev)

            dp.append(curr_costs)
            backpointers.append(curr_bp)

        # Backtrack best path
        best_path = [0] * num_frames
        best_last_idx = int(np.argmin(dp[-1]))
        best_path[-1] = best_last_idx

        for t in range(num_frames - 2, -1, -1):
            best_path[t] = backpointers[t + 1][best_path[t + 1]]

        # Assign calculated (string, fret) back to notes
        for t, frame in enumerate(frames):
            cand = frame_candidates[t][best_path[t]]
            for note_idx, note in enumerate(frame):
                string, fret = cand[note_idx]
                note['string'] = string
                note['fret'] = fret

        return sorted_notes

    def _generate_frame_candidates(self, frame):
        """Generates valid candidate assignments for a chord frame ensuring no string collisions."""
        possible_per_note = [self.get_possible_placements(n['pitch']) for n in frame]

        # Handle out-of-range pitches
        for i in range(len(possible_per_note)):
            if not possible_per_note[i]:
                possible_per_note[i] = [(0, -1)]

        # Cartesian product with uniqueness guard on string (if string > 0)
        valid_combos = []
        for combo in itertools.product(*possible_per_note):
            strings = [c[0] for c in combo if c[0] > 0]
            if len(strings) == len(set(strings)):
                valid_combos.append(combo)

        if not valid_combos:
            # Fallback if strict unique fails
            valid_combos = [tuple(p[0] for p in possible_per_note)]

        return valid_combos

    def _frame_initial_cost(self, combo):
        """Initial cost for a frame candidate tuple."""
        cost = 0.0
        frets = [c[1] for c in combo if c[1] > 0]
        if frets:
            avg_fret = sum(frets) / len(frets)
            cost += abs(avg_fret - 5.0)  # Prefer lower-mid neck position (around 5th fret)
        for s, f in combo:
            if f == 0:
                cost -= 3.0  # Bonus for open strings
        return cost

    def _transition_cost(self, prev_combo, curr_combo):
        """Calculates physical transition energy cost between two consecutive frame candidates."""
        prev_frets = [c[1] for c in prev_combo if c[1] > 0]
        curr_frets = [c[1] for c in curr_combo if c[1] > 0]

        if not prev_frets or not curr_frets:
            return 0.0

        avg_prev_fret = sum(prev_frets) / len(prev_frets)
        avg_curr_fret = sum(curr_frets) / len(curr_frets)

        fret_dist = abs(avg_curr_fret - avg_prev_fret)

        prev_strings = [c[0] for c in prev_combo if c[0] > 0]
        curr_strings = [c[0] for c in curr_combo if c[0] > 0]
        avg_prev_string = sum(prev_strings) / len(prev_strings) if prev_strings else 3.5
        avg_curr_string = sum(curr_strings) / len(curr_strings) if curr_strings else 3.5

        string_dist = abs(avg_curr_string - avg_prev_string)

        cost = (fret_dist * 1.5) + (string_dist * 0.5)

        for s, f in curr_combo:
            if f == 0:
                cost -= 2.0  # Open string preference

        return cost
