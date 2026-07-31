import itertools

import numpy as np

# Shortest gap between two onsets that can be a genuine re-pluck of one string. Even
# tremolo picking does not repeat a string faster than this, so a closer pair is two
# voices the solver could not separate, not one string played twice. Matches
# MERGE_GAP_SECONDS in post_process, which treats the same span as indistinguishable.
MIN_REPLUCK_SECONDS = 0.050


class TabMapper:
    """
    Optimized string assignment using dynamic programming (Viterbi solver)
    and strict polyphonic chord string collision guards.
    """
    TUNING = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

    PREFERRED_FRET = 5.0
    OPEN_STRING_BONUS = 3.0
    FRET_TRAVEL_WEIGHT = 1.5
    STRING_TRAVEL_WEIGHT = 0.5

    def __init__(self, max_frets=15, max_candidates=64):
        self.max_frets = max_frets
        self.max_candidates = max_candidates
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
        # Always non-empty: it is seeded with sorted_notes[0] and only ever
        # reassigned to a one-element list.
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
        first_costs = [self._frame_position_cost(c) for c in first_cands]
        dp.append(first_costs)
        backpointers.append([-1] * len(first_cands))

        for t in range(1, num_frames):
            prev_cands = frame_candidates[t - 1]
            curr_cands = frame_candidates[t]
            curr_costs = []
            curr_bp = []

            for curr_cand in curr_cands:
                min_cost = float('inf')
                best_prev = 0

                for prev_idx, prev_cand in enumerate(prev_cands):
                    t_cost = self._transition_cost(prev_cand, curr_cand)
                    cost = dp[t - 1][prev_idx] + t_cost
                    if cost < min_cost:
                        min_cost = cost
                        best_prev = prev_idx

                # Emission cost is applied at every frame, not just the first.
                # Without it, absolute neck position is only anchored at frame 0
                # and the preference decays away over the rest of the piece.
                curr_costs.append(min_cost + self._frame_position_cost(curr_cand))
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

        return self._resolve_string_overlaps(sorted_notes)

    @staticmethod
    def _resolve_string_overlaps(notes):
        """Shortens a note when a later note is plucked on the same string.

        The candidate guard in `_generate_frame_candidates` only prevents collisions
        *within* one 15ms chord frame. A note sustaining across frames can still be
        followed by another note assigned to its own string, which is physically
        impossible: a string sounds one note at a time, and the new pluck stops the
        old one. Left in place the overlap also defeats the per-string channel layout
        in `midi_gen`, since one channel's pitch bend would apply to both notes.

        Truncating rather than reassigning is deliberate -- the Viterbi state is a
        single frame, so the solver cannot see a sustain spanning frames, and making
        it sustain-aware would mean carrying which strings are still ringing in the
        state space.
        """
        by_string = {}
        for note in notes:
            if note.get('string', 0) > 0:
                by_string.setdefault(note['string'], []).append(note)

        for string_notes in by_string.values():
            string_notes.sort(key=lambda n: n['start'])
            # Pairwise over adjacent notes is sufficient: the list is start-sorted and
            # notes are only ever shortened, so resolving each pair cannot reintroduce
            # an overlap with a later one.
            for prev, nxt in zip(string_notes, string_notes[1:], strict=False):
                if nxt['start'] >= prev['end']:
                    continue

                if nxt['start'] - prev['start'] < MIN_REPLUCK_SECONDS:
                    # Too close together to be a re-pluck of the same string, so there is
                    # no earlier note to stop and truncating would leave an unplayable
                    # sliver (or, at an identical onset, a zero-length note). These are
                    # near-simultaneous unplaceable voices -- more than six at once, or
                    # duplicate detections `merge_notes` did not remove. The note keeps
                    # its pitch, velocity and length but gives up the string, which routes
                    # it to the unassigned track in `midi_gen` instead of colliding on a
                    # per-string channel.
                    prev['string'] = 0
                    prev['fret'] = -1
                    continue

                prev['end'] = nxt['start']
                # Bends past the new end would land after the recentre event generate_midi
                # writes at note end, so the bend would bleed into the next note.
                if prev.get('pitch_bends'):
                    prev['pitch_bends'] = [
                        b for b in prev['pitch_bends'] if b['time'] <= prev['end']
                    ]

        return notes

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
            valid_combos = [self._best_effort_assignment(possible_per_note)]

        # The product is unbounded in the number of notes sharing a 15ms window;
        # keep the most plausible candidates so the |prev|x|curr| Viterbi step
        # cannot blow up on a dense strum.
        if len(valid_combos) > self.max_candidates:
            valid_combos.sort(key=self._frame_position_cost)
            valid_combos = valid_combos[:self.max_candidates]

        return valid_combos

    def _best_effort_assignment(self, possible_per_note):
        """Fallback for frames with no collision-free assignment.

        Maximizes how many notes land on distinct strings (exact, via bipartite
        matching), then places any leftover note on its best placement. Simply
        taking each note's first placement -- as earlier revisions did -- could
        pile four notes onto string 6 even when five of them had a free string
        available.

        A residual collision means the frame is physically unplayable as
        detected: more than six simultaneous voices, or duplicate same-pitch
        detections that `merge_notes` would normally remove.
        """
        string_to_note = {}

        def try_assign(note_idx, seen):
            for string, _fret in possible_per_note[note_idx]:
                if string <= 0 or string in seen:
                    continue
                seen.add(string)
                held_by = string_to_note.get(string)
                if held_by is None or try_assign(held_by, seen):
                    string_to_note[string] = note_idx
                    return True
            return False

        # Most-constrained notes first, so a note with a single option is not
        # displaced by one that had alternatives.
        for note_idx in sorted(range(len(possible_per_note)),
                               key=lambda i: len(possible_per_note[i])):
            try_assign(note_idx, set())

        note_to_string = {note: string for string, note in string_to_note.items()}

        combo = []
        for note_idx, placements in enumerate(possible_per_note):
            string = note_to_string.get(note_idx)
            if string is None:
                combo.append(min(placements, key=lambda p: abs(p[1] - self.PREFERRED_FRET)))
            else:
                combo.append(next((s, f) for s, f in placements if s == string))
        return tuple(combo)

    def _frame_position_cost(self, combo):
        """Absolute (emission) cost of a candidate assignment.

        Combines a neck-position preference with an open-string bonus. The
        bonus is averaged over the frame rather than summed, so it does not
        grow with polyphony and swamp the position and travel terms.
        """
        cost = 0.0

        # Open strings do not constrain the fretting hand, so only fretted
        # notes contribute to the position estimate.
        frets = [c[1] for c in combo if c[1] > 0]
        if frets:
            avg_fret = sum(frets) / len(frets)
            cost += abs(avg_fret - self.PREFERRED_FRET)

        if combo:
            open_count = sum(1 for s, f in combo if f == 0)
            cost -= self.OPEN_STRING_BONUS * (open_count / len(combo))

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

        # Open-string preference lives in the emission term; counting it here
        # too would double-count it once per transition.
        return (fret_dist * self.FRET_TRAVEL_WEIGHT) + (string_dist * self.STRING_TRAVEL_WEIGHT)
