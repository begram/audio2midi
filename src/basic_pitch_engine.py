import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
from basic_pitch.inference import predict

from engine_base import TranscriptionEngine

# Basic Pitch emits per-note bend contours in units of 1/3 semitone; MIDI pitch
# bend uses 4096 ticks per semitone over a 14-bit signed range.
CONTOUR_BINS_PER_SEMITONE = 3
PITCH_BEND_SCALE = 4096
PITCH_BEND_MIN = -8192
PITCH_BEND_MAX = 8191


def _format_pitch_bends(bend_values, start, end):
    """Converts a note's raw bend contour into MIDI pitch bend events.

    Basic Pitch attaches the contour to the note itself, so events are
    unambiguously owned by one note -- no time-window matching, which would
    copy a single string's bend onto every simultaneous note.
    """
    if bend_values is None or len(bend_values) == 0:
        return []

    ticks = np.round(
        np.asarray(bend_values, dtype=np.float64)
        * PITCH_BEND_SCALE
        / CONTOUR_BINS_PER_SEMITONE
    )
    ticks = np.clip(ticks, PITCH_BEND_MIN, PITCH_BEND_MAX).astype(int)
    times = np.linspace(float(start), float(end), len(ticks))

    return [
        {'pitch': int(p), 'time': float(t)}
        for p, t in zip(ticks, times, strict=True)
    ]


class BasicPitchEngine(TranscriptionEngine):
    def transcribe(
        self, audio_data, sr,
        min_freq=None, max_freq=None,
        onset_threshold=0.5, frame_threshold=0.3,
        min_note_len=127.70, include_pitch_bends=False
    ):
        """
        Transcribes audio data using Spotify's Basic Pitch.
        Restricts frequency range to [min_freq, max_freq] and optionally extracts pitch bend events.
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_data.astype(np.float32), sr)

        try:
            _model_output, _midi_data, note_events = predict(
                audio_path=tmp_path,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=min_note_len,
                minimum_frequency=min_freq,
                maximum_frequency=max_freq
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        min_pitch = int(np.floor(librosa.hz_to_midi(min_freq))) if min_freq is not None else 0
        max_pitch = int(np.ceil(librosa.hz_to_midi(max_freq))) if max_freq is not None else 127

        formatted_notes = []
        for note in note_events:
            pitch = int(note[2])
            if pitch < min_pitch or pitch > max_pitch:
                continue

            note_start = float(note[0])
            note_end = float(note[1])

            note_dict = {
                'pitch': pitch,
                'start': note_start,
                'end': note_end,
                # Clamp to 1: a velocity of 0 is a note-off in most DAWs.
                'velocity': int(np.clip(round(note[3] * 127), 1, 127))
            }

            if include_pitch_bends:
                bend_values = note[4] if len(note) > 4 else None
                note_dict['pitch_bends'] = _format_pitch_bends(
                    bend_values, note_start, note_end
                )

            formatted_notes.append(note_dict)

        return formatted_notes
