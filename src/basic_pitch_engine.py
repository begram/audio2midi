import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import predict

from engine_base import TranscriptionEngine

# Pinned to the ONNX export of the model. Left to its default, basic-pitch picks whichever
# runtime happens to be importable (TensorFlow first), and its own dependencies install a
# different one per platform -- onnxruntime on Windows, tflite-runtime on Linux -- so the
# same command could run different backends on different machines. The ONNX model gives
# note-for-note identical output to the TensorFlow SavedModel on every fixture, and
# onnxruntime is ~1 GB lighter than TensorFlow.
MODEL_PATH = build_icassp_2022_model_path(FilenameSuffix.onnx)

# Note-creation defaults. The CLI imports these, so its defaults and the engine's cannot
# drift apart (the analysis tools' "baseline" preset uses the engine's).
#
# These are Basic Pitch's own general-purpose values, kept deliberately -- do not "fix" them
# from GuitarSet numbers alone. Stricter values (CLEAN_RECORDING_* below) are measurably more
# accurate on GuitarSet: swept on the dev split (players 00-03) they sit on a broad plateau,
# and on the held-out players (04-05, 120 tracks) they raise onset F1 0.775 -> 0.822. But
# GuitarSet is recorded under controlled conditions, and on the project's own recordings in
# tests/ (no ground truth; scored against onsets detected in the audio) every stricter
# setting found fewer of the audible note events (recall 0.692 -> 0.612) and none agreed
# better with the audio. Stricter thresholds mainly drop quiet and short notes, which is
# what an imperfect recording renders faintly, and a missed note costs a user more than a
# spurious one. So the defaults favour recall; the clean-recording values are documented
# in the README as opt-in flags.
DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MIN_NOTE_LENGTH_MS = 127.7

# Opt-in values for clean, GuitarSet-like recordings (good mic, quiet room). Costs on the
# held-out players, both small: offset F1 0.552 -> 0.525, --tab string accuracy 0.661 ->
# 0.658. 58 ms is 5 model frames at Basic Pitch's 86.13 frames per second.
CLEAN_RECORDING_ONSET_THRESHOLD = 0.6
CLEAN_RECORDING_FRAME_THRESHOLD = 0.5
CLEAN_RECORDING_MIN_NOTE_LENGTH_MS = 58.0

# Basic Pitch emits per-note bend contours in units of 1/3 semitone. MIDI pitch-wheel
# values span a 14-bit signed range; 4096 per semitone assumes the receiver's default
# +-2 semitone bend range, so bends wider than 2 semitones are clipped.
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
        onset_threshold=DEFAULT_ONSET_THRESHOLD, frame_threshold=DEFAULT_FRAME_THRESHOLD,
        min_note_len=DEFAULT_MIN_NOTE_LENGTH_MS, include_pitch_bends=False
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
                model_or_model_path=MODEL_PATH,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=min_note_len,
                minimum_frequency=min_freq,
                maximum_frequency=max_freq
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Mirror Basic Pitch's own bounding (note_creation.constrain_frequency), which keeps
        # pitch bins from round(min) up to but excluding round(max). 80-1400 Hz (MIDI 39.49
        # and 89.04) therefore means MIDI 39-88: D#2, a semitone of slack under the low E,
        # up to E6. A looser floor/ceil here would contradict the engine and admit 89-90.
        min_pitch = int(np.round(librosa.hz_to_midi(min_freq))) if min_freq is not None else 0
        max_pitch = (int(np.round(librosa.hz_to_midi(max_freq))) - 1
                     if max_freq is not None else 127)

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
