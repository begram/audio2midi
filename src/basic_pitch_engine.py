import numpy as np
import tempfile
import os
import soundfile as sf
import librosa
from basic_pitch.inference import predict
from engine_base import TranscriptionEngine

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
            model_output, midi_data, note_events = predict(
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

        pitch_bends = []
        if include_pitch_bends and midi_data is not None:
            for inst in midi_data.instruments:
                for pb in inst.pitch_bends:
                    pitch_bends.append({'pitch': pb.pitch, 'time': pb.time})

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
                'velocity': int(note[3] * 127)
            }

            if include_pitch_bends:
                note_pbs = [pb for pb in pitch_bends if note_start <= pb['time'] <= note_end]
                note_dict['pitch_bends'] = note_pbs

            formatted_notes.append(note_dict)

        return formatted_notes