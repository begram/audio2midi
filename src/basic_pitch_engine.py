import numpy as np
from basic_pitch.inference import predict
from engine_base import TranscriptionEngine

class BasicPitchEngine(TranscriptionEngine):
    def transcribe(self, audio_data, sr):
        # Basic Pitch predict returns (model_output, midi_data, note_events)
        _, _, note_events = predict(audio_path_or_array=audio_data)
        
        formatted_notes = []
        for note in note_events:
            formatted_notes.append({
                'pitch': int(note[2]),
                'start': float(note[0]),
                'end': float(note[1]),
                'velocity': int(note[3] * 127)
            })
        return formatted_notes