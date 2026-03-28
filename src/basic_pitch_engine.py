import numpy as np
import tempfile
import os
import soundfile as sf
from basic_pitch.inference import predict
from engine_base import TranscriptionEngine

class BasicPitchEngine(TranscriptionEngine):
    def transcribe(self, audio_data, sr):
        # basic-pitch inference expects a file path, so we use a temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            # Ensure audio is float32 and at correct SR
            sf.write(tmp_path, audio_data.astype(np.float32), sr)
        
        try:
            # Basic Pitch predict returns (model_output, midi_data, note_events)
            _, _, note_events = predict(audio_path=tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        formatted_notes = []
        for note in note_events:
            formatted_notes.append({
                'pitch': int(note[2]),
                'start': float(note[0]),
                'end': float(note[1]),
                'velocity': int(note[3] * 127)
            })
        return formatted_notes