from abc import ABC, abstractmethod

class TranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_data, sr):
        """Returns a list of note events (pitch, onset, offset, amplitude)."""
        pass