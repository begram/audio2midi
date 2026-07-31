from abc import ABC, abstractmethod


class TranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_data, sr, **kwargs):
        """Returns a list of note event dicts.

        Each dict carries 'pitch' (int, MIDI note number), 'start' and 'end'
        (float, seconds), 'velocity' (int, 1-127), and optionally
        'pitch_bends' (list of {'pitch': int MIDI ticks, 'time': float seconds}).

        Engine-specific tuning is passed through **kwargs; see
        BasicPitchEngine.transcribe for the options it accepts.
        """
