from abc import ABC, abstractmethod


class TranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_data, sr, **kwargs):
        """Returns a list of note event dicts.

        Each dict carries 'pitch' (int, MIDI note number), 'start' and 'end'
        (float, seconds), 'velocity' (int, 1-127), and optionally 'pitch_bends':
        a list of {'pitch': int, 'time': float seconds}, where 'pitch' is in MIDI
        pitch-wheel units (-8192..8191, 4096 per semitone, i.e. assuming the
        default +-2 semitone bend range) -- not time ticks.

        This is the note-dict contract every later stage reads and returns. Later
        stages only add keys: `string`/`fret` (TabMapper.assign_strings) and
        `start_ticks`/`end_ticks` (post_process.quantize_notes).

        Engine-specific tuning is passed through **kwargs; see
        BasicPitchEngine.transcribe for the options it accepts.
        """
