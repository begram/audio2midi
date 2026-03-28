import librosa
import numpy as np
from scipy.signal import butter, lfilter

def load_audio(file_path, target_sr=22050):
    """Loads 16/24-bit .wav files and resamples to target_sr."""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

def normalize_audio(audio, target_db=-1.0):
    """Peak normalizes audio to target_db."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        multiplier = 10**(target_db / 20)
        return (audio / max_val) * multiplier
    return audio

def high_pass_filter(audio, sr, cutoff=80):
    """Applies a Butterworth high-pass filter (80Hz default)."""
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, audio)

def noise_gate(audio, threshold=0.005):
    """Zeroes out samples below the absolute amplitude threshold."""
    audio = audio.copy()
    audio[np.abs(audio) < threshold] = 0
    return audio

def preprocess_pipeline(file_path, target_sr=22050, noise_threshold=0.005):
    """Full pre-processing pipeline for the transcription engine."""
    audio, sr = load_audio(file_path, target_sr)
    audio = normalize_audio(audio)
    audio = high_pass_filter(audio, sr)
    if noise_threshold > 0:
        audio = noise_gate(audio, threshold=noise_threshold)
    return audio, sr