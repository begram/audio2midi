import librosa
import numpy as np
from scipy.signal import lfilter


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
    """Applies a 1st-order (one-pole) high-pass IIR filter, 80Hz default.

    Implements y[n] = alpha * (y[n-1] + x[n] - x[n-1]) as a vectorized IIR:
    b = [alpha, -alpha], a = [1, -alpha]. The initial condition is chosen so
    that y[0] == x[0], matching the direct recurrence exactly.
    """
    if len(audio) == 0:
        return audio
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * np.pi * cutoff)
    alpha = rc / (rc + dt)

    zi = [(1.0 - alpha) * audio[0]]
    filtered, _ = lfilter([alpha, -alpha], [1.0, -alpha], audio, zi=zi)
    return filtered.astype(audio.dtype, copy=False)

def noise_gate(audio, threshold=0.005, frame_length=512, hop_length=128, smoothing_frames=5):
    """Attenuates passages whose short-term RMS falls below `threshold`.

    Gates on a smoothed RMS envelope rather than on individual samples.
    Sample-wise gating clips the waveform at every zero-crossing, and the
    resulting step discontinuities produce broadband harmonic distortion --
    exactly the spectral junk that Basic Pitch reports as phantom notes.
    """
    if threshold <= 0 or len(audio) == 0:
        return audio

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    gain = (rms >= threshold).astype(np.float64)

    # Smooth the binary gain curve into an attack/release ramp so the gate
    # opens and closes without clicking. Edge-pad first: convolving against
    # implicit zeros would fade in and out at every file boundary and clip the
    # attack of the first note.
    if smoothing_frames > 1 and len(gain) >= smoothing_frames:
        window = np.hanning(smoothing_frames)
        window /= window.sum()
        pad = smoothing_frames // 2
        padded = np.pad(gain, pad, mode='edge')
        gain = np.convolve(padded, window, mode='same')[pad:pad + len(rms)]

    if len(gain) == 1:
        gain_samples = np.full(len(audio), gain[0])
    else:
        frame_centers = np.arange(len(gain)) * hop_length
        gain_samples = np.interp(np.arange(len(audio)), frame_centers, gain)

    return (audio * gain_samples).astype(audio.dtype, copy=False)

def hpss_filter(audio, margin=1.0):
    """
    Applies Harmonic-Percussive Source Separation (HPSS) to extract
    the harmonic component of the audio, filtering out percussive attack transients.
    """
    harmonic, _ = librosa.effects.hpss(audio, margin=margin)
    return harmonic

def preprocess_pipeline(file_path, target_sr=22050, noise_threshold=0.005, use_hpss=False):
    """Full pre-processing pipeline for the transcription engine."""
    audio, sr = load_audio(file_path, target_sr)
    audio = normalize_audio(audio)
    audio = high_pass_filter(audio, sr)
    if use_hpss:
        audio = hpss_filter(audio)
    if noise_threshold > 0:
        audio = noise_gate(audio, threshold=noise_threshold)
    return audio, sr
