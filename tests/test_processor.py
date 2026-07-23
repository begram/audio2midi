import pytest
import numpy as np
from processor import load_audio, normalize_audio, high_pass_filter, noise_gate, hpss_filter, preprocess_pipeline

def test_hpss_filter():
    """Test that HPSS separates harmonic tones from percussive transients."""
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5))
    # Harmonic tone (440 Hz sine)
    harmonic = np.sin(2 * np.pi * 440 * t)
    # Transient click (short impulse)
    percussive = np.zeros_like(t)
    percussive[100:105] = 2.0

    combined = harmonic + percussive
    harmonic_extracted = hpss_filter(combined)

    assert harmonic_extracted.shape == combined.shape
    # Check that the sharp impulse peak in the harmonic output is significantly reduced compared to original
    assert np.max(np.abs(harmonic_extracted[100:105])) < np.max(np.abs(combined[100:105]))

def test_preprocess_pipeline_with_hpss(tmp_path):
    """Test full preprocess pipeline with HPSS enabled."""
    import soundfile as sf
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5))
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = str(tmp_path / "sample.wav")
    sf.write(wav_path, audio_data, sr)

    processed_audio, out_sr = preprocess_pipeline(wav_path, target_sr=sr, use_hpss=True)
    assert out_sr == sr
    assert len(processed_audio) == len(audio_data)
