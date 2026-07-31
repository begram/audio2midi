import numpy as np
import pytest
import soundfile as sf

from processor import (
    high_pass_filter,
    hpss_filter,
    load_audio,
    noise_gate,
    normalize_audio,
    preprocess_pipeline,
)


def _rms_at(signal, sr, freq):
    """Magnitude of `signal` at `freq`, via a single-bin Goertzel-style projection."""
    n = np.arange(len(signal))
    ref = np.exp(-2j * np.pi * freq * n / sr)
    return np.abs(np.sum(signal * ref)) / len(signal)


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
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5))
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = str(tmp_path / "sample.wav")
    sf.write(wav_path, audio_data, sr)

    processed_audio, out_sr = preprocess_pipeline(wav_path, target_sr=sr, use_hpss=True)
    assert out_sr == sr
    assert len(processed_audio) == len(audio_data)


def test_preprocess_pipeline_without_hpss_or_gate(tmp_path):
    """HPSS and the gate are both optional; noise_threshold=0 skips gating."""
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = str(tmp_path / "sample.wav")
    sf.write(wav_path, audio_data, sr)

    processed, out_sr = preprocess_pipeline(
        wav_path, target_sr=sr, noise_threshold=0.0, use_hpss=False
    )
    assert out_sr == sr
    assert len(processed) == len(audio_data)


# --- high-pass filter -------------------------------------------------------

def test_high_pass_filter_attenuates_below_cutoff():
    """The filter must actually reject sub-cutoff content, not just preserve shape."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    rumble, tone = 20.0, 440.0
    audio = (np.sin(2 * np.pi * rumble * t) + np.sin(2 * np.pi * tone * t)).astype(np.float64)

    filtered = high_pass_filter(audio, sr, cutoff=80)

    # 440 Hz passes nearly untouched; 20 Hz is strongly attenuated.
    assert _rms_at(filtered, sr, tone) > 0.9 * _rms_at(audio, sr, tone)
    assert _rms_at(filtered, sr, rumble) < 0.35 * _rms_at(audio, sr, rumble)


def test_high_pass_filter_is_monotonic_in_frequency():
    """Attenuation should decrease as frequency rises through the cutoff."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)

    gains = []
    for freq in (10.0, 40.0, 80.0, 320.0, 1000.0):
        tone = np.sin(2 * np.pi * freq * t)
        gains.append(_rms_at(high_pass_filter(tone, sr, cutoff=80), sr, freq))

    assert gains == sorted(gains), f"gain not monotonic across frequency: {gains}"


def test_high_pass_filter_preserves_first_sample():
    """y[0] == x[0]: the vectorized form must match the direct recurrence.

    A bare lfilter call with zero initial state gives y[0] = alpha * x[0]
    instead, which this pins down.
    """
    sr = 22050
    audio = np.array([0.7, -0.2, 0.4, 0.1, -0.6], dtype=np.float64)

    filtered = high_pass_filter(audio, sr, cutoff=80)

    assert filtered[0] == pytest.approx(audio[0])


def test_high_pass_filter_matches_direct_recurrence():
    """The vectorized IIR must equal the textbook one-pole loop sample-for-sample."""
    sr = 22050
    cutoff = 80
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(2048)

    dt = 1.0 / sr
    rc = 1.0 / (2.0 * np.pi * cutoff)
    alpha = rc / (rc + dt)
    expected = np.empty_like(audio)
    expected[0] = audio[0]
    for i in range(1, len(audio)):
        expected[i] = alpha * (expected[i - 1] + audio[i] - audio[i - 1])

    np.testing.assert_allclose(high_pass_filter(audio, sr, cutoff), expected, rtol=1e-9)


def test_high_pass_filter_handles_empty_input():
    empty = np.array([], dtype=np.float32)
    assert len(high_pass_filter(empty, 22050)) == 0


# --- normalization ----------------------------------------------------------

def test_normalize_audio_leaves_silence_alone():
    """Silent input must not divide by zero."""
    silence = np.zeros(128, dtype=np.float32)
    np.testing.assert_array_equal(normalize_audio(silence), silence)


# --- noise gate edge cases --------------------------------------------------

def test_noise_gate_handles_audio_shorter_than_hop():
    """Very short buffers yield a single envelope frame and must not crash."""
    audio = np.array([0.4, -0.5, 0.45, -0.42], dtype=np.float32)

    gated = noise_gate(audio, threshold=0.01)

    assert gated.shape == audio.shape
    assert np.any(gated != 0)


def test_noise_gate_handles_empty_input():
    empty = np.array([], dtype=np.float32)
    assert len(noise_gate(empty, threshold=0.01)) == 0


# --- audio loading ----------------------------------------------------------

@pytest.mark.parametrize("in_sr,subtype", [
    (44100, "PCM_16"),
    (48000, "PCM_24"),
    (96000, "PCM_24"),
])
def test_load_audio_resamples_and_downmixes(tmp_path, in_sr, subtype):
    """README promises 16/24-bit at 44.1/48/96 kHz; all land as mono at target_sr."""
    duration = 0.25
    t = np.linspace(0, duration, int(in_sr * duration), endpoint=False)
    stereo = np.stack([
        0.5 * np.sin(2 * np.pi * 440 * t),
        0.5 * np.sin(2 * np.pi * 440 * t + np.pi / 3),
    ], axis=-1)

    wav_path = str(tmp_path / f"sample_{in_sr}_{subtype}.wav")
    sf.write(wav_path, stereo, in_sr, subtype=subtype)

    audio, sr = load_audio(wav_path, target_sr=22050)

    assert sr == 22050
    assert audio.ndim == 1, "stereo input must be down-mixed to mono"
    assert len(audio) == pytest.approx(22050 * duration, rel=0.02)
    # The 440 Hz content survives the resample.
    assert _rms_at(audio, sr, 440.0) > 0.1
