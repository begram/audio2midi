import os
import subprocess

import pretty_midi
import pytest

# (fixture wav, expected note count, tolerance band)
REGRESSION_CASES = [
    ("Fingerpick_mono_44-16.wav", 143, (130, 160)),
    ("Fingerpick_stereo_48-24.wav", 143, (130, 160)),
    ("plektrumpick_mono_44-16.wav", 156, (145, 170)),
    ("plektrumstrum_mono_44-16.wav", 220, (210, 240)),
]


@pytest.mark.slow
@pytest.mark.parametrize("wav_name,expected,band", REGRESSION_CASES)
def test_transcription_regression(
    wav_name, expected, band, tmp_path, python_exe, cli_script, fixture_wav
):
    """End-to-end note-count regression at 100 BPM for each reference recording."""
    input_wav = fixture_wav(wav_name)
    output_midi = os.path.join(tmp_path, f"{wav_name}_regression.mid")

    result = subprocess.run(
        [
            python_exe, cli_script,
            input_wav, output_midi,
            "--bpm", "100", "--no-merge",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    assert os.path.exists(output_midi), "Output MIDI file was not created."

    pm = pretty_midi.PrettyMIDI(output_midi)
    total_notes = sum(len(inst.notes) for inst in pm.instruments)

    low, high = band
    assert low <= total_notes <= high, (
        f"Expected ~{expected} notes for {wav_name}, but found {total_notes}."
    )
    assert pm.get_tempo_changes()[1][0] == pytest.approx(100, rel=1e-3)
