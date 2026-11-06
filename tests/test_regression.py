import os
import subprocess

import pretty_midi
import pytest

# (fixture wav, measured note count, tolerance band). Counts are what the CLI defaults
# produce today (ONNX model, merging off); each band is roughly +-8% around it, so a real
# DSP or threshold change trips it while run-to-run numeric noise does not. Re-measure and
# update deliberately when defaults change.
REGRESSION_CASES = [
    ("Fingerpick_mono_44-16.wav", 139, (128, 150)),
    ("Fingerpick_stereo_48-24.wav", 139, (128, 150)),
    ("plektrumpick_mono_44-16.wav", 155, (143, 167)),
    ("plektrumstrum_mono_44-16.wav", 211, (194, 228)),
]


@pytest.mark.slow
@pytest.mark.parametrize("wav_name,expected,band", REGRESSION_CASES)
def test_transcription_regression(
    wav_name, expected, band, tmp_path, python_exe, cli_script, fixture_wav
):
    """End-to-end note-count regression at 100 BPM for each reference recording.

    Runs the CLI with its defaults only, so the bands guard what a user actually gets.
    """
    input_wav = fixture_wav(wav_name)
    output_midi = os.path.join(tmp_path, f"{wav_name}_regression.mid")

    result = subprocess.run(
        [
            python_exe, cli_script,
            input_wav, output_midi,
            "--bpm", "100",
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
