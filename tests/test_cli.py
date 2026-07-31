import os
import subprocess

import pytest


@pytest.fixture
def dummy_wav(tmp_path):
    """A file that exists and ends in .wav but is not decodable audio."""
    path = os.path.join(tmp_path, "dummy.wav")
    with open(path, "w") as f:
        f.write("not really audio but file exists")
    return path


def test_cli_help(python_exe, cli_script):
    """Test that 'python audio2midi.py --help' works."""
    result = subprocess.run(
        [python_exe, cli_script, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_cli_missing_bpm(python_exe, cli_script, dummy_wav):
    """Test that missing --bpm argument causes an error when files exist."""
    result = subprocess.run(
        [python_exe, cli_script, dummy_wav, "out.mid"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Error: Missing option '--bpm'" in result.stderr


def test_cli_rejects_out_of_range_bpm(python_exe, cli_script, dummy_wav):
    """A BPM of 0 must be rejected up front, not divided by downstream."""
    result = subprocess.run(
        [python_exe, cli_script, dummy_wav, "out.mid", "--bpm", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Invalid value" in result.stderr


def test_cli_rejects_inverted_frequency_bounds(python_exe, cli_script, dummy_wav):
    """--min-freq above --max-freq must be caught before inference."""
    result = subprocess.run(
        [
            python_exe, cli_script, dummy_wav, "out.mid",
            "--bpm", "120", "--min-freq", "1400", "--max-freq", "80",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be below maximum frequency" in result.stderr


def test_cli_reports_unreadable_audio_cleanly(tmp_path, python_exe, cli_script, dummy_wav):
    """A non-audio input should produce a message, not a raw traceback."""
    result = subprocess.run(
        [
            python_exe, cli_script, dummy_wav,
            os.path.join(tmp_path, "out.mid"), "--bpm", "120",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Could not read" in result.stderr
    assert "Traceback" not in result.stderr
