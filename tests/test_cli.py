import subprocess
import pytest
import os

def test_cli_help():
    """Test that 'python audio2midi.py --help' works."""
    result = subprocess.run([
        r"E:\sw_ws\repo1\audio2midi\venv\Scripts\python.exe", 
        r"E:\sw_ws\repo1\audio2midi\src\audio2midi.py", 
        "--help"
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage:" in result.stdout

def test_cli_missing_bpm(tmp_path):
    """Test that missing --bpm argument causes an error when files exist."""
    # Create an actual dummy wav file so we get past the path check
    dummy_wav = os.path.join(tmp_path, "dummy.wav")
    with open(dummy_wav, "w") as f:
        f.write("not really audio but file exists")
        
    result = subprocess.run([
        r"E:\sw_ws\repo1\audio2midi\venv\Scripts\python.exe", 
        r"E:\sw_ws\repo1\audio2midi\src\audio2midi.py", 
        dummy_wav, "out.mid"
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Error: Missing option '--bpm'" in result.stderr