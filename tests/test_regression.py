import subprocess
import pytest
import os
import pretty_midi

def test_fingerpick_regression(tmp_path):
    """Regression test for Fingerpick_mono_44-16.wav at 100 BPM."""
    input_wav = "tests/Fingerpick_mono_44-16.wav"
    output_midi = os.path.join(tmp_path, "Fingerpick_mono_44-16_regression.mid")
    python_exe = r"E:\sw_ws\repo1\audio2midi\venv\Scripts\python.exe"
    script_path = r"E:\sw_ws\repo1\audio2midi\src\audio2midi.py"
    
    # Ensure input file exists (it should, based on previous exploration)
    assert os.path.exists(input_wav), f"Input file {input_wav} not found."

    # Run the transcription
    result = subprocess.run([
        python_exe, script_path,
        input_wav, output_midi,
        "--bpm", "100"
    ], capture_output=True, text=True)
    
    # Check if command succeeded
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    assert os.path.exists(output_midi), "Output MIDI file was not created."
    
    # Load MIDI and verify note count
    pm = pretty_midi.PrettyMIDI(output_midi)
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    
    # Based on previous run, we expect around 143 notes. 
    # Allowing some margin for minor fluctuations in engine behavior if any.
    assert 130 <= total_notes <= 160, f"Expected ~143 notes, but found {total_notes}."
    
    # Check BPM
    assert pm.get_tempo_changes()[1][0] == pytest.approx(100, rel=1e-3)

def test_fingerpick_stereo_regression(tmp_path):
    """Regression test for Fingerpick_stereo_48-24.wav at 100 BPM."""
    input_wav = "tests/Fingerpick_stereo_48-24.wav"
    output_midi = os.path.join(tmp_path, "Fingerpick_stereo_48-24_regression.mid")
    python_exe = r"E:\sw_ws\repo1\audio2midi\venv\Scripts\python.exe"
    script_path = r"E:\sw_ws\repo1\audio2midi\src\audio2midi.py"
    
    assert os.path.exists(input_wav), f"Input file {input_wav} not found."

    result = subprocess.run([
        python_exe, script_path,
        input_wav, output_midi,
        "--bpm", "100"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    assert os.path.exists(output_midi), "Output MIDI file was not created."
    
    pm = pretty_midi.PrettyMIDI(output_midi)
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    
    # We also expect around 143 notes for this one.
    assert 130 <= total_notes <= 160, f"Expected ~143 notes, but found {total_notes}."
    assert pm.get_tempo_changes()[1][0] == pytest.approx(100, rel=1e-3)

def test_plektrumpick_regression(tmp_path):
    """Regression test for plektrumpick_mono_44-16.wav at 100 BPM."""
    input_wav = "tests/plektrumpick_mono_44-16.wav"
    output_midi = os.path.join(tmp_path, "plektrumpick_mono_44-16_regression.mid")
    python_exe = r"E:\sw_ws\repo1\audio2midi\venv\Scripts\python.exe"
    script_path = r"E:\sw_ws\repo1\audio2midi\src\audio2midi.py"
    
    assert os.path.exists(input_wav), f"Input file {input_wav} not found."

    result = subprocess.run([
        python_exe, script_path,
        input_wav, output_midi,
        "--bpm", "100"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    assert os.path.exists(output_midi), "Output MIDI file was not created."
    
    pm = pretty_midi.PrettyMIDI(output_midi)
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    
    # We detected 156 notes for this sample.
    assert 145 <= total_notes <= 170, f"Expected ~156 notes, but found {total_notes}."
    assert pm.get_tempo_changes()[1][0] == pytest.approx(100, rel=1e-3)

def test_plektrumstrum_regression(tmp_path):
    """Regression test for plektrumstrum_mono_44-16.wav at 100 BPM."""
    input_wav = "tests/plektrumstrum_mono_44-16.wav"
    output_midi = os.path.join(tmp_path, "plektrumstrum_mono_44-16_regression.mid")
    python_exe = r"E:\sw_ws\repo1\audio2midi\venv\Scripts\python.exe"
    script_path = r"E:\sw_ws\repo1\audio2midi\src\audio2midi.py"
    
    assert os.path.exists(input_wav), f"Input file {input_wav} not found."

    result = subprocess.run([
        python_exe, script_path,
        input_wav, output_midi,
        "--bpm", "100"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    assert os.path.exists(output_midi), "Output MIDI file was not created."
    
    pm = pretty_midi.PrettyMIDI(output_midi)
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    
    # We detected 220 notes for this sample.
    assert 210 <= total_notes <= 240, f"Expected ~220 notes, but found {total_notes}."
    assert pm.get_tempo_changes()[1][0] == pytest.approx(100, rel=1e-3)
