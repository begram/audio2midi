"""Tests that CLI options actually reach the pipeline components.

`test_cli.py` spawns the CLI as a subprocess, which verifies argument parsing but
leaves the orchestration logic in `audio2midi.main` unmeasured. These tests drive
the command in-process with the heavy stages patched out, so no audio decoding or
model inference happens.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

import audio2midi
from audio2midi import main


@pytest.fixture
def wav(tmp_path):
    """An input path that exists; decoding is patched out."""
    path = tmp_path / "in.wav"
    path.write_text("placeholder - preprocess_pipeline is patched")
    return str(path)


@pytest.fixture
def out(tmp_path):
    return str(tmp_path / "out.mid")


class _Harness:
    def __init__(self):
        self.engine_cls = MagicMock()
        self.generate = MagicMock()
        self.set_notes([
            {'pitch': 60, 'start': 0.10, 'end': 0.60, 'velocity': 100},
            {'pitch': 64, 'start': 0.10, 'end': 0.60, 'velocity': 90},
        ])

    def set_notes(self, notes):
        self.engine_cls.return_value.transcribe.return_value = notes

    @property
    def transcribe_kwargs(self):
        return self.engine_cls.return_value.transcribe.call_args.kwargs

    @property
    def written_notes(self):
        return self.generate.call_args.args[0]

    @property
    def written_bpm(self):
        return self.generate.call_args.args[1]


@pytest.fixture
def harness():
    h = _Harness()
    fake_audio = (np.zeros(1024, dtype=np.float32), 22050)
    with patch.object(audio2midi, 'preprocess_pipeline', return_value=fake_audio), \
         patch.object(audio2midi, 'BasicPitchEngine', h.engine_cls), \
         patch.object(audio2midi, 'generate_midi', h.generate):
        yield h


def run(args):
    result = CliRunner().invoke(main, args)
    return result


# --- TR-02: engine hyperparameters reach the engine -------------------------

def test_thresholds_are_forwarded_to_the_engine(harness, wav, out):
    """TR-02: model tuning options must arrive at BasicPitchEngine.transcribe."""
    result = run([
        wav, out, '--bpm', '120',
        '--onset-threshold', '0.7',
        '--frame-threshold', '0.2',
        '--min-note-length', '50',
    ])

    assert result.exit_code == 0, result.output
    kwargs = harness.transcribe_kwargs
    assert kwargs['onset_threshold'] == pytest.approx(0.7)
    assert kwargs['frame_threshold'] == pytest.approx(0.2)
    assert kwargs['min_note_len'] == pytest.approx(50.0)


def test_pitch_bend_flag_is_forwarded(harness, wav, out):
    assert run([wav, out, '--bpm', '120']).exit_code == 0
    assert harness.transcribe_kwargs['include_pitch_bends'] is False

    assert run([wav, out, '--bpm', '120', '--pitch-bend']).exit_code == 0
    assert harness.transcribe_kwargs['include_pitch_bends'] is True


# --- frequency bound resolution --------------------------------------------

def test_no_frequency_flags_leaves_bounds_unset(harness, wav, out):
    assert run([wav, out, '--bpm', '120']).exit_code == 0

    assert harness.transcribe_kwargs['min_freq'] is None
    assert harness.transcribe_kwargs['max_freq'] is None


def test_freq_bounds_flag_applies_guitar_range(harness, wav, out):
    assert run([wav, out, '--bpm', '120', '--freq-bounds']).exit_code == 0

    assert harness.transcribe_kwargs['min_freq'] == pytest.approx(80.0)
    assert harness.transcribe_kwargs['max_freq'] == pytest.approx(1400.0)


def test_explicit_bounds_override_freq_bounds_flag(harness, wav, out):
    assert run([
        wav, out, '--bpm', '120', '--freq-bounds',
        '--min-freq', '100', '--max-freq', '900',
    ]).exit_code == 0

    assert harness.transcribe_kwargs['min_freq'] == pytest.approx(100.0)
    assert harness.transcribe_kwargs['max_freq'] == pytest.approx(900.0)


def test_one_explicit_bound_still_pairs_with_freq_bounds_default(harness, wav, out):
    assert run([wav, out, '--bpm', '120', '--freq-bounds', '--min-freq', '120']).exit_code == 0

    assert harness.transcribe_kwargs['min_freq'] == pytest.approx(120.0)
    assert harness.transcribe_kwargs['max_freq'] == pytest.approx(1400.0)


def test_inverted_explicit_bounds_are_rejected(harness, wav, out):
    result = run([wav, out, '--bpm', '120', '--min-freq', '900', '--max-freq', '100'])

    assert result.exit_code != 0
    assert "must be below maximum frequency" in result.output
    harness.engine_cls.return_value.transcribe.assert_not_called()


# --- pipeline stage wiring --------------------------------------------------

def test_merge_is_applied_by_default_and_skipped_with_no_merge(harness, wav, out):
    """Two chattering same-pitch notes collapse unless --no-merge is given."""
    chatter = [
        {'pitch': 60, 'start': 0.00, 'end': 0.50, 'velocity': 100},
        {'pitch': 60, 'start': 0.45, 'end': 1.00, 'velocity': 90},
        {'pitch': 62, 'start': 0.00, 'end': 1.00, 'velocity': 90},
    ]

    harness.set_notes(list(chatter))
    assert run([wav, out, '--bpm', '120']).exit_code == 0
    assert len(harness.written_notes) == 2

    harness.set_notes(list(chatter))
    assert run([wav, out, '--bpm', '120', '--no-merge']).exit_code == 0
    assert len(harness.written_notes) == 3


def test_quantize_rewrites_note_times_in_seconds(harness, wav, out):
    """--quantize must reach generate_midi as snapped seconds, not raw ticks."""
    assert run([wav, out, '--bpm', '120', '--quantize', '1/4']).exit_code == 0

    starts = sorted(n['start'] for n in harness.written_notes)
    assert starts[0] == pytest.approx(0.0)  # 0.10s snaps to the 1/4 grid at 120 BPM
    assert all(n['end'] > n['start'] for n in harness.written_notes)


def test_tab_flag_assigns_strings(harness, wav, out):
    assert run([wav, out, '--bpm', '120', '--tab']).exit_code == 0

    assert all('string' in n for n in harness.written_notes)


def test_without_tab_no_string_is_assigned(harness, wav, out):
    assert run([wav, out, '--bpm', '120']).exit_code == 0

    assert all('string' not in n for n in harness.written_notes)


def test_velocity_curve_boosts_midrange(harness, wav, out):
    harness.set_notes([{'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 64}])

    assert run([wav, out, '--bpm', '120', '--velocity-curve', '5.0']).exit_code == 0

    assert harness.written_notes[0]['velocity'] > 64


def test_bpm_is_passed_through_to_the_generator(harness, wav, out):
    assert run([wav, out, '--bpm', '96']).exit_code == 0

    assert harness.written_bpm == 96


def test_instrument_name_is_passed_through(harness, wav, out):
    assert run([wav, out, '--bpm', '120', '--instrument', 'Nylon']).exit_code == 0

    assert harness.generate.call_args.kwargs['instrument_name'] == 'Nylon'


# --- failure paths ----------------------------------------------------------

def test_unwritable_output_reports_cleanly(harness, wav, out):
    """An OSError from the writer becomes a message, not a traceback."""
    harness.generate.side_effect = OSError("no such directory")

    result = run([wav, out, '--bpm', '120'])

    assert result.exit_code != 0
    assert "Could not write" in result.output
    assert "Traceback" not in result.output


def test_undecodable_audio_reports_cleanly(wav, out):
    with patch.object(audio2midi, 'preprocess_pipeline', side_effect=RuntimeError("bad header")):
        result = run([wav, out, '--bpm', '120'])

    assert result.exit_code != 0
    assert "Could not read" in result.output
    assert "bad header" in result.output


# --- option range validation ------------------------------------------------

@pytest.mark.parametrize("option,value", [
    ('--bpm', '0'),
    ('--bpm', '500'),
    ('--quantize-strength', '1.5'),
    ('--quantize-strength', '-0.1'),
    ('--velocity-threshold', '200'),
    ('--velocity-threshold', '-1'),
    ('--noise-threshold', '2.0'),
    ('--onset-threshold', '1.5'),
    ('--frame-threshold', '-0.2'),
    ('--min-duration', '-0.1'),
    ('--velocity-curve', '-1.0'),
    ('--min-freq', '0'),
])
def test_out_of_range_options_are_rejected(harness, wav, out, option, value):
    args = [wav, out, option, value]
    if option != '--bpm':
        args += ['--bpm', '120']

    result = run(args)

    assert result.exit_code != 0, f"{option} {value} should have been rejected"
    assert "Invalid value" in result.output
    harness.engine_cls.return_value.transcribe.assert_not_called()


def test_invalid_quantize_grid_is_rejected(harness, wav, out):
    result = run([wav, out, '--bpm', '120', '--quantize', '1/7'])

    assert result.exit_code != 0
    assert "Invalid value" in result.output
