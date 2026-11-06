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
        self.preprocess = MagicMock(return_value=(np.zeros(1024, dtype=np.float32), 22050))
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
    with patch.object(audio2midi, 'preprocess_pipeline', h.preprocess), \
         patch.object(audio2midi, 'BasicPitchEngine', h.engine_cls), \
         patch.object(audio2midi, 'generate_midi', h.generate):
        yield h


@pytest.fixture
def real_writer():
    """Like `harness`, but the real generate_midi runs, so the file on disk can be checked."""
    h = _Harness()
    with patch.object(audio2midi, 'preprocess_pipeline', h.preprocess), \
         patch.object(audio2midi, 'BasicPitchEngine', h.engine_cls):
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


def test_cli_defaults_are_the_engines_defaults(harness, wav, out):
    """One source of truth: the analysis tools' "baseline" preset uses the engine's own
    defaults, so if the CLI's drifted from them the tools would measure a different
    configuration than users get.

    The values are Basic Pitch's own, kept on purpose for imperfect recordings; see the
    comment on DEFAULT_ONSET_THRESHOLD before changing them.
    """
    import basic_pitch_engine as bpe

    assert run([wav, out, '--bpm', '120']).exit_code == 0

    kwargs = harness.transcribe_kwargs
    assert kwargs['onset_threshold'] == bpe.DEFAULT_ONSET_THRESHOLD == 0.5
    assert kwargs['frame_threshold'] == bpe.DEFAULT_FRAME_THRESHOLD == 0.3
    assert kwargs['min_note_len'] == bpe.DEFAULT_MIN_NOTE_LENGTH_MS == 127.7


def test_help_shows_every_default():
    """Defaults appear for all options, not only the ones someone remembered to flag."""
    result = run(['--help'])

    assert result.exit_code == 0
    for shown in ('default: 0.03', 'default: 10', 'default: 0.005', 'default: 1.0',
                  'default: 0.5', 'default: 0.3', 'default: 127.7', 'Acoustic Guitar'):
        assert shown in result.output, f"--help is missing {shown!r}"


def test_clean_recording_values_are_accepted_and_forwarded(harness, wav, out):
    """The README's opt-in flags for clean recordings must reach the engine as written."""
    import basic_pitch_engine as bpe

    assert run([
        wav, out, '--bpm', '120',
        '--onset-threshold', str(bpe.CLEAN_RECORDING_ONSET_THRESHOLD),
        '--frame-threshold', str(bpe.CLEAN_RECORDING_FRAME_THRESHOLD),
        '--min-note-length', str(bpe.CLEAN_RECORDING_MIN_NOTE_LENGTH_MS),
    ]).exit_code == 0

    kwargs = harness.transcribe_kwargs
    assert kwargs['onset_threshold'] == pytest.approx(0.6)
    assert kwargs['frame_threshold'] == pytest.approx(0.5)
    assert kwargs['min_note_len'] == pytest.approx(58.0)


def test_pitch_bend_flag_is_forwarded(harness, wav, out):
    assert run([wav, out, '--bpm', '120']).exit_code == 0
    assert harness.transcribe_kwargs['include_pitch_bends'] is False

    assert run([wav, out, '--bpm', '120', '--pitch-bend']).exit_code == 0
    assert harness.transcribe_kwargs['include_pitch_bends'] is True


def test_pitch_bend_without_tab_warns_about_shared_channel(harness, wav, out):
    """Without --tab every note shares one channel, so bends leak between notes."""
    result = run([wav, out, '--bpm', '120', '--pitch-bend'])
    assert result.exit_code == 0
    assert "one MIDI channel" in result.output

    result = run([wav, out, '--bpm', '120', '--pitch-bend', '--tab'])
    assert result.exit_code == 0
    assert "one MIDI channel" not in result.output


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

def test_merge_is_off_by_default_and_applied_with_merge(harness, wav, out):
    """Merging is opt-in: it absorbs repeated notes, so the default must keep both.

    The pair below is what the rule catches -- a note sustaining until the next time the
    same pitch is played. GuitarSet shows those are usually genuine repeats, so merging
    them costs more recall than the precision it buys.
    """
    repeated = [
        {'pitch': 60, 'start': 0.00, 'end': 0.50, 'velocity': 100},
        {'pitch': 60, 'start': 0.45, 'end': 1.00, 'velocity': 90},
        {'pitch': 62, 'start': 0.00, 'end': 1.00, 'velocity': 90},
    ]

    harness.set_notes(list(repeated))
    assert run([wav, out, '--bpm', '120']).exit_code == 0
    assert len(harness.written_notes) == 3

    harness.set_notes(list(repeated))
    assert run([wav, out, '--bpm', '120', '--merge']).exit_code == 0
    assert len(harness.written_notes) == 2

    # --no-merge still parses, so existing invocations keep working.
    harness.set_notes(list(repeated))
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


def test_preprocessing_options_reach_the_pipeline(harness, wav, out):
    assert run([wav, out, '--bpm', '120']).exit_code == 0
    assert harness.preprocess.call_args.kwargs == {'noise_threshold': 0.005, 'use_hpss': False}

    assert run([wav, out, '--bpm', '120', '--hpss', '--noise-threshold', '0.02']).exit_code == 0
    assert harness.preprocess.call_args.kwargs == {'noise_threshold': 0.02, 'use_hpss': True}


def test_cleaning_thresholds_reach_clean_notes(harness, wav, out):
    with patch.object(audio2midi, 'clean_notes', wraps=audio2midi.clean_notes) as spy:
        assert run([
            wav, out, '--bpm', '120', '--min-duration', '0.2', '--velocity-threshold', '50',
        ]).exit_code == 0

    assert spy.call_args.kwargs == {'min_duration': 0.2, 'velocity_threshold': 50}


def test_quantize_strength_reaches_the_quantizer(harness, wav, out):
    with patch.object(audio2midi, 'quantize_notes', wraps=audio2midi.quantize_notes) as spy:
        assert run([
            wav, out, '--bpm', '120', '--quantize', '1/8', '--quantize-strength', '0.4',
        ]).exit_code == 0

    assert spy.call_args.kwargs == {'grid_resolution': '1/8', 'strength': pytest.approx(0.4)}


def test_velocity_threshold_applies_before_the_velocity_curve(harness, wav, out):
    """CLAUDE.md ordering rule 3: the threshold judges raw engine confidence.

    Velocity 8 is below the threshold of 10, but the curve would lift it to about 19. If the
    curve ran first, this note would wrongly survive.
    """
    harness.set_notes([
        {'pitch': 60, 'start': 0.0, 'end': 0.5, 'velocity': 8},
        {'pitch': 64, 'start': 0.0, 'end': 0.5, 'velocity': 100},
    ])

    assert run([
        wav, out, '--bpm', '120', '--velocity-threshold', '10', '--velocity-curve', '5.0',
    ]).exit_code == 0

    assert [n['pitch'] for n in harness.written_notes] == [64]


def test_bpm_is_passed_through_to_the_generator(harness, wav, out):
    assert run([wav, out, '--bpm', '96']).exit_code == 0

    assert harness.written_bpm == 96


def test_instrument_name_is_passed_through(harness, wav, out):
    assert run([wav, out, '--bpm', '120', '--instrument', 'Nylon']).exit_code == 0

    assert harness.generate.call_args.kwargs['instrument_name'] == 'Nylon'


# --- whole pipeline, real MIDI writer ----------------------------------------

def test_full_pipeline_output_keeps_strings_and_bends_consistent(real_writer, wav, out):
    """Every stage after the engine runs for real, and the written file is checked.

    Each stage has its own unit tests; this checks what they produce together, which is
    where the ordering rules in CLAUDE.md matter. The input sustains notes into later plucks
    on the only string that can play them (forcing overlap truncation), carries bends of
    both signs that end exactly on the note's end, and is quantized, which lines note
    boundaries up on shared ticks.
    """
    import pretty_midi

    def bent(pitch, start, end, first, last):
        return {'pitch': pitch, 'start': start, 'end': end, 'velocity': 100,
                'pitch_bends': [{'pitch': first, 'time': start}, {'pitch': last, 'time': end}]}

    real_writer.set_notes([
        bent(40, 0.00, 1.40, 400, 900),     # low E: string 6 only, sustains into...
        bent(41, 0.52, 1.60, -600, 300),    # ...F2, fret 1 on string 6 (truncates the E)
        bent(40, 1.01, 2.00, -300, 700),    # E again, string 6, truncates the F
        {'pitch': 64, 'start': 0.00, 'end': 2.0, 'velocity': 90},  # held high e
    ])

    result = run([wav, out, '--bpm', '120', '--quantize', '1/8', '--tab', '--pitch-bend'])
    assert result.exit_code == 0, result.output

    pm = pretty_midi.PrettyMIDI(out)
    for inst in pm.instruments:
        if 'String' not in inst.name:
            continue
        notes = sorted(inst.notes, key=lambda n: n.start)
        # One string sounds one note at a time.
        for prev, nxt in zip(notes, notes[1:], strict=False):
            assert nxt.start >= prev.end - 1e-6, f"{inst.name}: overlapping notes"
        # Every bend event falls within one of this channel's notes.
        for pb in inst.pitch_bends:
            assert any(n.start - 1e-6 <= pb.time <= n.end + 1e-6 for n in notes), \
                f"{inst.name}: bend at {pb.time} outside every note"
        # The wheel is back at 0 when the last note on the channel ends.
        last_end = notes[-1].end
        value = 0
        for pb in inst.pitch_bends:
            if pb.time <= last_end + 1e-9:
                value = pb.pitch
        assert value == 0, f"{inst.name}: bend bleeds past the final note"

    string_6 = next(i for i in pm.instruments if i.name.endswith('String 6'))
    assert len(string_6.notes) == 3


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
