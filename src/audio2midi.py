import click

from basic_pitch_engine import BasicPitchEngine
from midi_gen import generate_midi
from post_process import (
    apply_logarithmic_velocity,
    clean_notes,
    merge_notes,
    quantize_notes,
    ticks_to_seconds,
)
from processor import preprocess_pipeline
from tab_engine import TabMapper

GUITAR_MIN_FREQ = 80.0
GUITAR_MAX_FREQ = 1400.0

@click.command()
@click.argument('input_wav', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_midi', type=click.Path(dir_okay=False))
@click.option('--bpm', type=click.IntRange(1, 400), required=True,
              help='Beats Per Minute for the output MIDI.')
@click.option('--quantize', type=click.Choice(['1/4', '1/8', '1/16', '1/32']),
              help='Optional quantization grid.')
@click.option('--quantize-strength', type=click.FloatRange(0.0, 1.0), default=1.0,
              help='Quantization snapping strength (0.0 to 1.0).')
@click.option('--min-duration', type=click.FloatRange(min=0.0), default=0.030,
              help='Minimum note duration in seconds (post-filter).')
@click.option('--min-note-length', type=click.FloatRange(min=0.0), default=127.70,
              help='Basic Pitch minimum note length in milliseconds (engine-level).')
@click.option('--velocity-threshold', type=click.IntRange(0, 127), default=10,
              help='Minimum velocity (0-127) to include a note.')
@click.option('--velocity-curve', type=click.FloatRange(min=0.0), default=0.0,
              help='Logarithmic velocity curve factor (e.g. 5.0). 0.0 disables.')
@click.option('--noise-threshold', type=click.FloatRange(0.0, 1.0), default=0.005,
              help='RMS threshold for the noise gate (0.0 to 1.0). 0.0 disables.')
@click.option('--hpss', is_flag=True,
              help='Enable Harmonic-Percussive Source Separation to filter pick clicks.')
@click.option('--freq-bounds', is_flag=True,
              help='Restrict frequency detection to acoustic guitar range (80-1400 Hz).')
@click.option('--min-freq', type=click.FloatRange(min=1.0), default=None,
              help='Explicit minimum frequency bound in Hz.')
@click.option('--max-freq', type=click.FloatRange(min=1.0), default=None,
              help='Explicit maximum frequency bound in Hz.')
@click.option('--onset-threshold', type=click.FloatRange(0.0, 1.0), default=0.5,
              help='Basic Pitch onset detection threshold.')
@click.option('--frame-threshold', type=click.FloatRange(0.0, 1.0), default=0.3,
              help='Basic Pitch frame confidence threshold.')
@click.option('--pitch-bend', is_flag=True,
              help='Extract and write pitch bend events (slides, vibrato).')
@click.option('--no-merge', is_flag=True,
              help='Disable merging of overlapping identical pitches.')
@click.option('--instrument', default='Acoustic Guitar',
              help='Instrument name for MIDI track.')
@click.option('--tab', is_flag=True,
              help='Assign notes to strings (tablature mode).')
def main(
    input_wav, output_midi, bpm, quantize, quantize_strength,
    min_duration, min_note_length, velocity_threshold, velocity_curve, noise_threshold,
    hpss, freq_bounds, min_freq, max_freq, onset_threshold, frame_threshold,
    pitch_bend, no_merge, instrument, tab
):
    """Polyphonic Guitar-to-MIDI Converter"""
    eff_min_freq = min_freq if min_freq is not None else (GUITAR_MIN_FREQ if freq_bounds else None)
    eff_max_freq = max_freq if max_freq is not None else (GUITAR_MAX_FREQ if freq_bounds else None)

    if eff_min_freq is not None and eff_max_freq is not None and eff_min_freq >= eff_max_freq:
        raise click.BadParameter(
            f"minimum frequency ({eff_min_freq} Hz) must be below maximum frequency ({eff_max_freq} Hz).",
            param_hint='--min-freq / --max-freq'
        )

    click.echo(f"[*] Processing: {input_wav}")

    click.echo("[*] Pre-processing audio...")
    try:
        audio, sr = preprocess_pipeline(
            input_wav,
            noise_threshold=noise_threshold,
            use_hpss=hpss
        )
    except Exception as exc:
        raise click.ClickException(f"Could not read '{input_wav}' as audio: {exc}") from exc

    click.echo("[*] Transcribing (this may take a moment)...")
    engine = BasicPitchEngine()

    notes = engine.transcribe(
        audio, sr,
        min_freq=eff_min_freq,
        max_freq=eff_max_freq,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        min_note_len=min_note_length,
        include_pitch_bends=pitch_bend
    )
    click.echo(f"[*] Detected {len(notes)} notes.")

    click.echo("[*] Cleaning and merging notes...")
    notes = clean_notes(notes, min_duration=min_duration, velocity_threshold=velocity_threshold)
    if velocity_curve > 0:
        click.echo(f"[*] Applying logarithmic velocity curve (k={velocity_curve})...")
        notes = apply_logarithmic_velocity(notes, curvature=velocity_curve)

    if not no_merge:
        notes = merge_notes(notes)
    click.echo(f"[*] Resulting {len(notes)} unique note events.")

    if quantize:
        click.echo(f"[*] Quantizing to {quantize} grid at {bpm} BPM (strength={quantize_strength})...")
        notes = quantize_notes(notes, bpm, grid_resolution=quantize, strength=quantize_strength)
        notes = ticks_to_seconds(notes, bpm)

    if tab:
        click.echo("[*] Estimating string assignments...")
        mapper = TabMapper()
        notes = mapper.assign_strings(notes)

    click.echo(f"[*] Generating MIDI at {bpm} BPM...")
    try:
        generate_midi(notes, bpm, output_midi, instrument_name=instrument)
    except OSError as exc:
        raise click.ClickException(f"Could not write '{output_midi}': {exc}") from exc

    click.echo(f"[+] Success! MIDI saved to: {output_midi}")

if __name__ == '__main__':
    main()
