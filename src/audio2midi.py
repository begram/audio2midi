import click
import sys
from tqdm import tqdm
import os

from processor import preprocess_pipeline
from basic_pitch_engine import BasicPitchEngine
from post_process import clean_notes, quantize_notes, merge_notes
from midi_gen import generate_midi

@click.command()
@click.argument('input_wav', type=click.Path(exists=True))
@click.argument('output_midi', type=click.Path())
@click.option('--bpm', type=int, required=True, help='Beats Per Minute for the output MIDI.')
@click.option('--quantize', type=click.Choice(['1/4', '1/8', '1/16', '1/32']), help='Optional quantization grid.')
@click.option('--min-duration', type=float, default=0.030, help='Minimum note duration in seconds.')
@click.option('--velocity-threshold', type=int, default=10, help='Minimum velocity (0-127) to include a note.')
@click.option('--noise-threshold', type=float, default=0.005, help='Amplitude threshold for noise gate (0.0 to 1.0).')
@click.option('--no-merge', is_flag=True, help='Disable merging of overlapping identical pitches.')
@click.option('--instrument', default='Acoustic Guitar', help='Instrument name for MIDI track.')
@click.option('--tab', is_flag=True, help='Assign notes to strings (tablature mode).')
def main(input_wav, output_midi, bpm, quantize, min_duration, velocity_threshold, noise_threshold, no_merge, instrument, tab):
    """Polyphonic Guitar-to-MIDI Converter"""
    click.echo(f"[*] Processing: {input_wav}")
    
    click.echo(f"[*] Pre-processing audio...")
    audio, sr = preprocess_pipeline(input_wav, noise_threshold=noise_threshold)
    
    click.echo(f"[*] Transcribing (this may take a moment)...")
    engine = BasicPitchEngine()
    notes = engine.transcribe(audio, sr)
    click.echo(f"[*] Detected {len(notes)} notes.")
    
    click.echo(f"[*] Cleaning and merging notes...")
    notes = clean_notes(notes, min_duration=min_duration, velocity_threshold=velocity_threshold)
    if not no_merge:
        notes = merge_notes(notes)
    click.echo(f"[*] Resulting {len(notes)} unique note events.")
    
    if quantize:
        click.echo(f"[*] Quantizing to {quantize} grid at {bpm} BPM...")
        notes = quantize_notes(notes, bpm, grid_resolution=quantize)
        seconds_per_quarter = 60.0 / bpm
        ticks_per_quarter = 480
        for n in notes:
            n['start'] = (n['start_ticks'] / ticks_per_quarter) * seconds_per_quarter
            n['end'] = (n['end_ticks'] / ticks_per_quarter) * seconds_per_quarter

    if tab:
        click.echo("[*] Estimating string assignments...")
        from tab_engine import TabMapper
        mapper = TabMapper()
        notes = mapper.assign_strings(notes)

    click.echo(f"[*] Generating MIDI at {bpm} BPM...")
    generate_midi(notes, bpm, output_midi, instrument_name=instrument)
    
    click.echo(f"[+] Success! MIDI saved to: {output_midi}")

if __name__ == '__main__':
    main()