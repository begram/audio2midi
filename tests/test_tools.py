"""Tests for the analysis tooling in tools/.

The GuitarSet numbers behind the project's tuning decisions (merging off by default, the
max_frets limit, the per-note position cost) all come from `evaluate_guitarset`, so its
bookkeeping needs the same scrutiny as the pipeline it measures. No inference runs here.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import engine_cache
import evaluate_guitarset as eg

# --- presets mirror the CLI ----------------------------------------------------

def test_engine_presets_use_the_cli_and_engine_constants():
    """A preset that hardcodes its own numbers silently measures something users don't get."""
    import analyze_transcription as at
    import audio2midi
    import basic_pitch_engine as bpe

    freq = {"min_freq": audio2midi.GUITAR_MIN_FREQ, "max_freq": audio2midi.GUITAR_MAX_FREQ}
    clean = {
        "onset_threshold": bpe.CLEAN_RECORDING_ONSET_THRESHOLD,
        "frame_threshold": bpe.CLEAN_RECORDING_FRAME_THRESHOLD,
        "min_note_len": bpe.CLEAN_RECORDING_MIN_NOTE_LENGTH_MS,
    }
    for presets in (eg.ENGINE_PRESETS, at.ENGINE_PRESETS):
        assert presets["baseline"] == {}  # i.e. the engine's DEFAULT_* values
        assert presets["freq_bounds"] == freq
        assert presets["tuned_clean"] == clean


# --- GuitarSet string numbering ------------------------------------------------

def test_guitarset_string_numbering_runs_opposite_to_tab_engine():
    """GuitarSet data_source 0 is low E; tab_engine calls low E string 6."""
    assert eg.guitarset_source_to_string(0) == 6
    assert eg.guitarset_source_to_string(5) == 1
    assert eg.guitarset_source_to_string("2") == 4  # JAMS stores it as a string


def test_load_reference_reads_strings_and_skips_empty_notes(tmp_path):
    def note_track(source, observations):
        return {
            "namespace": "note_midi",
            "annotation_metadata": {"data_source": source},
            "data": [{"time": t, "duration": d, "value": v} for t, d, v in observations],
        }

    doc = {"annotations": [
        note_track("0", [(1.0, 0.5, 40.1), (2.0, 0.0, 41.0)]),   # zero-length dropped
        note_track("5", [(0.5, 0.25, 64.2)]),
        {"namespace": "pitch_contour", "annotation_metadata": {"data_source": "0"},
         "data": [{"time": 0.0, "duration": 1.0, "value": 99}]},  # not a note track
    ]}
    path = tmp_path / "track.jams"
    path.write_text(json.dumps(doc))

    ref = eg.load_reference(path)

    assert [(n["start"], n["string"]) for n in ref] == [(0.5, 1), (1.0, 6)]
    assert ref[1]["end"] == pytest.approx(1.5)
    assert ref[1]["pitch"] == pytest.approx(40.1)  # continuous f0, not rounded


# --- string accuracy -------------------------------------------------------------

def _note(pitch, start, string):
    return {"pitch": pitch, "start": start, "end": start + 0.4, "string": string}


def test_string_accuracy_counts_only_assigned_matches():
    """An unassigned note is excluded rather than counted wrong; it is reported separately."""
    ref = [_note(52, 0.0, 5), _note(55, 1.0, 4), _note(59, 2.0, 3)]
    est = [_note(52, 0.0, 5),   # right string
           _note(55, 1.0, 3),   # wrong string
           _note(59, 2.0, 0)]   # unassigned

    accuracy, matched, unassigned_rate = eg.string_accuracy(ref, est)

    assert matched == 3
    assert accuracy == pytest.approx(0.5)
    assert unassigned_rate == pytest.approx(1 / 3)


def test_string_accuracy_ignores_unmatched_notes():
    """A spurious note has no reference counterpart whose string it could be compared to."""
    ref = [_note(52, 0.0, 5)]
    est = [_note(52, 0.0, 5), _note(70, 3.0, 1)]

    accuracy, matched, _ = eg.string_accuracy(ref, est)

    assert matched == 1
    assert accuracy == pytest.approx(1.0)


# --- track selection ---------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Two excerpts per player per style, for all six GuitarSet players."""
    for player in range(6):
        for excerpt in ("BN1-129-Eb", "Rock1-130-A"):
            for style in ("comp", "solo"):
                (tmp_path / f"0{player}_{excerpt}_{style}_mic.wav").touch()
    monkeypatch.setattr(eg, "AUDIO_DIR", tmp_path)
    return tmp_path


def test_select_tracks_balances_comping_and_solo(corpus):
    """Sorted names alternate comp/solo, so a naive stride would pick one style only."""
    chosen = eg.select_tracks(4)
    styles = [("comp" if "_comp" in w.stem else "solo") for w in chosen]

    assert sorted(styles) == ["comp", "comp", "solo", "solo"]
    assert len(set(chosen)) == 4


def test_default_split_never_draws_test_players(corpus):
    """Routine tuning runs must not see the held-out players, even when asking for all tracks."""
    chosen = eg.select_tracks(None)

    assert {eg.player_of(w) for w in chosen} == {"00", "01", "02", "03"}


def test_splits_are_disjoint_and_cover_every_player(corpus):
    dev = set(eg.select_tracks(None, split="dev"))
    test = set(eg.select_tracks(None, split="test"))
    every = set(eg.select_tracks(None, split="all"))

    assert not dev & test
    assert dev | test == every
    assert {eg.player_of(w) for w in test} == {"04", "05"}


# --- engine output cache -----------------------------------------------------------

@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """engine_cache with stand-in engine sources and the expensive calls stubbed out."""
    src = tmp_path / "src"
    src.mkdir()
    for name in ("processor.py", "basic_pitch_engine.py"):
        (src / name).write_text(f"# {name} v1\n")
    monkeypatch.setattr(engine_cache, "SRC_DIR", src)

    engine = MagicMock()
    engine.return_value.transcribe.return_value = [
        {"pitch": 60, "start": 0.0, "end": 1.0, "velocity": 90}
    ]
    monkeypatch.setattr(engine_cache, "BasicPitchEngine", engine)
    monkeypatch.setattr(engine_cache, "preprocess_pipeline", MagicMock(return_value=([], 22050)))

    def transcribe(params=None):
        return engine_cache.transcribe_cached(
            Path("clip.wav"), params or {}, tmp_path / "cache"
        )

    return src, engine, transcribe


def test_engine_cache_reuses_unchanged_results(cache_env):
    _src, engine, transcribe = cache_env

    first = transcribe()
    second = transcribe()

    assert first == second
    assert engine.return_value.transcribe.call_count == 1


def test_engine_cache_invalidates_when_engine_source_changes(cache_env):
    """A preset of {} means "current defaults", so editing a default must miss the cache."""
    src, engine, transcribe = cache_env

    transcribe()
    (src / "processor.py").write_text("# processor.py v2: new noise gate default\n")
    transcribe()

    assert engine.return_value.transcribe.call_count == 2


def test_engine_cache_separates_presets(cache_env):
    _src, engine, transcribe = cache_env

    transcribe({})
    transcribe({"use_hpss": True})

    assert engine.return_value.transcribe.call_count == 2
