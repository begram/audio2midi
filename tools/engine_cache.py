"""On-disk cache of raw engine output, shared by the analysis tools.

Inference is the only expensive step, so both tools cache what `BasicPitchEngine`
returns. The key must cover everything that can change that output -- not just the
preset parameters but the code behind them: a preset of `{}` means "whatever the
current defaults are", so editing a default in `processor.py` would otherwise keep
serving results computed under the old one.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from basic_pitch_engine import BasicPitchEngine
from processor import preprocess_pipeline

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
# Every module on the path from wav to raw note list.
_ENGINE_SOURCES = ("processor.py", "basic_pitch_engine.py")
PREPROCESS_KEYS = {"noise_threshold", "use_hpss"}


def _code_fingerprint():
    digest = hashlib.sha1()
    for name in _ENGINE_SOURCES:
        digest.update((SRC_DIR / name).read_bytes())
    digest.update(version("basic-pitch").encode())
    return digest.hexdigest()


def transcribe_cached(wav_path, engine_params, cache_dir, use_cache=True):
    """Runs preprocessing + inference, caching the raw note list on disk."""
    pre = {k: v for k, v in engine_params.items() if k in PREPROCESS_KEYS}
    infer = {k: v for k, v in engine_params.items() if k not in PREPROCESS_KEYS}

    payload = json.dumps(
        {"wav": wav_path.name, "pre": pre, "infer": infer, "code": _code_fingerprint()},
        sort_keys=True,
    )
    key = hashlib.sha1(payload.encode()).hexdigest()[:12]
    cache_path = Path(cache_dir) / f"{wav_path.stem}__{key}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())

    audio, sr = preprocess_pipeline(str(wav_path), **pre)
    notes = BasicPitchEngine().transcribe(audio, sr, **infer)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(notes))
    return notes
