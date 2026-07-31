import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CLI_SCRIPT = REPO_ROOT / "src" / "audio2midi.py"


@pytest.fixture(scope="session")
def python_exe():
    """The interpreter running the suite, so tests are not tied to one machine."""
    return sys.executable


@pytest.fixture(scope="session")
def cli_script():
    return str(CLI_SCRIPT)


@pytest.fixture(scope="session")
def fixture_wav():
    """Resolves a test audio fixture by name, independent of the working directory."""
    def _resolve(name):
        path = TESTS_DIR / name
        if not path.exists():
            pytest.skip(f"Audio fixture {name} is not present in {TESTS_DIR}.")
        return str(path)
    return _resolve
