from setuptools import setup

setup(
    name="audio2midi",
    version="0.1.0",
    # src/ holds flat top-level modules, not a package, so find_packages(where="src")
    # returns an empty list and installs no code at all.
    package_dir={"": "src"},
    py_modules=[
        "audio2midi",
        "processor",
        "basic_pitch_engine",
        "engine_base",
        "post_process",
        "midi_gen",
        "tab_engine",
    ],
    python_requires=">=3.10,<3.11",
    install_requires=[
        "basic-pitch>=0.2.0",
        "tensorflow>=2.7.0",
        "librosa>=0.9.0",
        "pydub>=0.25.1",
        "scipy>=1.7.0",
        # basic_pitch_engine imports soundfile directly; librosa happens to pull it in
        # too, but that is not a contract.
        "soundfile>=0.10.0",
        "numpy<2.0.0",
        "pretty_midi>=0.2.9",
        "mido>=1.2.10",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock>=3.7.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.0.260",
            # Used by the analysis tooling in tools/.
            "mir_eval>=0.7",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audio2midi=audio2midi:main",
        ],
    },
)
