from setuptools import setup, find_packages

setup(
    name="audio2midi",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "basic-pitch>=0.2.0",
        "tensorflow>=2.7.0",
        "librosa>=0.9.0",
        "pydub>=0.25.1",
        "scipy>=1.7.0",
        "numpy<2.0.0",
        "pretty_midi>=0.2.9",
        "mido>=1.2.10",
        "click>=8.0.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-mock", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "audio2midi=audio2midi:main",
        ],
    },
)
