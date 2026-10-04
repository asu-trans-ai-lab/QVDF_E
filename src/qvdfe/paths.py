"""Locate the repository and its data independently of the current working directory."""
from pathlib import Path
import os


def repo_root() -> Path:
    """Repository root: the folder holding ``src/qvdfe`` (located from this file, not from the working directory)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Data folder. The environment variable ``QVDFE_DATA`` overrides the default ``<repo>/data``."""
    env = os.environ.get('QVDFE_DATA')
    return Path(env).resolve() if env else repo_root() / 'data'


def frozen_dir() -> Path:
    """Folder with the frozen outputs (calibration, transitions, applicability).

    In the public release they sit in ``data/frozen``; in the working package directly in ``data``.
    """
    d = data_dir()
    f = d / 'frozen' if (d / 'frozen' / 'calibration').exists() else d
    if not (f / 'calibration').exists():
        raise FileNotFoundError(
            f'QVDFE frozen data not found under {d}. Use an editable install from a clone (pip install -e .) '
            'or set the environment variable QVDFE_DATA to the repository data folder.')
    return f
