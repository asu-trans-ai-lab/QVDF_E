"""Paths for the reproduction entries: repository root, frozen inputs and the output folder."""
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent
ROOT = LIB.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qvdfe.paths import frozen_dir  # noqa: E402

FROZEN = frozen_dir()
OUT = ROOT / 'reproduce' / 'output'
for _d in ('figs', 'qa', 'calculations', 'audit'):
    (OUT / _d).mkdir(parents=True, exist_ok=True)
