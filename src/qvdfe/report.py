"""Small helpers for examples and checks: provenance stamp and comparison with expected results."""
import json
import subprocess

import numpy as np

from .paths import repo_root


def provenance():
    """Paper version and repository commit (or 'unknown' outside a git checkout)."""
    from . import PAPER, __version__
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo_root(), capture_output=True,
                                text=True, timeout=10).stdout.strip() or 'unknown'
    except Exception:  # noqa: BLE001
        commit = 'unknown'
    return {'paper': PAPER, 'qvdfe_version': __version__, 'commit': commit}


def close(a, b, rtol=1e-6, atol=1e-9):
    """Combined absolute and relative tolerance, so values near zero are not judged by relative error alone."""
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def compare(results, expected, rtol=1e-6, atol=1e-9, prefix=''):
    """Compare every numeric leaf of ``results`` with ``expected``. Returns a list of (key, got, want, ok)."""
    rows = []
    for k, want in expected.items():
        if k in ('provenance',):
            continue
        got = results.get(k) if isinstance(results, dict) else None
        if isinstance(got, np.generic):
            got = got.item()
        key = f'{prefix}{k}'
        if isinstance(want, dict):
            rows += compare(got or {}, want, rtol, atol, key + '.')
        elif isinstance(want, bool) or isinstance(want, str):
            rows.append((key, got, want, got == want))
        elif isinstance(want, (int, float)):
            rows.append((key, got, want, got is not None and close(got, want, rtol, atol)))
    return rows


def _plain(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def write_and_check(folder, results, rtol=1e-6, atol=1e-9):
    """Write results.json beside the example and compare with expected/results.json if it exists.

    Returns True when every compared value agrees (or when there is no expected file yet)."""
    results = {**results, 'provenance': provenance()}
    (folder / 'results.json').write_text(json.dumps(results, indent=2, default=_plain), encoding='utf-8')
    exp = folder / 'expected' / 'results.json'
    if not exp.exists():
        print('no expected/results.json yet; wrote results.json')
        return True
    rows = compare(results, json.loads(exp.read_text(encoding='utf-8')), rtol, atol)
    bad = [r for r in rows if not r[3]]
    print(f'compared {len(rows)} values with expected/results.json: {len(rows) - len(bad)} agree')
    for k, got, want, _ in bad:
        print(f'  MISMATCH {k}: got {got}, expected {want}')
    return not bad
