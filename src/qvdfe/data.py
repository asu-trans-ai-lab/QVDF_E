"""Loaders for the frozen data used by the tutorials, examples and reproduction scripts."""
import json

import numpy as np
import pandas as pd

from .paths import frozen_dir

POLLUTANTS = ['CO2', 'NOx', 'CO', 'HC']


def load_rates(path=None):
    """Cubic zero-acceleration rates fitted to the archived MOVES-derived passenger-car table.

    Returns {pollutant: {'coef': array(c0..c3), 'vmin': mph, 'vmax': mph}}; r in g/(veh h), v in mph.
    The original MOVES run (version, year, settings) was not recorded; results start from this derived table.
    """
    p = path or frozen_dir() / 'calibration' / 'cubic_rate_coefficients.csv'
    df = pd.read_csv(p)
    return {r.pollutant: {'coef': np.array([r.c0, r.c1, r.c2, r.c3], float), 'vmin': float(r.speed_min_mph),
                          'vmax': float(r.speed_max_mph)} for r in df.itertuples()}


def load_reference_state():
    """Detector-78 reference state of Appendix F (saved full-precision values)."""
    return json.loads((frozen_dir() / 'calibration' / 'audit' / 'table3_reference_state.json').read_text(encoding='utf-8'))


def load_cards():
    """Average-weekday QVDF parameter cards with the FD of a representative detector per corridor
    (detector 78 for Arizona). Columns as in parameter_card.csv plus C, vf, L, kj, omega, sensor."""
    A = frozen_dir() / 'calibration' / 'average_weekday'
    pc = pd.read_csv(A / 'parameter_card.csv')
    sp = pd.read_csv(A / 'sensor_parameters.csv')
    out = []
    for row in pc.to_dict('records'):
        s = sp[sp.corridor == row['corridor']]
        if row['corridor'] == 'AZ_I10_W':
            s = s[s.sensor.astype(str) == '78']
        if len(s):
            s0 = s.iloc[0]
            row.update(C=s0.C_vphpl, vf=s0.vf_mph, L=s0.L_mi, kj=s0.kj, omega=s0.omega_mph, sensor=str(s0.sensor))
        out.append(row)
    return pd.DataFrame(out)


def load_episodes(kind='average_weekday'):
    """Episode results (kind 'average_weekday' or 'daily_validation') with boolean flags parsed."""
    df = pd.read_csv(frozen_dir() / 'calibration' / kind / 'episode_results_v2.csv', low_memory=False)
    for c in df.columns:
        if df[c].dtype == object and set(df[c].dropna().astype(str).str.lower().unique()) <= {'true', 'false'}:
            df[c] = df[c].astype(str).str.lower().eq('true')
    return df
