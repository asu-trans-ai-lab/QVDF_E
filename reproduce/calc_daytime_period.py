# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""One daytime accounting window for the planning-period totals.

AM + MD + PM summed per sensor-day; metrics recomputed on the supported windows from the summed totals, never by
combining period metrics. The script first reproduces the frozen coverage counts and pooled period metrics.
Outputs go to reproduce/output/calculations.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = _ROOT
PER = _FROZEN / 'calibration/average_weekday/period_results_v2.csv'
MET = _FROZEN / 'calibration/average_weekday/period_emission_metrics.csv'
EP = _FROZEN / 'calibration/average_weekday/episode_results_v2.csv'
OUT = _OUT / 'calculations'
OUT.mkdir(exist_ok=True)
POLS = ['CO2', 'NOx', 'CO', 'HC']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


P = pd.read_csv(PER, low_memory=False)
for c in ('complete_observations', 'all_episodes_valid', 'period_model_pass', 'holdout', 'is_average_day'):
    P[c] = P[c].astype(str).str.lower().eq('true')
P['sample'] = np.where(P.dataset.str.contains('PeMS'), 'PeMS', 'AZ')

# Gate 1: reproduce the frozen coverage counts (all periods, supported, congested volume share).
coverage = {}
for (s, per), g in P.groupby(['sample', 'period']):
    coverage[f'{s}-{per}'] = {'all': int(len(g)), 'supported': int(g.period_model_pass.sum()),
                              'congested_volume_pct': float(100 * g.D_veh.sum() / g.V_veh.sum())}
expected = {'AZ-NT': (50, 49, 1.5), 'AZ-AM': (50, 50, 0.0), 'AZ-MD': (50, 26, 25.0), 'AZ-PM': (50, 26, 64.1),
            'PeMS-NT': (347, 314, 1.0), 'PeMS-AM': (347, 277, 29.0), 'PeMS-MD': (347, 246, 15.1), 'PeMS-PM': (347, 258, 29.3)}
for k, (a, s, c) in expected.items():
    got = coverage[k]
    assert got['all'] == a and got['supported'] == s and abs(got['congested_volume_pct'] - c) < 0.051, (k, got, (a, s, c))


def metrics(frame, method, pol):
    o = frame[f'O_H_{pol}'].to_numpy(float)
    e = frame[f'{method}_H_{pol}'].to_numpy(float)
    r2 = 1 - ((e - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    return {'N': int(len(o)), 'R2': float(r2), 'bias_pct': float(100 * (e.sum() / o.sum() - 1)),
            'WAPE_pct': float(100 * np.abs(e - o).sum() / o.sum()), 'RMSE': float(np.sqrt(((e - o) ** 2).mean())), 'MAE': float(np.abs(e - o).mean())}


# Gate 2: reproduce the pooled period metrics file on the supported periods.
M = pd.read_csv(MET)
sup = P[P.period_model_pass]
gate2 = []
for r in M[(M['sample'] == 'All average-weekday profiles') & (M.period == 'All')].itertuples():
    got = metrics(sup, r.method, r.pollutant)
    ok = got['N'] == r.N and abs(got['R2'] - r.R2) < 1e-9 and abs(got['bias_pct'] - r.bias_pct) < 1e-9 and abs(got['WAPE_pct'] - r.WAPE_pct) < 1e-9
    gate2.append({'pollutant': r.pollutant, 'method': r.method, 'ok': bool(ok), 'N': got['N'], 'file_N': int(r.N)})
assert all(g['ok'] for g in gate2), gate2

# DAY window.
day_parts = P[P.period.isin(['AM', 'MD', 'PM'])]
agg = {c: 'sum' for c in ['H_h', 'V_veh', 'D_veh'] + [f'{m}_H_{p}' for m in ('O', 'S', 'M') for p in POLS]}
agg.update({'period_model_pass': 'all', 'complete_observations': 'all', 'all_episodes_valid': 'all', 'sample': 'first', 'corridor': 'first', 'holdout': 'first'})
DAY = day_parts.groupby(['sensor_uid', 'date']).agg(agg).reset_index()
DAY['n_parts'] = day_parts.groupby(['sensor_uid', 'date']).size().to_numpy()
assert (DAY.n_parts == 3).all() and np.allclose(DAY.H_h, 14.0)
DAY.to_csv(OUT / 'daytime_period_totals.csv', index=False)

cov_day = {}
for s, g in DAY.groupby('sample'):
    cov_day[s] = {'all': int(len(g)), 'supported': int(g.period_model_pass.sum()),
                  'congested_volume_pct': float(100 * g.D_veh.sum() / g.V_veh.sum())}
sup_day = DAY[DAY.period_model_pass]
met_day = {f'{m}-{p}': metrics(sup_day, m, p) for m in ('M', 'S') for p in POLS}
met_day_by_sample = {s: {f'{m}-{p}': metrics(g, m, p) for m in ('M', 'S') for p in POLS} for s, g in sup_day.groupby('sample')}

# Episodes crossing the internal boundaries, and the external daytime boundaries.
E = pd.read_csv(EP, low_memory=False)
t0 = pd.to_datetime(E.t0)
t3 = pd.to_datetime(E.t3)
h0 = t0.dt.hour + t0.dt.minute / 60
h3 = t3.dt.hour + t3.dt.minute / 60 + (t3.dt.normalize() - t0.dt.normalize()).dt.days * 24
internal = ((h0 < 10) & (h3 > 10)) | ((h0 < 16) & (h3 > 16))
external = ((h0 < 6) & (h3 > 6)) | ((h0 < 20) & (h3 > 20))
cross = {'episodes': int(len(E)), 'cross_internal_boundary': int(internal.sum()), 'cross_daytime_window_boundary': int(external.sum()),
         'cross_internal_eligible': int((internal & E.calibration_eligible.astype(str).str.lower().eq('true')).sum())}

summary = {'coverage_gate': coverage, 'metrics_gate': gate2, 'day_coverage': cov_day, 'day_metrics_pooled': met_day,
           'day_metrics_by_sample': met_day_by_sample, 'boundary_crossing': cross,
           'inputs': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (PER, MET, EP)}}
(OUT / 'daytime_period_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
