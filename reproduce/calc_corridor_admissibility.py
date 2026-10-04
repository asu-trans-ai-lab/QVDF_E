# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Corridor-scale geometric admissibility on Arizona I-10 daily episodes.

Single-link versus corridor (4.591 mi) finite-link inequality on daily episodes matched to a corridor traversal, at the
same bottleneck queue speed. The script first reproduces the frozen corridor summary counts and median ratios from the
same derived records. Outputs go to reproduce/output/calculations.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = _ROOT
DVC = _FROZEN / 'transitions/corridor/az_detector_vs_corridor.csv'
CEP = _FROZEN / 'transitions/corridor/az_corridor_episodes.csv'
SUM = _FROZEN / 'transitions/corridor/summary.json'
DEP = _FROZEN / 'calibration/daily_validation/episode_results_v2.csv'
OUT = _OUT / 'calculations'
OUT.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


summ = json.loads(SUM.read_text(encoding='utf-8'))
L_corr = float(summ['corridor_span_mi'])
Tff_h = float(summ['corridor_Tff_min']) / 60.0

dvc = pd.read_csv(DVC)
dvc['sensor'] = dvc.sensor.astype(str)
dvc['date'] = dvc.date.astype(str)
dep = pd.read_csv(DEP, low_memory=False)
dep['sensor'] = dep.sensor.astype(str)
dep['date'] = dep.date.astype(str)
for c in ('physical_observed_pass', 'calibration_eligible', 'holdout'):
    dep[c] = dep[c].astype(str).str.lower().eq('true')

# Join the corridor records to the daily episode results on detector, date, period and duration.
dvc['P_key'] = dvc.P_h.round(6)
dep['P_key'] = dep.P_h.round(6)
dvc['w_key'] = dvc.wt2_local_min.round(3)
dep['w_key'] = (dep.wt2_h * 60).round(3)
keys = ['sensor', 'date', 'period', 'P_key', 'w_key']
cols = keys + ['episode_id', 'vq_mph', 'vf_mph', 'L_mi', 'wt2_h', 'Tf_h', 'physical_observed_pass', 'calibration_eligible', 'D_veh', 'x_h']
depu = dep[cols].drop_duplicates(subset=keys, keep=False)
dup_dropped = int(len(dep) - len(dep[cols].drop_duplicates(subset=keys)))
m = dvc.merge(depu, on=keys, how='left', suffixes=('', '_ep'))
assert len(m) == len(dvc), (len(m), len(dvc))
unmatched = int(m.episode_id.isna().sum())
print('join: corridor rows', len(dvc), 'unmatched', unmatched, 'ambiguous daily keys dropped', dup_dropped)

# Reproduction gate: the transitions summary counts and median corridor/local ratios per detector.
gate = {}
for det in ('78', '84', '137', '139'):
    sub = m[(m.sensor == det) & np.isfinite(m.corridor_peak_delay_min) & (m.wt2_local_min > 0)]
    candidates = {'all_finite': sub, 'eligible': sub[sub.eligible.astype(str).str.lower().eq('true')],
                  'eligible_matched': sub[sub.eligible.astype(str).str.lower().eq('true') & sub.episode_id.notna()]}
    target_n, target_ratio = summ[f'det{det}_n'], summ[f'det{det}_median_ratio_corr_over_local']
    found = None
    for name, c in candidates.items():
        ratio = float(np.median(c.corridor_peak_delay_min / c.wt2_local_min)) if len(c) else np.nan
        corr = float(np.corrcoef(np.log(c.corridor_peak_delay_min), np.log(c.wt2_local_min))[0, 1]) if len(c) > 2 else np.nan
        if len(c) == target_n and abs(ratio - target_ratio) < 1e-6:
            found = (name, len(c), ratio, corr)
            break
    assert found, (det, target_n, target_ratio, {k: (len(v), float(np.median(v.corridor_peak_delay_min / v.wt2_local_min)) if len(v) else None) for k, v in candidates.items()})
    gate[det] = {'subset': found[0], 'n': found[1], 'median_ratio': found[2], 'corr_log': found[3],
                 'summary_n': target_n, 'summary_ratio': target_ratio, 'summary_corr_log': summ[f'det{det}_corr_log']}
    assert abs(found[3] - summ[f'det{det}_corr_log']) < 1e-6, gate[det]
subset_name = gate['78']['subset']

# Admissibility on the matched episodes with a valid observed-state queue speed.
sel = m[np.isfinite(m.corridor_peak_delay_min) & (m.wt2_local_min > 0) & m.episode_id.notna()].copy()
if subset_name != 'all_finite':
    sel = sel[sel.eligible.astype(str).str.lower().eq('true')]
sel['vq_valid'] = np.isfinite(sel.vq_mph) & (sel.vq_mph > 0) & (sel.vq_mph < sel.vf_mph)
sel['link_allow_min'] = 60 * sel.L_mi * (1 / sel.vq_mph - 1 / sel.vf_mph)
sel['corr_allow_min'] = 60 * (L_corr / sel.vq_mph - Tff_h)
sel['link_admitted'] = sel.vq_valid & (sel.wt2_local_min <= sel.link_allow_min)
sel['corr_admitted'] = sel.vq_valid & (sel.corridor_peak_delay_min <= sel.corr_allow_min)
# Cross-check: the single-link test with the file's own observed delay must agree with the frozen flag.
agree = (sel.link_admitted == sel.physical_observed_pass)[sel.vq_valid]
cross = {'single_link_agrees_with_frozen_flag': int(agree.sum()), 'single_link_disagrees': int((~agree).sum())}
sel.to_csv(OUT / 'corridor_admissibility_matched.csv', index=False)

per_det = {}
for det in ('78', '84', '137', '139'):
    s = sel[sel.sensor == det]
    v = s[s.vq_valid]
    per_det[det] = {'matched': int(len(s)), 'valid_queue_state': int(len(v)),
                    'link_admitted': int(v.link_admitted.sum()), 'corridor_admitted': int(v.corr_admitted.sum()),
                    'both': int((v.link_admitted & v.corr_admitted).sum()), 'corridor_only': int((~v.link_admitted & v.corr_admitted).sum()),
                    'link_only': int((v.link_admitted & ~v.corr_admitted).sum()),
                    'median_link_allow_min': float(v.link_allow_min.median()), 'median_corr_allow_min': float(v.corr_allow_min.median()),
                    'median_local_peak_min': float(v.wt2_local_min.median()), 'median_corridor_peak_min': float(v.corridor_peak_delay_min.median()),
                    'median_P_h': float(v.P_h.median())}

# Corridor-defined episodes matched to a detector-78 state on the same date with overlapping time.
cep = pd.read_csv(CEP)
cep['date'] = cep.date.astype(str)
d78 = dep[(dep.sensor == '78')].copy()
d78['h0'] = pd.to_datetime(d78.t0).dt.hour + pd.to_datetime(d78.t0).dt.minute / 60
d78['h3'] = pd.to_datetime(d78.t3).dt.hour + pd.to_datetime(d78.t3).dt.minute / 60
rows = []
for r in cep.itertuples():
    cand = d78[(d78.date == r.date) & (d78.h0 < r.t3_h) & (d78.h3 > r.t0_h)]
    if len(cand) == 0:
        rows.append({'date': r.date, 'P_h': r.P_h, 'peak_delay_min': r.peak_delay_h * 60, 'matched': False}); continue
    cand = cand.assign(overlap=np.minimum(cand.h3, r.t3_h) - np.maximum(cand.h0, r.t0_h)).sort_values('overlap', ascending=False)
    e = cand.iloc[0]
    valid = np.isfinite(e.vq_mph) and 0 < e.vq_mph < e.vf_mph
    allow = 60 * (L_corr / e.vq_mph - Tff_h) if valid else np.nan
    rows.append({'date': r.date, 'P_h': r.P_h, 'peak_delay_min': r.peak_delay_h * 60, 'matched': True, 'episode_id': e.episode_id,
                 'vq_mph': e.vq_mph, 'vq_valid': bool(valid), 'corr_allow_min': allow, 'corr_admitted': bool(valid and r.peak_delay_h * 60 <= allow)})
C = pd.DataFrame(rows)
C.to_csv(OUT / 'corridor_defined_episodes_admissibility.csv', index=False)
corr_def = {'corridor_episodes': int(len(C)), 'matched_to_detector_78': int(C.matched.sum()),
            'valid_queue_state': int(C.get('vq_valid', pd.Series(dtype=bool)).fillna(False).sum()),
            'corridor_admitted': int(C.get('corr_admitted', pd.Series(dtype=bool)).fillna(False).sum())}

summary = {'corridor_span_mi': L_corr, 'corridor_Tff_min': Tff_h * 60, 'gate': gate, 'subset_used': subset_name, 'unmatched_corridor_rows': unmatched,
           'cross_check': cross, 'per_detector': per_det, 'corridor_defined_episodes': corr_def,
           'inputs': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (DVC, CEP, SUM, DEP)}}
(OUT / 'corridor_admissibility_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
