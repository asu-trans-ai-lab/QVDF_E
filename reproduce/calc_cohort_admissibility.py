# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Cohort transition admissibility on the frozen average-weekday episodes.

For every admitted average-weekday episode the closed-queue delay profile is reconstructed, differentiated with respect
to actual time and checked for physical validity (nonnegative arrival rate, cohort count and mean delay reproduced).
The share of the cohort whose delay lies inside the transition-admissibility band of Eq. (26) is then computed, with
the shares below and above the band, for four pairs of peak deceleration/acceleration magnitudes. An empty band is
counted, not scored. Outputs go to reproduce/output/calculations.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = _ROOT
EP = _FROZEN / 'calibration/average_weekday/episode_results_v2.csv'
REF = _FROZEN / 'calibration/audit/table3_reference_state.json'
OUT = _OUT / 'calculations'
OUT.mkdir(exist_ok=True)
MPH_TO_MS = 0.44704
PAIRS = [(1.0, 0.67), (1.5, 1.0), (2.0, 1.5), (3.0, 2.0)]   # (a_down, a_up) in m/s^2; the paper's illustration is (1.5, 1.0)
BASE = (1.5, 1.0)
N_GRID = 20001


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tag_of(a_dn, a_up):
    return f'{a_dn:g}_{a_up:g}'


ref = json.loads(REF.read_text(encoding='utf-8'))
geom = ref.get('transition_geometry', {})

E = pd.read_csv(EP, low_memory=False)
flags = ['physical_predicted_pass', 'profile_closed_pass', 'cubic_domain_pass', 'matched_ladder_pass']
for f in flags:
    E[f] = E[f].astype(str).str.lower().eq('true')
fitted = E[E.calibration_eligible.astype(str).str.lower().eq('true') & np.isfinite(E.P_hat_h)]
adm = E[E.matched_ladder_pass].copy()
gate = {'fitted_episodes': int(len(fitted)), 'admitted_episodes': int(len(adm)),
        'admitted_subset_of_finite_link': bool((adm.physical_predicted_pass).all())}
assert gate['admitted_episodes'] == 137, gate
assert gate['admitted_subset_of_finite_link'], gate

u = np.linspace(0.0, 1.0, N_GRID)


def profile(theta, a_shape):
    """Normalized delay profile h(u) and its derivative, per the paper's reconstruction rule."""
    if theta <= 2.0 / 3.0:
        a = float(a_shape)
        base = 4 * u * (1 - u)
        h = base ** a
        with np.errstate(divide='ignore', invalid='ignore'):
            dh = np.where(base > 0, a * base ** (a - 1) * 4 * (1 - 2 * u), 0.0)
    else:
        b = theta / (1 - theta)
        s = np.abs(2 * u - 1)
        h = 1 - s ** b
        with np.errstate(divide='ignore', invalid='ignore'):
            dh = np.where(s > 0, -b * s ** (b - 1) * np.sign(2 * u - 1) * 2, 0.0)
    return h, dh


rows = []
for r in adm.itertuples():
    theta = float(r.theta)
    h, dh = profile(theta, r.profile_shape_a)
    P = float(r.P_hat_h)
    wpk = float(r.wt2_hat_h)
    mu = float(r.mu_hat_vphpl)
    D = float(r.D_veh)
    w = wpk * h                       # h
    wprime = (wpk / P) * dh           # dimensionless, derivative with respect to actual time
    lam = mu * (1 + wprime)           # veh/h/lane
    t = u * P
    count = np.trapezoid(lam, t)
    wbar_time = np.trapezoid(w, t) / P
    wbar_veh = np.trapezoid(lam * w, t) / count
    valid = {'lambda_min': float(lam.min()), 'count_rel_err': float(abs(count - D) / D),
             'wbar_rel_err_vs_file': float(abs(wbar_veh - r.wbar_hat_h) / r.wbar_hat_h),
             'theta_rel_err': float(abs(wbar_time / wpk - theta) / theta)}
    rec = {'episode_id': r.episode_id, 'corridor': r.corridor, 'sensor': r.sensor, 'date': r.date, 'period': r.period,
           'dataset': 'AZ' if r.corridor == 'AZ_I10_W' else 'PeMS',
           'D_veh': D, 'P_hat_h': P, 'x_h': float(r.x_h), 'vf_mph': float(r.vf_mph), 'vq_hat_mph': float(r.vq_hat_mph), 'L_mi': float(r.L_mi),
           'wt2_hat_min': wpk * 60, 'wbar_hat_min': float(r.wbar_hat_h) * 60, 'theta': theta,
           'profile_family': r.profile_family, **valid}
    vf, vq, L = float(r.vf_mph), float(r.vq_hat_mph), float(r.L_mi)
    dv_ms = (vf - vq) * MPH_TO_MS
    for a_dn, a_up in PAIRS:
        T_dn = 3 * dv_ms / (2 * a_dn)
        T_up = 3 * dv_ms / (2 * a_up)
        T_h = (T_dn + T_up) / 3600.0
        w_lo = (vf - vq) * T_h / (2 * vf)
        w_hi = L * (1 / vq - 1 / vf) - (vf - vq) * T_h / (2 * vq)
        empty = bool(w_hi < w_lo)
        inside = (w >= w_lo) & (w <= w_hi)
        below = (w < w_lo) & (not empty)
        above = (w > w_hi) & (not empty)
        tg = tag_of(a_dn, a_up)
        rec.update({f'T_down_s_{tg}': T_dn, f'T_up_s_{tg}': T_up, f'w_lo_min_{tg}': w_lo * 60, f'w_hi_min_{tg}': w_hi * 60,
                    f'share_inside_{tg}': float(np.trapezoid(lam * inside, t) / count),
                    f'share_below_{tg}': float(np.trapezoid(lam * below, t) / count),
                    f'share_above_{tg}': float(np.trapezoid(lam * above, t) / count),
                    f'band_empty_{tg}': empty, f'mean_delay_admitted_{tg}': bool(w_lo <= r.wbar_hat_h <= w_hi),
                    f'peak_delay_admitted_{tg}': bool(w_lo <= wpk <= w_hi)})
    rows.append(rec)
R = pd.DataFrame(rows)

# Validity gates: every reconstructed profile must be physically feasible before any share is reported.
validity = {'lambda_min_over_episodes': float(R.lambda_min.min()), 'max_count_rel_err': float(R.count_rel_err.max()),
            'max_wbar_rel_err_vs_file': float(R.wbar_rel_err_vs_file.max()), 'max_theta_rel_err': float(R.theta_rel_err.max()),
            'profile_families': sorted(R.profile_family.astype(str).unique().tolist())}
assert validity['lambda_min_over_episodes'] >= -1e-9, validity
assert validity['max_count_rel_err'] < 1e-6, validity
assert validity['max_theta_rel_err'] < 1e-4, validity
R.to_csv(OUT / 'cohort_admissibility_episodes.csv', index=False)


def q(s, p):
    return float(np.percentile(s, p)) if len(s) else float('nan')


summary = {'gate': gate, 'validity': validity, 'pairs': {}, 'grid_points': N_GRID,
           'reference_transition_geometry': geom,
           'inputs': {str(EP.relative_to(ROOT)).replace(chr(92), '/'): sha(EP), str(REF.relative_to(ROOT)).replace(chr(92), '/'): sha(REF)}}
for a_dn, a_up in PAIRS:
    tg = tag_of(a_dn, a_up)
    ne = ~R[f'band_empty_{tg}']
    si, sb, sa = R.loc[ne, f'share_inside_{tg}'], R.loc[ne, f'share_below_{tg}'], R.loc[ne, f'share_above_{tg}']
    summary['pairs'][tg] = {
        'a_down': a_dn, 'a_up': a_up,
        'n_band_nonempty': int(ne.sum()), 'n_band_empty': int((~ne).sum()),
        'n_band_empty_by_dataset': {k: int(v) for k, v in R.loc[~ne].groupby('dataset').size().items()},
        'n_band_nonempty_by_dataset': {k: int(v) for k, v in R.loc[ne].groupby('dataset').size().items()},
        'median_L_mi_band_empty': float(R.loc[~ne, 'L_mi'].median()) if (~ne).any() else None,
        'median_L_mi_band_nonempty': float(R.loc[ne, 'L_mi'].median()) if ne.any() else None,
        'inside_median_pct': 100 * q(si, 50), 'inside_q25_pct': 100 * q(si, 25), 'inside_q75_pct': 100 * q(si, 75),
        'below_median_pct': 100 * q(sb, 50), 'below_q25_pct': 100 * q(sb, 25), 'below_q75_pct': 100 * q(sb, 75),
        'above_median_pct': 100 * q(sa, 50), 'above_q25_pct': 100 * q(sa, 25), 'above_q75_pct': 100 * q(sa, 75),
        'cohort_weighted_inside_pct_all': 100 * float((R[f'share_inside_{tg}'] * R.D_veh).sum() / R.D_veh.sum()),
        'n_full_cohort': int((R[f'share_inside_{tg}'] >= 1 - 1e-9).sum()), 'n_share_ge_0_9': int((R[f'share_inside_{tg}'] >= 0.9).sum()),
        'n_mean_delay_admitted': int(R[f'mean_delay_admitted_{tg}'].sum()),
        'n_peak_delay_admitted': int(R[f'peak_delay_admitted_{tg}'].sum())}

# Worked example for Appendix F: the longest admitted detector-78 PM episode.
ex = R[(R.corridor == 'AZ_I10_W') & (R.sensor.astype(str) == '78') & (R.period == 'PM')].sort_values('P_hat_h', ascending=False)
assert len(ex) >= 1
ex = ex.iloc[0]
summary['example'] = {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else str(v)) for k, v in ex.items()}
(OUT / 'cohort_admissibility_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
