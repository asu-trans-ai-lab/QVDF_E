"""Extract the small datasets that the learning notebooks and the interactive dashboard share.

Everything comes from the frozen files in data/frozen. Outputs:
  docs/data/qvdfe_data.js   one JavaScript file (window.QVDFE_DATA = {...}) so the dashboard works from file:// and GitHub Pages
  docs/data/qvdfe_data.json the same object as JSON for the notebooks
No calibration or reported number is changed; this is a re-serialization of existing results.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'src'))
from qvdfe.paths import frozen_dir  # noqa: E402
F = frozen_dir()
OUT = ROOT / 'docs' / 'data'
OUT.mkdir(parents=True, exist_ok=True)
A = F / 'calibration/average_weekday'
D = F / 'calibration/daily_validation'
POLS = ['CO2', 'NOx', 'CO', 'HC']


def r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def truthy(s):
    return s.astype(str).str.lower().eq('true')


data = {}

# 1. Cubic zero-acceleration rates (g/veh h) and their validity range.
cr = pd.read_csv(F / 'calibration/cubic_rate_coefficients.csv')
data['rates'] = {row.pollutant: {'c0': row.c0, 'c1': row.c1, 'c2': row.c2, 'c3': row.c3, 'vmin': row.speed_min_mph, 'vmax': row.speed_max_mph}
                 for row in cr.itertuples()}

# 2. Reference state (Appendix F) with the saved full-precision parameters.
ref = json.loads((F / 'calibration/audit/table3_reference_state.json').read_text(encoding='utf-8'))
data['reference_state'] = {k: ref[k] for k in ('x_h', 'P_h', 'C_vphpl', 'vf_mph', 'L_mi', 'Tf_h', 'mu_vphpl', 'omega_mph', 'kq_veh_mi_lane', 'vq_mph',
                                               'peak_delay_h', 'mean_delay_h', 'kj_veh_mi_lane')}
data['reference_state']['transition_geometry'] = ref.get('transition_geometry', {})

# 3. Calibrated QVDF cards (average weekday) with the FD parameters of a representative detector per corridor.
pc = pd.read_csv(A / 'parameter_card.csv')
sp = pd.read_csv(A / 'sensor_parameters.csv')
cards = []
for row in pc.itertuples():
    sensors = sp[sp.corridor == row.corridor]
    if row.corridor == 'AZ_I10_W':
        sensors = sensors[sensors.sensor.astype(str) == '78']
    s0 = sensors.iloc[0] if len(sensors) else None
    cards.append({'corridor': row.corridor, 'period': row.period, 'scope': row.calibration_scope, 'N_source': int(row.N_source),
                  'fd': r(row.fd), 'n': r(row.n), 'fp': r(row.fp), 's': r(row.s), 'theta': r(row.theta), 'alpha': r(row.alpha), 'beta': r(row.beta),
                  'x_min': r(row.x_min_h), 'x_max': r(row.x_max_h), 'x_median': r(row.x_median_h),
                  'C': r(s0.C_vphpl) if s0 is not None else None, 'vf': r(s0.vf_mph) if s0 is not None else None,
                  'L': r(s0.L_mi) if s0 is not None else None, 'kj': r(s0.kj) if s0 is not None else None, 'omega': r(s0.omega_mph) if s0 is not None else None,
                  'sensor': str(s0.sensor) if s0 is not None else None})
data['cards'] = cards

# 4. Average-weekday episodes: calibration diagnostics, admission, emission ladder and Gamma.
ep = pd.read_csv(A / 'episode_results_v2.csv', low_memory=False)
for c in ('calibration_eligible', 'physical_predicted_pass', 'matched_ladder_pass'):
    ep[c] = truthy(ep[c])
rows = []
for e in ep.itertuples():
    d = {'id': e.episode_id, 'dataset': 'AZ' if e.corridor == 'AZ_I10_W' else 'PeMS', 'corridor': e.corridor, 'sensor': str(e.sensor), 'date': str(e.date), 'period': e.period,
         'x': r(e.x_h), 'P': r(e.P_h), 'P_hat': r(e.P_hat_h), 'D': r(e.D_veh, 1), 'vt2': r(e.vt2_obs_mph, 2), 'vt2_hat': r(e.vt2_hat_mph, 2),
         'vq_obs': r(e.vq_mph, 2), 'vq_hat': r(e.vq_hat_mph, 2), 'mu_obs': r(e.mu_obs_vphpl, 1), 'mu_hat': r(e.mu_hat_vphpl, 1),
         'L': r(e.L_mi, 3), 'vf': r(e.vf_mph, 2), 'wt2_hat_min': r(e.wt2_hat_h * 60, 3), 'wbar_hat_min': r(e.wbar_hat_h * 60, 3),
         'eligible': bool(e.calibration_eligible), 'fitted': bool(e.calibration_eligible and np.isfinite(e.P_hat_h)),
         'finite_link': bool(e.physical_predicted_pass), 'admitted': bool(e.matched_ladder_pass)}
    if e.matched_ladder_pass:
        for p in POLS:
            d[f'O_{p}'] = r(getattr(e, f'O_{p}'), 1)
            d[f'M_{p}'] = r(getattr(e, f'M_{p}'), 1)
            d[f'S_{p}'] = r(getattr(e, f'S_{p}'), 1)
            d[f'Gamma_{p}'] = r(getattr(e, f'Gamma_cubic_{p}'), 4)
    rows.append(d)
data['episodes'] = rows

# 5. Emission comparison metrics on the admitted sample (Table 8 of the paper).
adm = ep[ep.matched_ladder_pass]
metrics = {}
for p in POLS:
    o = adm[f'O_{p}'].to_numpy(float)
    for m in ('M', 'S'):
        x = adm[f'{m}_{p}'].to_numpy(float)
        metrics[f'{m}_{p}'] = {'R2': r(1 - ((x - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()), 'bias_pct': r(100 * (x.sum() / o.sum() - 1), 2),
                               'WAPE_pct': r(100 * np.abs(x - o).sum() / o.sum(), 2), 'N': int(len(o))}
data['emission_metrics'] = metrics

# 6. Elasticity cards (Fig. 12 of the paper).
mc = pd.read_csv(A / 'marginal_emission_card.csv')
mc = mc[truthy(mc.valid)]
data['elasticity_cards'] = [{'corridor': m.corridor, 'period': m.period, 'pollutant': m.pollutant, 'x': r(m.x_h), 'beta': r(m.beta), 'vq': r(m.vq, 2),
                             'Gamma': r(m.Gamma), 'eps_Gamma': r(m.epsilon_Gamma_x), 'beta_plus_eps': r(m.beta_plus_epsilon),
                             'increment_g': r(m.congestion_g_per_vehicle), 'toll_g': r(m.toll_g_per_vehicle)} for m in mc.itertuples()]

# 7. Paired figure data: detector 78, November 2016 (the same days as Fig. 7), downsampled to 5-min bins as stored.
fig7 = json.loads((F / 'applicability/fig7_paired_episode_audit.json').read_text(encoding='utf-8'))
daily = pd.read_csv(D / 'traffic_panel.csv.gz', usecols=['sensor_uid', 'date', 'datetime', 'flow_vph', 'speed_mph'], low_memory=False)
daily = daily[(daily.sensor_uid == fig7['sensor']) & daily.date.astype(str).isin(fig7['dates'])].copy()
daily['h'] = (pd.to_datetime(daily.datetime) - pd.to_datetime(daily.datetime).dt.normalize()).dt.total_seconds() / 3600
days = {}
for date, g in daily.groupby(daily.date.astype(str)):
    g = g.sort_values('h')
    days[date] = {'h': [r(v, 4) for v in g.h], 'speed': [r(v, 2) for v in g.speed_mph], 'flow': [r(v, 1) for v in g.flow_vph]}
monthly = pd.read_csv(A / 'traffic_panel.csv.gz', usecols=['sensor_uid', 'month', 'minute', 'flow_vph', 'speed_mph'], low_memory=False)
monthly = monthly[(monthly.sensor_uid == fig7['sensor']) & (monthly.month == fig7['month'])].sort_values('minute')
data['fig7'] = {'sensor': fig7['sensor'], 'month': fig7['month'], 'dates': fig7['dates'], 'representative_date': fig7['representative_date'],
                'representative_episodes': fig7['representative_episodes'], 'average_profile_episode': fig7['average_profile_episode'],
                'vf_daily': fig7['vf_daily_mph'], 'vf_monthly': fig7['vf_monthly_mph'], 'days': days,
                'average': {'h': [r(v / 60, 4) for v in monthly.minute], 'speed': [r(v, 2) for v in monthly.speed_mph], 'flow': [r(v, 1) for v in monthly.flow_vph]}}

# 8. The three applicability calculations.
coh = json.loads((F / 'applicability/cohort_admissibility_summary.json').read_text(encoding='utf-8'))
data['cohort'] = {'pairs': coh['pairs'], 'example': {k: coh['example'][k] for k in coh['example'] if not k.startswith(('T_down', 'T_up')) or '1.5_1' in k}}
cor = json.loads((F / 'applicability/corridor_admissibility_summary.json').read_text(encoding='utf-8'))
data['corridor'] = {'span_mi': cor['corridor_span_mi'], 'Tff_min': cor['corridor_Tff_min'], 'per_detector': cor['per_detector'], 'corridor_defined': cor['corridor_defined_episodes']}
day = json.loads((F / 'applicability/daytime_period_summary.json').read_text(encoding='utf-8'))
data['daytime'] = {'coverage': day['day_coverage'], 'metrics_by_sample': day['day_metrics_by_sample'], 'boundary_crossing': day['boundary_crossing']}
ex = pd.read_csv(F / 'calibration/audit/sequential_exclusion_counts.csv')
data['exclusions'] = ex.to_dict(orient='records')

# 9. Pigou two-route results (Appendix E) and the time-CO2 trade-off curve, downsampled.
pg = pd.read_csv(F / 'calibration/pigou/full_surrogate_poa.csv')
data['pigou_poa'] = pg.round(6).to_dict(orient='records')
sel = json.loads((F / 'calibration/pigou/pigou_selection.json').read_text(encoding='utf-8'))
data['pigou_selection'] = {k: sel[k] for k in sel if not isinstance(sel[k], (dict, list)) and 'source' not in k and 'path' not in k}   # no local paths
for kind in ('full', 'surrogate'):
    p = F / 'calibration/pigou' / kind / 'network_curves.csv'
    if p.exists():
        nc = pd.read_csv(p)
        keep = [c for c in nc.columns if any(k in c.lower() for k in ('share', 'f_over', 'time', 'co2', 'eta', 'weight'))][:8]
        step = max(1, len(nc) // 300)
        data[f'pigou_curve_{kind}'] = {'columns': keep, 'rows': nc[keep].iloc[::step].round(5).values.tolist()}

js = 'window.QVDFE_DATA = ' + json.dumps(data, separators=(',', ':')) + ';\n'
(OUT / 'qvdfe_data.js').write_text(js, encoding='utf-8')
(OUT / 'qvdfe_data.json').write_text(json.dumps(data, indent=0), encoding='utf-8')
print({k: (len(v) if hasattr(v, '__len__') else v) for k, v in data.items() if k not in ('reference_state',)}, 'bytes:', (OUT / 'qvdfe_data.js').stat().st_size)
