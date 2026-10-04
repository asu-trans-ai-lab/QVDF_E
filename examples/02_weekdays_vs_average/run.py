"""Example 2 (real data, paper Fig. 7): individual weekdays versus the average-weekday profile.

Detector 78, Arizona I-10 westbound, November 2016: the twelve complete weekdays and the monthly average-weekday
profile built from exactly those days. Data: legacy ADOT loop-detector records, cleaned, synchronized and
post-processed for Zhou et al. (2022), used for research purposes (extract in data/examples/).

Checks:
  1. the twelve days reproduce the monthly profile bin by bin (arithmetic mean flow; flow-weighted harmonic mean
     speed, which preserves mean vehicle-hours);
  2. the episode identified on the averaged profile is compared with the daily episodes: its duration and volume are
     not the averages of the daily episode metrics, because averaging shifts and smooths onsets and clearances.
Outputs: figure.png, results.json; compared with expected/results.json.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
import qvdfe  # noqa: E402
from qvdfe.report import write_and_check  # noqa: E402

cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
X = qvdfe.data_dir() / 'examples'
days = pd.read_csv(X / cfg['days_file'])
avg = pd.read_csv(X / cfg['average_file'])
eps = pd.read_csv(X / cfg['daily_episodes_file'])
meta = json.loads((X / cfg['meta_file']).read_text(encoding='utf-8'))

# 1. Bin-by-bin reconstruction of the monthly profile.
piv_q = days.pivot(index='minute', columns='date', values='flow_vph')
piv_v = days.pivot(index='minute', columns='date', values='speed_mph')
mean_q = piv_q.mean(axis=1)
harm_v = piv_q.sum(axis=1) / (piv_q / piv_v).sum(axis=1)
a = avg.set_index('minute')
rel_q = float((mean_q - a.flow_vph).abs().max() / a.flow_vph.abs().max())
pos = a.flow_vph > 0
rel_v = float(((harm_v - a.speed_mph).abs() / a.speed_mph)[pos].max())

# 2. Episodes: daily PM episodes versus the episode on the averaged profile.
pm = eps[eps.period == cfg['period']]
no_pm = sorted(set(meta['dates']) - set(pm.date.astype(str)))      # days whose congestion was detected in another period only
ae = meta['average_profile_episode']
res = {
    'n_days': int(piv_q.shape[1]), 'bins': int(piv_q.shape[0]),
    'profile_check': {'flow_max_rel_diff': rel_q, 'speed_max_rel_diff': rel_v, 'passes': bool(rel_q < 1e-6 and rel_v < 1e-6)},
    'days_without_pm_episode': no_pm,
    'daily_pm_episodes': {'count': int(len(pm)), 'P_h_mean': float(pm.P_h.mean()), 'P_h_min': float(pm.P_h.min()),
                          'P_h_max': float(pm.P_h.max()), 'D_veh_mean': float(pm.D_veh.mean()),
                          'mean_delay_min_mean': float(pm.wbar_h.mean() * 60), 'min_speed_mph_mean': float(pm.min_speed_mph.mean())},
    'average_profile_episode': {'t0': ae['t0'][11:16], 't3': ae['t3'][11:16], 'P_h': float(ae['P_h']), 'D_veh': float(ae['D_veh'])},
    'average_profile_min_speed_mph': float(avg[(avg.minute >= 14 * 60) & (avg.minute < 20 * 60)].speed_mph.min()),
}
print('REAL DATA: detector 78, November 2016,', res['n_days'], 'complete weekdays (legacy ADOT data, research use)')
print(f"1. monthly profile from the drawn days: max relative difference flow {rel_q:.1e}, speed {rel_v:.1e} -> "
      f"{'PASS' if res['profile_check']['passes'] else 'FAIL'}")
d = res['daily_pm_episodes']
print(f"2. {d['count']} daily PM episodes on {res['n_days']} days (no PM episode on {', '.join(no_pm) or 'none'}; that day's congestion was detected in another period).")
print(f"   PM episodes: P from {d['P_h_min']:.2f} to {d['P_h_max']:.2f} h (mean {d['P_h_mean']:.2f} h), "
      f"mean D {d['D_veh_mean']:.0f} veh/lane, mean of daily minimum speeds {d['min_speed_mph_mean']:.1f} mph")
print(f"   averaged-profile episode {res['average_profile_episode']['t0']}-{res['average_profile_episode']['t3']}: "
      f"P = {ae['P_h']:.2f} h, D = {ae['D_veh']:.0f} veh/lane; minimum averaged speed {res['average_profile_min_speed_mph']:.1f} mph")
print('   The averaged profile is one calibration target; its episode is not the average of the daily episode metrics.')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey='row')
    rep = meta['representative_date']
    h = piv_q.index / 60
    for date in piv_q.columns:
        c, lw, z = ('#2a78d6', 1.4, 3) if date == rep else ('#c8c8c8', .7, 1)
        ax[0, 0].plot(h, piv_v[date], color=c, lw=lw, zorder=z, label=rep if date == rep else None)
        ax[1, 0].plot(h, piv_q[date], color=c, lw=lw, zorder=z)
    ax[0, 1].plot(a.index / 60, a.speed_mph, color='k'); ax[1, 1].plot(a.index / 60, a.flow_vph, color='k')
    h0 = int(ae['t0'][11:13]) + int(ae['t0'][14:16]) / 60
    h3 = int(ae['t3'][11:13]) + int(ae['t3'][14:16]) / 60
    for a_ in ax[:, 1]:
        a_.axvspan(h0, h3, color='#8C1D40', alpha=.08)
    for r in pm[pm.date == rep].itertuples():
        for a_ in ax[:, 0]:
            a_.axvspan(int(r.t0[11:13]) + int(r.t0[14:16]) / 60, int(r.t3[11:13]) + int(r.t3[14:16]) / 60, color='#2a78d6', alpha=.08)
    for a_ in ax[0]:
        a_.axhline(0.7 * meta['vf_monthly_mph'], color='grey', ls='--', lw=.7)
    ax[0, 0].set(title=f"{res['n_days']} complete weekdays", ylabel='speed (mph)'); ax[0, 0].legend(frameon=False)
    ax[0, 1].set(title='average-weekday profile (same days)')
    ax[1, 0].set(xlabel='clock time (h)', ylabel='flow (veh/h/lane)'); ax[1, 1].set(xlabel='clock time (h)')
    for a_ in ax.flat:
        a_.set_xlim(6, 20); a_.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / 'figure.png', dpi=120)
except ImportError:
    pass

ok = write_and_check(HERE, res) and res['profile_check']['passes']
sys.exit(0 if ok else 1)
