# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Fig. 7: individual weekdays beside the average-weekday profile.

Detector 78 (AZ I-10 westbound), one month. Left column: recorded five-minute speed and
flow on every complete weekday that forms the monthly average-weekday profile, with one
representative day highlighted. Right column: the average-weekday profile built from exactly
those days (arithmetic mean flow; flow-weighted harmonic mean speed). The script first
verifies that the right column equals the aggregation of the left column bin by bin and
that the drawn episode boundaries reproduce the frozen calibration episode results; it refuses to
draw otherwise. Underlying data and every calculated result are unchanged.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = _ROOT
sys.path.insert(0, str(ROOT / 'scripts/figures'))
import polish_style as P  # noqa: E402
P.apply()

SENSOR, MONTH, REP = 'AZ_I10_W::78', '2016-11', '2016-11-01'
DAILY = _FROZEN / 'calibration/daily_validation/traffic_panel.csv.gz'
MONTHLY = _FROZEN / 'calibration/average_weekday/traffic_panel.csv.gz'
SUPPORT = _FROZEN / 'calibration/average_weekday/i10_monthly_day_support.csv'
DAILY_EP = _FROZEN / 'calibration/daily_validation/episode_results_v2.csv'
AVG_EP = _FROZEN / 'calibration/average_weekday/episode_results_v2.csv'
OUT = _OUT / 'calculations'
OUT.mkdir(exist_ok=True)
FIGS = _OUT / 'figs'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hour_of_day(values):
    t = pd.to_datetime(pd.Series(values))
    return ((t - t.dt.normalize()).dt.total_seconds() / 3600).to_numpy()


support = pd.read_csv(SUPPORT)
days = support[(support.sensor_uid == SENSOR) & (support.month == MONTH) & support.included]['date'].astype(str).tolist()
assert REP in days, (REP, days)

daily = pd.read_csv(DAILY, usecols=['sensor_uid', 'date', 'minute', 'datetime', 'flow_vph', 'speed_mph', 'lanes',
                                     'vf_mph', 'C_vphpl', 'holdout'], low_memory=False)
daily = daily[(daily.sensor_uid == SENSOR) & daily.date.astype(str).isin(days)].copy()
daily['date'] = daily.date.astype(str)
missing = sorted(set(days) - set(daily.date.unique()))
assert not missing, missing
daily['h'] = hour_of_day(daily.datetime)
assert (daily.groupby('date').size() == 288).all()

monthly = pd.read_csv(MONTHLY, usecols=['sensor_uid', 'month', 'minute', 'flow_vph', 'speed_mph', 'days', 'vf_mph', 'lanes'],
                      low_memory=False)
monthly = monthly[(monthly.sensor_uid == SENSOR) & (monthly.month == MONTH)].sort_values('minute').reset_index(drop=True)
assert len(monthly) == 288 and int(monthly.days.iloc[0]) == len(days), (len(monthly), monthly.days.iloc[0], len(days))
assert monthly.minute.is_unique and monthly.minute.max() == 1435, (monthly.minute.min(), monthly.minute.max())

# Flow convention: find which of flow_vph or flow_vph/lanes reproduces the counted episode demand D.
dep = pd.read_csv(DAILY_EP)
rep_eps = dep[(dep.sensor_uid == SENSOR) & (dep.date.astype(str) == REP)].copy()
assert len(rep_eps) >= 1
rep_day = daily[daily.date == REP].sort_values('minute').reset_index(drop=True)


def counted(flow, t0, t3):
    h0, h3 = hour_of_day([t0])[0], hour_of_day([t3])[0]
    m = (rep_day.h.to_numpy() >= h0) & (rep_day.h.to_numpy() < h3)
    return float((flow[m] / 12).sum())


conv = None
for name, flow in [('per_lane', rep_day.flow_vph.to_numpy()),
                   ('total_over_lanes', (rep_day.flow_vph / rep_day.lanes).to_numpy())]:
    errs = [abs(counted(flow, r.t0, r.t3) - r.D_veh) / r.D_veh for r in rep_eps.itertuples()]
    if max(errs) < 1e-6:
        conv = name
        break
assert conv, 'neither flow convention reproduces the counted demand'


def per_lane(frame):
    return frame.flow_vph.to_numpy() if conv == 'per_lane' else (frame.flow_vph / frame.lanes).to_numpy()


# Verify the monthly profile equals the aggregation of the drawn days.
piv_q = daily.pivot(index='h', columns='date', values='flow_vph').sort_index()
piv_v = daily.pivot(index='h', columns='date', values='speed_mph').sort_index()
mean_q = piv_q.mean(axis=1).to_numpy()
w = piv_q.to_numpy()
v = piv_v.to_numpy()
with np.errstate(divide='ignore', invalid='ignore'):
    harm = w.sum(axis=1) / (w / v).sum(axis=1)
mq = monthly.flow_vph.to_numpy()
mv = monthly.speed_mph.to_numpy()
rel_q = np.abs(mean_q - mq) / np.maximum(np.abs(mq), 1e-9)
pos = w.sum(axis=1) > 0
rel_v = np.abs(harm[pos] - mv[pos]) / np.maximum(np.abs(mv[pos]), 1e-9)
checks = {'flow_max_rel_diff': float(rel_q.max()), 'speed_max_rel_diff_positive_flow_bins': float(rel_v.max()),
          'positive_flow_bins': int(pos.sum()), 'flow_convention': conv}
assert rel_q.max() < 1e-6, checks
assert rel_v.max() < 1e-6, checks

aep = pd.read_csv(AVG_EP)
avg_ep = aep[(aep.sensor_uid == SENSOR) & (aep.date.astype(str) == MONTH + '-01')]
assert len(avg_ep) == 1
avg_ep = avg_ep.iloc[0]
monthly['h'] = monthly.minute / 60.0
mq_lane = per_lane(monthly)
hv = monthly.h.to_numpy()
ma = (hv >= hour_of_day([avg_ep.t0])[0]) & (hv < hour_of_day([avg_ep.t3])[0])
d_avg = float((mq_lane[ma] / 12).sum())
assert abs(d_avg - avg_ep.D_veh) / avg_ep.D_veh < 1e-6, (d_avg, avg_ep.D_veh)

vf_d = float(rep_day.vf_mph.iloc[0])
vf_m = float(monthly.vf_mph.iloc[0])

# ---------- draw ----------
fig, axes = plt.subplots(2, 2, figsize=(P.TEXT_WIDTH_IN, 5.0), sharex=True, layout='constrained')
(a, b), (c, d) = axes
x0, x1 = 6.0, 21.0
def window(frame):
    return frame[(frame.h >= x0 - 1 / 12) & (frame.h <= x1 + 1 / 12)]


for date in days:
    if date == REP:
        continue
    day = window(daily[daily.date == date].sort_values('minute'))
    a.plot(day.h, day.speed_mph, color=P.LIGHT, lw=0.6, zorder=1)
    c.step(day.h, per_lane(day), where='post', color=P.LIGHT, lw=0.6, zorder=1)
rep_w = window(rep_day)
mon_w = window(monthly)
a.plot(rep_w.h, rep_w.speed_mph, color=P.ACCENT, lw=1.4, zorder=3)
c.step(rep_w.h, per_lane(rep_w), where='post', color=P.ACCENT, lw=1.1, zorder=3)
b.plot(mon_w.h, mon_w.speed_mph, color=P.INK, lw=1.4, zorder=3)
d.step(mon_w.h, per_lane(mon_w), where='post', color=P.INK, lw=1.1, zorder=3)


def episode_marks(ax_speed, ax_flow, eps, flow_h, flow_vals, fill):
    for e in eps:
        h0, h3 = hour_of_day([e['t0']])[0], hour_of_day([e['t3']])[0]
        for ax in (ax_speed, ax_flow):
            ax.axvline(h0, color=P.DARK, ls=':', lw=0.7, zorder=2)
            ax.axvline(h3, color=P.DARK, ls=':', lw=0.7, zorder=2)
        ax_speed.text(h0, 0.98, r'$t_0$', transform=ax_speed.get_xaxis_transform(), ha='right', va='top',
                      fontsize=7.5, color=P.DARK)
        ax_speed.text(h3, 0.98, r'$t_3$', transform=ax_speed.get_xaxis_transform(), ha='left', va='top',
                      fontsize=7.5, color=P.DARK)
        m = (flow_h >= h0) & (flow_h < h3)
        edges = np.r_[flow_h[m], h3]
        ax_flow.stairs(flow_vals[m], edges, fill=True, color=fill, alpha=0.15, zorder=1.5)


rep_list = [{'t0': r.t0, 't3': r.t3, 'P_h': float(r.P_h), 'D_veh': float(r.D_veh), 'period': r.period}
            for r in rep_eps.sort_values('t0').itertuples()]
episode_marks(a, c, rep_list, rep_day.h.to_numpy(), per_lane(rep_day), P.ACCENT)
avg_list = [{'t0': avg_ep.t0, 't3': avg_ep.t3, 'P_h': float(avg_ep.P_h), 'D_veh': float(avg_ep.D_veh),
             'period': avg_ep.period}]
episode_marks(b, d, avg_list, hv, mq_lane, P.INK)

for ax, vf in ((a, vf_d), (b, vf_m)):
    ax.plot([x0, x1], [vf, vf], color=P.MID, lw=0.8, ls='--', zorder=2)
    ax.plot([x0, x1], [0.7 * vf, 0.7 * vf], color=P.MID, lw=0.8, ls='-.', zorder=2)
    ax.text(x1 - 0.1, vf + 1.5, r'$v_f$', ha='right', va='bottom', fontsize=7.5, color=P.MID)
    ax.text(x1 - 0.1, 0.7 * vf + 1.5, r'$0.7v_f$', ha='right', va='bottom', fontsize=7.5, color=P.MID)

longest = max(rep_list, key=lambda e: e['P_h'])
h0r, h3r = hour_of_day([longest['t0']])[0], hour_of_day([longest['t3']])[0]
mid = rep_day[(rep_day.h >= h0r) & (rep_day.h < h3r)]
i = mid.speed_mph.idxmin()
ts = pd.Timestamp(REP)
a.annotate(f"{ts.day} {ts.strftime('%b')}", xy=(mid.loc[i, 'h'], mid.loc[i, 'speed_mph']),
           xytext=(mid.loc[i, 'h'] - 3.0, mid.loc[i, 'speed_mph'] + 24), color=P.ACCENT, fontsize=7.5,
           arrowprops=dict(arrowstyle='-', lw=0.6, color=P.ACCENT))
a.text(x0 + 0.2, 0.06, f'{len(days)} complete weekdays', transform=a.get_xaxis_transform(), fontsize=7.5,
       color=P.MID, va='bottom')
ymax_s = max(vf_d, vf_m, daily.speed_mph.max(), monthly.speed_mph.max()) * 1.15
ymax_f = max(per_lane(daily).max(), mq_lane.max()) * 1.18
a.set(ylabel='Speed (mph)', ylim=(0, ymax_s), title='(a) Individual weekdays: speed')
b.set(ylim=(0, ymax_s), title='(b) Average weekday: speed')
c.set(ylabel='Flow (veh/h/lane)', ylim=(0, ymax_f), xlabel='Clock time (h)', title='(c) Individual weekdays: flow')
d.set(ylim=(0, ymax_f), xlabel='Clock time (h)', title='(d) Average weekday: flow')
for ax in axes.flat:
    ax.set_xlim(x0, x1)
    ax.set_xticks(range(6, 22, 2))
    ax.spines[['top', 'right']].set_visible(False)
fig.savefig(FIGS / 'fig7_paired_episode.pdf')
fig.savefig(FIGS / 'fig7_paired_episode.png', dpi=600)

audit = {'sensor': SENSOR, 'month': MONTH, 'n_days': len(days), 'dates': days, 'representative_date': REP,
         'representative_episodes': rep_list, 'average_profile_episode': avg_list[0],
         'average_profile_D_recomputed': d_avg, 'vf_daily_mph': vf_d, 'vf_monthly_mph': vf_m, 'checks': checks,
         'all_days_holdout': bool(daily.holdout.all()),
         'inputs': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (DAILY, MONTHLY, SUPPORT, DAILY_EP, AVG_EP)}}
(OUT / 'fig7_paired_episode_audit.json').write_text(json.dumps(audit, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in audit.items() if k not in ('dates', 'inputs')}, indent=1))
