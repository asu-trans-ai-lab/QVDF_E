"""Example 3 (paper Appendix F, Table F2): applicability conditions on one long average-weekday episode.

The longest admitted detector-78 PM episode is worked through two different conditions:
  * baseline feasibility (Eq. 9): the delay fits the finite link in the two-speed state;
  * common-transition feasibility (Eq. 26): both finite transitions also fit.
The cohort is reconstructed from the fitted profile (the paper's reconstruction rule), split by the band of Eq. (26), and the
admitted subset's own volume D_A and vehicle-weighted mean delay wbar_A enter Eq. (27):
    E_A = D_A [e0 + Gamma wbar_A + de_tr].
The result is the ADMITTED-SUBSET SPEED-ONLY EMISSIONS: it is not the full-episode total and it contains no
acceleration-dependent (operating-mode) term. The script also shows the error from substituting the full-episode
mean delay for wbar_A.
"""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
import qvdfe  # noqa: E402
from qvdfe.report import write_and_check  # noqa: E402

cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
E = qvdfe.load_episodes()
r = E[E.episode_id == cfg['episode_id']].iloc[0]
rates = qvdfe.load_rates()
vf, vq, L = float(r.vf_mph), float(r.vq_hat_mph), float(r.L_mi)

# Baseline feasibility.
allow = qvdfe.finite_link_allowance(L, vf, vq)
prof = qvdfe.delay_profile(float(r.P_hat_h), float(r.wt2_hat_h), float(r.theta), float(r.mu_hat_vphpl))

# Transition feasibility.
Td_s = qvdfe.transition_time_s(vf - vq, cfg['a_down_ms2'])
Tu_s = qvdfe.transition_time_s(vf - vq, cfg['a_up_ms2'])
lo, hi = qvdfe.transition_band(L, vf, vq, Td_s / 3600, Tu_s / 3600)
s = qvdfe.cohort_shares(prof['t'], prof['lam'], prof['w'], lo, hi)

res = {
    'episode': {'id': r.episode_id, 'D_veh_lane': float(r.D_veh), 'P_hat_h': float(r.P_hat_h), 'x_h': float(r.x_h),
                'vf_mph': vf, 'vq_hat_mph': vq, 'L_mi': L, 'peak_delay_min': float(r.wt2_hat_h) * 60,
                'mean_delay_min': float(r.wbar_hat_h) * 60, 'reconstructed_D_veh_lane': prof['D'],
                'lambda_min_vphpl': prof['lambda_min']},
    'baseline_Eq9': {'allowance_min': allow * 60, 'peak_vehicle_ok': bool(r.wt2_hat_h <= allow + 1e-12)},
    'transition_Eq26': {'T_down_s': Td_s, 'T_up_s': Tu_s, 'w_lo_min': lo * 60, 'w_hi_min': hi * 60,
                        'share_inside_pct': 100 * s['share_inside'], 'share_below_pct': 100 * s['share_below'],
                        'share_above_pct': 100 * s['share_above'], 'D_A_veh_lane': s['D_A'], 'wbar_A_min': s['wbar_A'] * 60},
    'admitted_subset_speed_only_emissions': {},
}
print('LONG EPISODE (Appendix F, Table F2):', r.episode_id)
print(f"  D = {r.D_veh:.0f} veh/lane, P_hat = {r.P_hat_h:.2f} h, v_q = {vq:.2f} mph, peak delay {r.wt2_hat_h*60:.2f} min, mean delay {r.wbar_hat_h*60:.2f} min")
print(f"1. baseline (Eq. 9): allowance {allow*60:.2f} min -> peak-delay vehicle {'fits' if res['baseline_Eq9']['peak_vehicle_ok'] else 'does not fit'}")
print(f"2. transitions (Eq. 26, a = {cfg['a_down_ms2']}/{cfg['a_up_ms2']} m/s^2): T_down {Td_s:.1f} s, T_up {Tu_s:.1f} s, band {lo*60:.2f}-{hi*60:.2f} min")
print(f"   cohort inside {100*s['share_inside']:.1f}%, below {100*s['share_below']:.1f}% (near onset and clearance), above {100*s['share_above']:.1f}% (near the peak)")
print(f"   admitted subset: D_A = {s['D_A']:.0f} veh/lane, own mean delay wbar_A = {s['wbar_A']*60:.3f} min (episode mean {r.wbar_hat_h*60:.3f} min)")
print('3. admitted-subset speed-only emissions, Eq. (27), per lane (not the full-episode total; no operating-mode term):')
for p in qvdfe.POLLUTANTS:
    c = rates[p]['coef']
    de = qvdfe.speed_only_correction(Td_s / 3600, vf, vq, c) + qvdfe.speed_only_correction(Tu_s / 3600, vq, vf, c)
    EA = qvdfe.subset_total(s['D_A'], s['wbar_A'], L, vf, vq, c, de)
    # direct sum over admitted vehicles of e0 + Gamma w + de: must equal E_A
    inside = (prof['w'] >= lo) & (prof['w'] <= hi)
    e0, G = qvdfe.link_emission(L, vf, vq, 0.0, c)['e0'], float(qvdfe.gamma(vq, vf, c))
    direct = float(np.trapezoid(prof['lam'] * (e0 + G * prof['w'] + de) * inside, prof['t']))
    wrong = qvdfe.subset_total(s['D_A'], float(r.wbar_hat_h), L, vf, vq, c, de)
    res['admitted_subset_speed_only_emissions'][p] = {
        'de_tr_g_per_veh': de, 'E_A_g_lane': EA, 'direct_sum_g_lane': direct,
        'with_episode_mean_delay_g_lane': wrong, 'substitution_error_pct': 100 * (wrong / EA - 1)}
    print(f"   {p:4s} de_tr = {de:+.4g} g/veh;  E_A = {EA:.4g} g;  using the episode mean delay instead: {wrong:.4g} g ({100*(wrong/EA-1):+.2f}%)")

checks = {'reconstruction_conserves_D': abs(prof['D'] - r.D_veh) < 1e-6 * r.D_veh, 'arrivals_nonnegative': prof['lambda_min'] >= -1e-9,
          'baseline_feasible': res['baseline_Eq9']['peak_vehicle_ok'],
          'shares_sum_to_one': abs(s['share_inside'] + s['share_below'] + s['share_above'] - 1) < 1e-9,
          'subset_identity': all(np.isclose(v['E_A_g_lane'], v['direct_sum_g_lane'], rtol=1e-9)
                                 for v in res['admitted_subset_speed_only_emissions'].values())}
res['checks'] = checks
print('checks:', ', '.join(f'{k} {"PASS" if v else "FAIL"}' for k, v in checks.items()))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    t, w, lam = prof['t'], prof['w'] * 60, prof['lam']
    inside = (prof['w'] >= lo) & (prof['w'] <= hi)
    ax[0].plot(t, w, color='k')
    ax[0].axhspan(lo * 60, hi * 60, color='#8C1D40', alpha=.12, label='transition band (Eq. 26)')
    ax[0].axhline(allow * 60, color='k', ls='--', lw=.8, label='finite-link allowance (Eq. 9)')
    ax[0].axhline(r.wbar_hat_h * 60, color='grey', ls=':', label='episode mean delay')
    ax[0].axhline(s['wbar_A'] * 60, color='#8C1D40', ls='-.', label='admitted mean delay')
    ax[0].set(xlabel='time from onset (h)', ylabel='delay (min)', title='Reconstructed delay profile'); ax[0].legend(frameon=False, fontsize=7, loc='upper left')
    ax[1].fill_between(t, 0, lam, where=inside, color='#8C1D40', alpha=.35, label='admitted vehicles')
    ax[1].fill_between(t, 0, lam, where=~inside, color='grey', alpha=.25, label='not admitted')
    ax[1].set(xlabel='time from onset (h)', ylabel='arrival rate (veh/h/lane)', title='Cohort split by the band'); ax[1].set_ylim(0, 1.35 * lam.max()); ax[1].legend(frameon=False, fontsize=8, loc='upper center', ncol=2)
    for a_ in ax:
        a_.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / 'figure.png', dpi=120)
except ImportError:
    pass

ok = write_and_check(HERE, res) and all(checks.values())
sys.exit(0 if ok else 1)
