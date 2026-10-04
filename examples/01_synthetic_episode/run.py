"""Example 1 (synthetic): one closed bottleneck episode from queue to emissions.

All inputs are in config.json. The script checks, in order:
  1. the queued state on the triangular fundamental diagram (Eqs. 1-2);
  2. vehicle conservation and total delay of the Newell closed queue (Eqs. 3-4); the 9/16 mean/peak ratio is a
     property of Newell's quadratic inflow model only;
  3. the finite-link condition for the peak-delay vehicle (Eq. 9);
  4. the per-vehicle two-speed emission and the episode total, in both the delay form (Eq. 10) and the
     VMT/VHT form (Eq. 12).
Outputs: figure.png, results.json; compared with expected/results.json. Units are printed with every value.
Run from anywhere:  python examples/01_synthetic_episode/run.py
"""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))   # works without installation; pip install -e . also works
import qvdfe  # noqa: E402
from qvdfe.report import write_and_check  # noqa: E402

cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
C, vf, kj, mu, P, b, L = (float(cfg[k]) for k in ('C_vphpl', 'vf_mph', 'kj_veh_per_mi_lane', 'mu_vphpl', 'P_h',
                                                    'newell_b_veh_per_h3', 'L_mi'))
pol = cfg['pollutant']
coef = qvdfe.load_rates()[pol]['coef']

# 1. Queued state.
kc, omega = qvdfe.fd_triangular(C, vf, kj)
vq, kq, _ = qvdfe.queue_speed(mu, C, vf, kj)
vq, kq = float(vq), float(kq)

# 2. Closed queue: analytic Newell values and a numerical integration of the same inflow.
nq = qvdfe.NewellQueue(mu=mu, P=P, b=b)
t = np.linspace(0, P, 4001)
num = qvdfe.closed_queue(t, nq.lam(t), mu)

# 3. Finite-link condition.
allow = qvdfe.finite_link_allowance(L, vf, vq)

# 4. Emissions.
mean = qvdfe.link_emission(L, vf, vq, nq.mean_delay, coef)
peak = qvdfe.link_emission(L, vf, vq, nq.peak_delay, coef)
E_delay_form = qvdfe.episode_total(nq.D, nq.mean_delay, L, vf, vq, coef)
VMT, VHT = nq.D * L, nq.D * (L / vf + nq.mean_delay)
E_vmt_form = qvdfe.vmt_vht_form(VMT, VHT, vf, vq, coef)

res = {
    'state': {'kc_veh_mi_lane': kc, 'omega_mph': omega, 'kq_veh_mi_lane': kq, 'vq_mph': vq},
    'queue': {'D_veh_lane': nq.D, 'W_P_veh_h_lane': nq.W, 'peak_delay_min': nq.peak_delay * 60,
              'mean_delay_min': nq.mean_delay * 60, 'mean_over_peak_newell': nq.mean_delay / nq.peak_delay,
              'numerical_D_veh_lane': num['D'], 'numerical_W_P_veh_h_lane': num['W_P'],
              'numerical_closure_error_veh': num['closure_error']},
    'finite_link': {'allowance_min': allow * 60, 'peak_vehicle_ok': bool(nq.peak_delay <= allow),
                    'T_free_peak_vehicle_min': peak['T_free'] * 60, 'T_queue_peak_vehicle_min': peak['T_queue'] * 60},
    'emission': {'pollutant': pol, 'e0_g_per_veh': mean['e0'], 'Gamma_g_per_veh_h': mean['Gamma'],
                 'e_mean_vehicle_g': mean['e'], 'E_P_delay_form_kg_lane': E_delay_form / 1000,
                 'E_P_vmt_vht_form_kg_lane': E_vmt_form / 1000, 'VMT_veh_mi': VMT, 'VHT_veh_h': VHT},
}

print('SYNTHETIC EXAMPLE (illustrative values, not observed data)')
print(f'1. queued state: omega = {omega:.2f} mph, k_q = {kq:.2f} veh/mi/lane, v_q = {vq:.2f} mph')
print(f'2. closed queue: D = mu P = {nq.D:.0f} veh/lane (numerical {num["D"]:.3f}); W_P = {nq.W:.3f} veh h/lane '
      f'(numerical {num["W_P"]:.3f}); peak delay {nq.peak_delay*60:.3f} min, mean delay {nq.mean_delay*60:.3f} min; '
      f'mean/peak = {nq.mean_delay/nq.peak_delay:.4f} (9/16 for the Newell model only)')
print(f'3. finite link: allowance L(1/vq - 1/vf) = {allow*60:.3f} min; peak-delay vehicle '
      f'{"fits" if nq.peak_delay <= allow else "does NOT fit"} (T_free = {peak["T_free"]*60:.3f} min)')
print(f'4. {pol}: e0 = {mean["e0"]:.2f} g/veh, Gamma = {mean["Gamma"]:.2f} g/(veh h), mean-delay vehicle e = {mean["e"]:.2f} g;')
print(f'   episode total {E_delay_form/1000:.4f} kg/lane (delay form) = {E_vmt_form/1000:.4f} kg/lane (VMT/VHT form)')

checks = {'conservation_D': abs(num['D'] - nq.D) < 1e-6 * nq.D, 'total_delay_W': abs(num['W_P'] - nq.W) < 1e-6 * nq.W,
          'closed_queue': num['closure_error'] < 1e-6 * nq.D, 'peak_vehicle_fits_link': bool(nq.peak_delay <= allow),
          'two_forms_agree': abs(E_delay_form - E_vmt_form) < 1e-9 * abs(E_delay_form)}
res['checks'] = checks
print('checks:', ', '.join(f'{k} {"PASS" if v else "FAIL"}' for k, v in checks.items()))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ref = 0.9 * mu   # oblique plot: subtracting a fixed rate makes the gap (the queue) visible
    ax[0].plot(t, mu * t + nq.Q(t) - ref * t, color='#2a78d6', label='arrivals A(t)')
    ax[0].plot(t, mu * t - ref * t, color='k', label='departures D(t)')
    ax[0].fill_between(t, mu * t - ref * t, mu * t + nq.Q(t) - ref * t, color='#2a78d6', alpha=.12, label='queue Q(t)')
    ax[0].set(xlabel='time from onset (h)', ylabel=f'N(t) - {ref:.0f} t  (veh/lane)', title='Cumulative curves (oblique)')
    ax[0].legend(frameon=False)
    ax[1].plot(t, nq.w(t) * 60, color='#8C1D40', label='delay w(t)')
    ax[1].axhline(allow * 60, color='k', ls='--', label='finite-link allowance')
    ax[1].axhline(nq.mean_delay * 60, color='grey', ls=':', label='mean delay')
    ax[1].set(xlabel='time from onset (h)', ylabel='delay (min)', title='Delay and the finite-link condition')
    ax[1].legend(frameon=False, fontsize=8)
    ww = np.linspace(0, allow, 50)
    ax[2].plot(ww * 60, [qvdfe.link_emission(L, vf, vq, x, coef)['e'] for x in ww], color='k')
    ax[2].set(xlabel='delay w (min)', ylabel=f'{pol} per vehicle (g)', title='e = e0 + Gamma w')
    for a_ in ax:
        a_.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Synthetic example', fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / 'figure.png', dpi=130)
except ImportError:
    pass

ok = write_and_check(HERE, res) and all(checks.values())
sys.exit(0 if ok else 1)
