"""Frozen-result reproduction: the core package recomputes published values from the frozen data.

Displayed paper values are checked against their own rounding. Rounded components are not required to add to the
rounded total (Table F1: 294.6 + 31.7 = 326.3 as displayed, while the full-precision total rounds to 326.2).
"""
import json

import numpy as np
import pytest

import qvdfe
from qvdfe.data import load_reference_state

RATES = qvdfe.load_rates()


def rounds_to(value, shown, decimals):
    return abs(round(value, decimals) - shown) < 0.5 * 10 ** (-decimals) + 1e-12


# Table F1 (Appendix F): per-vehicle emissions at the detector-78 reference state, as printed.
TABLE_F1 = {'CO2': (294.6, 31.7, 326.2, 1), 'NOx': (0.4138, -0.0658, 0.3480, 4),
            'CO': (1.9527, 0.6141, 2.5669, 4), 'HC': (0.0819, 0.0263, 0.1082, 4)}


@pytest.mark.parametrize('p', qvdfe.POLLUTANTS)
def test_table_F1_full_precision_and_display(p):
    ref = load_reference_state()
    e = qvdfe.link_emission(ref['L_mi'], ref['vf_mph'], ref['vq_mph'], ref['mean_delay_h'], RATES[p]['coef'])
    inc = e['Gamma'] * ref['mean_delay_h']
    assert np.isclose(e['e0'] + inc, e['e'], rtol=1e-12)                         # full-precision sum
    s0, si, st, d = TABLE_F1[p]
    assert rounds_to(e['e0'], s0, d) and rounds_to(inc, si, d) and rounds_to(e['e'], st, d)


def test_reference_state_queue_speed_and_transitions():
    ref = load_reference_state()
    vq, kq, om = qvdfe.queue_speed(ref['mu_vphpl'], ref['C_vphpl'], ref['vf_mph'], ref['kj_veh_mi_lane'])
    assert np.isclose(vq, ref['vq_mph'], rtol=1e-10) and np.isclose(om, ref['omega_mph'], rtol=1e-10)
    g = ref['transition_geometry']
    dv = ref['vf_mph'] - ref['vq_mph']
    Td, Tu = qvdfe.transition_time_s(dv, 1.5), qvdfe.transition_time_s(dv, 1.0)
    assert np.isclose(Td, g['T_down_s'], rtol=1e-9) and np.isclose(Tu, g['T_up_s'], rtol=1e-9)
    lo, hi = qvdfe.transition_band(ref['L_mi'], ref['vf_mph'], ref['vq_mph'], Td / 3600, Tu / 3600)
    assert np.isclose(lo * 60, g['admissible_delay_lower_min'], rtol=1e-9)
    assert np.isclose(hi * 60, g['admissible_delay_upper_min'], rtol=1e-9)
    assert (lo <= ref['mean_delay_h'] <= hi) == g['mean_admitted'] and (lo <= ref['peak_delay_h'] <= hi) == g['peak_admitted']


def test_episode_counts():
    E = qvdfe.load_episodes()
    fitted = E.calibration_eligible & np.isfinite(E.P_hat_h)
    assert (len(E), int(fitted.sum()), int(E.matched_ladder_pass.sum())) == (325, 286, 137)


def test_core_gamma_and_queue_speed_reproduce_frozen_episode_values():
    E = qvdfe.load_episodes()
    A = E[E.matched_ladder_pass]
    for p in qvdfe.POLLUTANTS:
        g = np.array([qvdfe.gamma_cubic(q, f, RATES[p]['coef']) for q, f in zip(A.vq_hat_mph, A.vf_mph)])
        s = A[f'Gamma_cubic_{p}'].to_numpy(float)
        assert np.allclose(g, s, rtol=1e-9, atol=1e-9 * np.abs(s).max())
    F = E[E.calibration_eligible & np.isfinite(E.P_hat_h)]
    vq, _, _ = qvdfe.queue_speed(F.mu_hat_vphpl.to_numpy(), F.C_vphpl.to_numpy(), F.vf_mph.to_numpy(), F.kj.to_numpy())
    assert np.allclose(vq, F.vq_hat_mph.to_numpy(), rtol=1e-10, atol=1e-10)
    m = (F.theta.to_numpy() <= 2 / 3) & np.isfinite(F.profile_shape_a.to_numpy())   # shape stored only where used
    assert m.sum() >= 30
    a = np.array([qvdfe.profile_shape(t) for t in F.theta.to_numpy()[m]])
    assert np.allclose(a, F.profile_shape_a.to_numpy()[m], rtol=1e-9, atol=1e-12)


# Table 8 of the paper, as printed: (R2, bias %, WAPE %) for M and S.
TABLE_8 = {'CO2': ((0.906, 20.7, 22.4), (0.991, -4.6, 5.9)), 'NOx': ((0.505, 46.5, 47.7), (0.953, -9.6, 12.4)),
           'CO': ((0.968, 2.1, 11.2), (0.992, -2.9, 6.1)), 'HC': ((0.964, 5.0, 12.6), (0.996, -2.5, 4.2))}


@pytest.mark.parametrize('p', qvdfe.POLLUTANTS)
def test_table_8_metrics(p):
    A = qvdfe.load_episodes()
    A = A[A.matched_ladder_pass]
    o = A[f'O_{p}'].to_numpy(float)
    for m, shown in zip(('M', 'S'), TABLE_8[p]):
        x = A[f'{m}_{p}'].to_numpy(float)
        r2 = 1 - ((x - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
        bias = 100 * (x.sum() / o.sum() - 1)
        wape = 100 * np.abs(x - o).sum() / o.sum()
        assert rounds_to(r2, shown[0], 3) and rounds_to(bias, shown[1], 1) and rounds_to(wape, shown[2], 1), (m, r2, bias, wape)


# Table F2 (Appendix F, second example) at the illustrative peak magnitudes (1.5, 1.0) m/s^2, as printed.
def test_table_F2_long_episode_cohort_shares():
    summary = json.loads((qvdfe.frozen_dir() / 'applicability' / 'cohort_admissibility_summary.json').read_text(encoding='utf-8'))
    E = qvdfe.load_episodes()
    r = E[E.episode_id == summary['example']['episode_id']].iloc[0]
    prof = qvdfe.delay_profile(r.P_hat_h, r.wt2_hat_h, r.theta, r.mu_hat_vphpl)
    dv = r.vf_mph - r.vq_hat_mph
    Td, Tu = qvdfe.transition_time_s(dv, 1.5), qvdfe.transition_time_s(dv, 1.0)
    lo, hi = qvdfe.transition_band(r.L_mi, r.vf_mph, r.vq_hat_mph, Td / 3600, Tu / 3600)
    s = qvdfe.cohort_shares(prof['t'], prof['lam'], prof['w'], lo, hi)
    assert rounds_to(Td, 22.7, 1) and rounds_to(Tu, 34.0, 1)
    assert rounds_to(lo * 60, 0.37, 2) and rounds_to(hi * 60, 1.88, 2)
    assert rounds_to(100 * s['share_inside'], 29.5, 1) and rounds_to(100 * s['share_below'], 31.0, 1)
    assert rounds_to(100 * s['share_above'], 39.4, 1)


def test_reference_state_printed_values():
    ref = load_reference_state()
    dv = ref['vf_mph'] - ref['vq_mph']
    Td, Tu = qvdfe.transition_time_s(dv, 1.5), qvdfe.transition_time_s(dv, 1.0)
    lo, hi = qvdfe.transition_band(ref['L_mi'], ref['vf_mph'], ref['vq_mph'], Td / 3600, Tu / 3600)
    assert rounds_to(Td, 20.79, 2) and rounds_to(Tu, 31.19, 2)        # Appendix F text
    assert rounds_to(lo * 60, 0.314, 3) and rounds_to(hi * 60, 1.421, 3)
    assert rounds_to(ref['vq_mph'], 17.65, 2)


def test_response_reproduces_frozen_elasticity_cards():
    import pandas as pd
    F = qvdfe.frozen_dir() / 'calibration' / 'average_weekday'
    M = pd.read_csv(F / 'marginal_emission_card.csv')
    M = M[M.valid.astype(str).str.lower() == 'true']
    pc = pd.read_csv(F / 'parameter_card.csv').set_index(['corridor', 'period'])
    assert len(M) == 44
    for r in M.itertuples():
        c = pc.loc[(r.corridor, r.period)]
        card = dict(fd=c.fd, n=c.n, fp=c.fp, s=c.s, theta=c.theta, C=r.C_vphpl, vf=r.vf_mph, L=r.L_mi)
        o = qvdfe.response(card, r.x_h, RATES[r.pollutant]['coef'])
        for mine, frozen in (('vq', 'vq'), ('Gamma', 'Gamma'), ('eps_Gamma', 'epsilon_Gamma_x'),
                             ('increment_g', 'congestion_g_per_vehicle'), ('external_g', 'toll_g_per_vehicle')):
            want = getattr(r, frozen)
            assert np.isclose(o[mine], want, rtol=1e-9, atol=1e-9 * max(1.0, abs(want))), (r.corridor, r.period, mine)


def test_episode_total_reproduces_frozen_two_speed_totals():
    A = qvdfe.load_episodes()
    A = A[A.matched_ladder_pass]
    for p in qvdfe.POLLUTANTS:
        v = np.array([qvdfe.episode_total(r.D_veh, r.wbar_hat_h, r.L_mi, r.vf_mph, r.vq_hat_mph, RATES[p]['coef'])
                      for r in A.itertuples()])
        assert np.allclose(v, A[f'A_{p}'].to_numpy(float), rtol=1e-9)


def test_committed_tutorials_match_their_generator():
    import importlib.util
    root = qvdfe.repo_root()
    spec = importlib.util.spec_from_file_location('build_tutorials', root / 'teaching' / 'build_tutorials.py')
    mod = importlib.util.module_from_spec(spec)
    import sys
    argv, sys.argv = sys.argv, ['build_tutorials.py']
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    stale = mod.stale_notebooks()
    assert not stale, f'regenerate with python teaching/build_tutorials.py: {stale}'
