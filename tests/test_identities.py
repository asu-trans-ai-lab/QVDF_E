"""Mathematical identities and conservation checks (fast; they need only the cubic rate table).

Comparisons use numpy.isclose with both an absolute and a relative tolerance. If the data folder is missing, the
whole module is skipped with a message instead of failing at import.
"""
import os

import numpy as np
import pytest

import qvdfe

try:
    RATES = qvdfe.load_rates()
except FileNotFoundError as exc:   # data folder not available
    pytest.skip(f'rate table not available: {exc}', allow_module_level=True)
CO2 = RATES['CO2']['coef']


def close(a, b, rtol=1e-6, atol=1e-9):
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


# ---------------- closed queue -----------------------------------------------------------------
@pytest.mark.parametrize('profile', ['newell', 'sine'])
def test_closed_queue_conservation_and_total_delay(profile):
    mu, P = 1500.0, 2.0
    t = np.linspace(0, P, 8001)
    if profile == 'newell':
        lam = qvdfe.NewellQueue(mu=mu, P=P, b=100).lam(t)
    else:   # a second closed profile: sinusoidal surplus with zero net excess
        lam = mu + 120 * np.sin(2 * np.pi * t / P)
    q = qvdfe.closed_queue(t, lam, mu)
    assert q['closure_error'] < 1e-6 * mu * P
    assert close(q['D'], mu * P, rtol=1e-8)                                  # D = mu P
    assert close(q['W_P'], q['D'] * q['mean_delay_vehicle'], rtol=1e-4)      # integral Q = integral lam w = D * wbar
    assert close(q['mean_delay_time'], q['mean_delay_vehicle'], rtol=1e-4)    # time average = vehicle average


def test_newell_nine_sixteenths_only_for_newell():
    n = qvdfe.NewellQueue(mu=1500, P=2, b=100)
    assert close(n.mean_delay / n.peak_delay, 9 / 16, rtol=1e-12)
    t = np.linspace(0, 2, 8001)
    q = qvdfe.closed_queue(t, 1500 + 120 * np.sin(np.pi * t), 1500)
    assert not close(q['mean_delay_time'] / q['w'].max(), 9 / 16, rtol=1e-3)   # another closed profile, another ratio


# ---------------- states -----------------------------------------------------------------------
def test_queue_speed_on_fd():
    kc, om = qvdfe.fd_triangular(2000, 60, 200)
    assert close(kc * 60, 2000) and close(om * (200 - kc), 2000)
    vq, kq, _ = qvdfe.queue_speed(1500, 2000, 60, 200)
    assert close(vq * kq, 1500) and close(om * (200 - kq), 1500)              # on the congested branch
    vc, _, _ = qvdfe.queue_speed(2000, 2000, 60, 200)
    assert close(vc, 60)                                                      # mu = C gives vf


# ---------------- two-speed emissions ----------------------------------------------------------
@pytest.mark.parametrize('p', qvdfe.POLLUTANTS)
@pytest.mark.parametrize('vq', [5.0, 12.0, 17.65, 30.0, 45.0, 59.0])
def test_gamma_closed_form_equals_chord_form(p, vq):
    c, vf = RATES[p]['coef'], 64.0
    g1 = qvdfe.gamma(vq, vf, c)
    g2 = qvdfe.gamma_cubic(vq, vf, c)
    f = lambda v: qvdfe.cubic_rate(v, c) / v
    g3 = (f(vq) - f(vf)) / (1 / vq - 1 / vf)
    scale = abs(qvdfe.cubic_rate(vf, c))
    assert close(g1, g2, rtol=1e-10, atol=1e-10 * scale) and close(g1, g3, rtol=1e-10, atol=1e-10 * scale)


def test_gamma_near_sign_change_uses_absolute_tolerance():
    c, vf = RATES['CO2']['coef'], 64.0
    vs = np.linspace(5, 63, 2000)
    g = qvdfe.gamma_cubic(vs, vf, c)
    i = np.argmin(np.abs(g))
    assert close(qvdfe.gamma(vs[i], vf, c), g[i], rtol=1e-8, atol=1e-8 * abs(qvdfe.cubic_rate(vf, c)))


def test_time_distance_split_and_finite_link():
    L, vf, vq, w = 0.5, 64.0, 17.65, 0.01
    e = qvdfe.link_emission(L, vf, vq, w, CO2)
    assert close(e['T_free'] + e['T_queue'], L / vf + w)                      # time balance
    assert close(vf * e['T_free'] + vq * e['T_queue'], L)                      # distance balance
    allow = qvdfe.finite_link_allowance(L, vf, vq)
    assert qvdfe.finite_link_ok(allow, L, vf, vq) and not qvdfe.finite_link_ok(allow * 1.001, L, vf, vq)
    assert close(qvdfe.link_emission(L, vf, vq, allow, CO2)['T_free'], 0, atol=1e-12)


@pytest.mark.parametrize('p', qvdfe.POLLUTANTS)
def test_episode_total_equals_vmt_vht_form(p):
    c, D, L, vf, vq, wbar = RATES[p]['coef'], 2364.0, 0.5, 64.0, 17.65, 0.0139
    E1 = qvdfe.episode_total(D, wbar, L, vf, vq, c)
    E2 = qvdfe.vmt_vht_form(D * L, D * (L / vf + wbar), vf, vq, c)
    assert close(E1, E2, rtol=1e-12, atol=1e-9)


# ---------------- demand-dependent response ----------------------------------------------------
def test_fixed_service_increment_elasticity_equals_beta():
    card = dict(fd=1.0, n=1.0, fp=2.0, s=1.3, theta=0.5, C=1500, vf=64, L=0.5, kj=220)
    r = qvdfe.response(card, 1.5, CO2)
    assert close(r['eps_Gamma'], 0, atol=1e-12) and close(r['increment_elasticity'], r['beta'])


def test_increment_elasticity_matches_numerical_derivative():
    card = dict(fd=1.2, n=1.4, fp=3.0, s=1.1, theta=0.55, C=1500, vf=64, L=0.5, kj=220)
    x, h = 1.4, 1e-5
    f = lambda z: qvdfe.response(card, z, CO2)['increment_g']
    num = x * (f(x + h) - f(x - h)) / (2 * h) / f(x)
    assert close(qvdfe.response(card, x, CO2)['increment_elasticity'], num, rtol=1e-6)
    E = lambda D: D * qvdfe.response(card, D / 1500, CO2)['e_g']
    D0 = x * 1500
    assert close((E(D0 + 1e-3) - E(D0 - 1e-3)) / 2e-3, qvdfe.response(card, x, CO2)['marginal_g'], rtol=1e-6)


# ---------------- finite transitions -----------------------------------------------------------
def test_kernel_preserves_time_and_distance_and_9_over_140():
    xi = np.linspace(0, 1, 200001)
    h = qvdfe.kernel(xi)
    assert close(np.trapezoid(h, xi), 0.5, rtol=1e-9)                       # distance (v1+v2)T/2
    assert close(qvdfe.kernel_prime(xi).max(), 1.5, rtol=1e-9)
    for p in qvdfe.POLLUTANTS:
        c = RATES[p]['coef']
        v1, v2, T = 64.0, 17.65, 20.79 / 3600
        v = v1 + (v2 - v1) * h
        exact = T * np.trapezoid(qvdfe.cubic_rate(v, c), xi) - T / 2 * (qvdfe.cubic_rate(v1, c) + qvdfe.cubic_rate(v2, c))
        assert close(exact, qvdfe.speed_only_correction(T, v1, v2, c), rtol=1e-6, atol=1e-12)


def test_reference_transition_durations():
    assert close(qvdfe.transition_time_s(64.0 - 17.65, 1.5), 3 * (64 - 17.65) * 0.44704 / 3.0)


def test_band_endpoints_are_where_each_portion_equals_half_the_transitions():
    L, vf, vq, Td, Tu = 0.5, 64.0, 17.65, 20.79 / 3600, 31.19 / 3600
    lo, hi = qvdfe.transition_band(L, vf, vq, Td, Tu)
    S = Td + Tu
    assert close(qvdfe.link_emission(L, vf, vq, lo, CO2)['T_queue'], S / 2)
    assert close(qvdfe.link_emission(L, vf, vq, hi, CO2)['T_free'], S / 2)


def test_subset_identity_on_synthetic_cohort():
    n = qvdfe.NewellQueue(mu=1500, P=2, b=100)
    t = np.linspace(0, 2, 40001)
    L, vf, vq = 1.0, 60.0, 20.0
    Td = qvdfe.transition_time_s(vf - vq, 1.5) / 3600
    Tu = qvdfe.transition_time_s(vf - vq, 1.0) / 3600
    lo, hi = qvdfe.transition_band(L, vf, vq, Td, Tu)
    s = qvdfe.cohort_shares(t, n.lam(t), n.w(t), lo, hi)
    assert close(s['share_inside'] + s['share_below'] + s['share_above'], 1, rtol=1e-6)
    de = qvdfe.speed_only_correction(Td, vf, vq, CO2) + qvdfe.speed_only_correction(Tu, vq, vf, CO2)
    inside = (n.w(t) >= lo) & (n.w(t) <= hi)
    per_vehicle = np.array([qvdfe.link_emission(L, vf, vq, w, CO2)['e'] for w in n.w(t)]) + de
    direct = np.trapezoid(n.lam(t) * per_vehicle * inside, t)                # sum over admitted vehicles
    assert close(qvdfe.subset_total(s['D_A'], s['wbar_A'], L, vf, vq, CO2, de), direct, rtol=1e-9)
    # the full-episode mean delay is not a substitute for wbar_A
    assert not close(s['wbar_A'], n.mean_delay, rtol=1e-3)


# ---------------- QVDF, guards and helpers -----------------------------------------------------------------
def test_qvdf_branches():
    P, sev, wbar_factor = qvdfe.qvdf(np.array([0.5, 2.0]), fd=1.2, n=1.1, fp=0.9, s=1.0, theta=0.5)
    assert close(P[1], 1.2 * 2.0 ** 1.1) and close(P[0], max(1.2 * 0.5 ** 1.1, 0.5))
    assert np.allclose(wbar_factor, 0.5 * sev)


def test_fixed_service_anchored_at_same_state_has_same_level_and_beta_elasticity():
    card = dict(fd=1.2373, n=1.0754, fp=0.9631, s=0.9517, theta=0.4581, C=1501.6, vf=64.16, L=1.04, kj=220)
    x0 = 1.574
    rc = qvdfe.response(card, x0, CO2)
    fixed = {**card, 'n': 1.0, 'fd': rc['P'] / x0}            # same mu, vq, Gamma and delay at x0
    rf = qvdfe.response(fixed, x0, CO2)
    for k in ('mu', 'vq', 'Gamma', 'mean_delay', 'e_g'):
        assert close(rc[k], rf[k], rtol=1e-12)
    assert close(rf['increment_elasticity'], rf['beta'], atol=1e-12)


@pytest.mark.parametrize('call', [
    lambda: qvdfe.queue_speed(2100, 2000, 60, 200),
    lambda: qvdfe.finite_link_allowance(1.0, 60, 0.0),
    lambda: qvdfe.finite_link_allowance(1.0, 60, 60.0),
    lambda: qvdfe.transition_band(1.0, 60, 0.0, 0.01, 0.01),
    lambda: qvdfe.delay_profile(2.0, 0.01, 1.0, 1500),
    lambda: qvdfe.cohort_shares([0, 1], [0, 0], [0, 0], 0.0, 1.0),
    lambda: qvdfe.closed_queue(np.linspace(0, 1, 11), np.full(11, 1000.0), 1500),
])
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_gamma_limit_at_free_flow_is_finite_for_cubic():
    g = qvdfe.gamma(64.0, 64.0, CO2)
    assert np.isfinite(g) and close(g, qvdfe.gamma_cubic(63.999999, 64.0, CO2), rtol=1e-6)


def test_data_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv('QVDFE_DATA', str(tmp_path))
    assert qvdfe.data_dir() == tmp_path.resolve()
    with pytest.raises(FileNotFoundError):
        qvdfe.frozen_dir()
    monkeypatch.delenv('QVDFE_DATA')
    assert (qvdfe.repo_root() / 'src' / 'qvdfe').exists()


def test_report_compare_uses_absolute_and_relative_tolerance():
    rows = qvdfe.report.compare({'a': 1e-12, 'b': {'c': 1.0000001}, 'd': True}, {'a': 0.0, 'b': {'c': 1.0}, 'd': True})
    assert all(r[3] for r in rows) and len(rows) == 3
    assert not qvdfe.report.compare({'a': 1.01}, {'a': 1.0})[0][3]
