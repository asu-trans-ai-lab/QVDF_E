"""Two-speed emission accounting (paper Section 3, Eqs. 6-12 and Eq. 27).

Rates r(v) are zero-acceleration (speed-only) rates in g/(veh h) at speed v in mph. ``rate`` may be the cubic
coefficients (c0, c1, c2, c3) or any callable v -> r(v). Speed changes are instantaneous in this section; Section 5
(``qvdfe.transitions``) adds finite transitions. Functions that return dicts accept scalars only.
"""
import numpy as np


def cubic_rate(v, coef):
    """r(v) = c0 + c1 v + c2 v^2 + c3 v^3, g/(veh h). Same as two_regime_emissions.cubic_rate. Accepts arrays."""
    c0, c1, c2, c3 = np.asarray(coef, float)
    v = np.asarray(v, float)
    return c0 + c1 * v + c2 * v ** 2 + c3 * v ** 3


def cubic_rate_second_derivative(v, coef):
    """r''(v) = 2 c2 + 6 c3 v, g/(veh h mph^2). Accepts arrays."""
    _, _, c2, c3 = np.asarray(coef, float)
    return 2 * c2 + 6 * c3 * np.asarray(v, float)


def _is_coef(rate):
    return not callable(rate)


def _as_rate(rate):
    return rate if callable(rate) else (lambda v: cubic_rate(v, rate))


def _check_speeds(vf, vq):
    vq_arr = np.asarray(vq, float)
    if np.any(vq_arr <= 0) or np.any(vq_arr >= vf):
        raise ValueError(f'the two-speed state needs 0 < v_q < v_f (got v_q={vq}, v_f={vf})')


def gamma_cubic(vq, vf, coef):
    """Closed form of Gamma for the cubic rate: c0 - (c2 vf + c3 vf^2) vq - c3 vf vq^2 (two_regime_emissions.gamma_from_cubic).

    Exact for every vq, including the limit vq -> vf. Accepts arrays."""
    c0, _, c2, c3 = np.asarray(coef, float)
    vq = np.asarray(vq, float)
    return c0 - (c2 * vf + c3 * vf ** 2) * vq - c3 * vf * vq ** 2


def gamma(vq, vf, rate):
    """Delay-to-emission conversion factor (Eq. 8), g/(veh h).

    Gamma(vq) = [vf r(vq) - vq r(vf)] / (vf - vq) = [f(vq) - f(vf)] / (1/vq - 1/vf), with f(v) = r(v)/v the emission
    per mile. Gamma < 0 when a mile in the queue emits less than a mile at vf (for CO2 with the archived rates this
    happens for vq above about 27 mph when vf = 60 mph). For cubic coefficients the closed form ``gamma_cubic`` is used,
    which avoids cancellation near vq = vf. For a callable rate, 0 < vq < vf is required. Accepts arrays.
    """
    if _is_coef(rate):
        return gamma_cubic(vq, vf, rate)
    _check_speeds(vf, vq)
    vq = np.asarray(vq, float)
    return (vf * rate(vq) - vq * rate(vf)) / (vf - vq)


def link_emission(L, vf, vq, w, rate):
    """Per-vehicle emission on a link of length L (mi) with delay w (h) in the two-speed state (Eq. 8). Scalars only.

    Returns dict: e0 = r(vf) L/vf (g), Gamma (g/(veh h)), e = e0 + Gamma w (g), and the time split T_free, T_queue (h)
    from Eqs. (6)-(7). T_free < 0 means the delay violates the finite-link condition (Eq. 9).
    """
    _check_speeds(vf, vq)
    r = _as_rate(rate)
    Tf = L / vf
    G = float(gamma(vq, vf, rate))
    T_queue = vf / (vf - vq) * w
    return {'e0': float(r(vf) * Tf), 'Gamma': G, 'e': float(r(vf) * Tf + G * w), 'Tf': Tf,
            'T_free': Tf + w - T_queue, 'T_queue': T_queue}


def episode_total(D, wbar, L, vf, vq, rate):
    """Episode emission total E_P = D [e0 + Gamma(vq) wbar] (Eq. 10), g per lane. D veh/lane, wbar h.

    Conditions: one link, closed episode, a common two-speed state (vf, vq) for all vehicles, common speed-only rates,
    and every vehicle's delay within the finite-link allowance (Eq. 9)."""
    e = link_emission(L, vf, vq, wbar, rate)
    return D * (e['e0'] + e['Gamma'] * wbar)


def vmt_vht_form(VMT, VHT, vf, vq, rate):
    """Equivalent form (Eq. 12): E_P = [r(vf)/vf] VMT + Gamma(vq) (VHT - VMT/vf). VMT veh mi, VHT veh h. Same conditions as Eq. (10)."""
    _check_speeds(vf, vq)
    r = _as_rate(rate)
    return float(r(vf) / vf * VMT + gamma(vq, vf, rate) * (VHT - VMT / vf))


def finite_link_allowance(L, vf, vq):
    """Largest delay (h) the link can carry in the two-speed state: L (1/vq - 1/vf) (Eq. 9)."""
    _check_speeds(vf, vq)
    return L * (1.0 / np.asarray(vq, float) - 1.0 / vf)


def finite_link_ok(w, L, vf, vq, tol=1e-10):
    """Finite-link condition T_free >= 0, i.e. w <= L (1/vq - 1/vf) (Eq. 9)."""
    return np.asarray(w, float) <= finite_link_allowance(L, vf, vq) + tol


def subset_total(D_A, wbar_A, L, vf, vq, rate, de_tr):
    """Admitted-subset speed-only emissions E_A = D_A [e0 + Gamma wbar_A + de_tr] (Eq. 27), g per lane.

    D_A is the volume of the vehicles whose delay lies inside the transition band (Eq. 26) and wbar_A their own
    vehicle-weighted mean delay (h); de_tr is the speed-only transition correction per vehicle (g). Not the full-episode
    total and contains no acceleration-dependent (operating-mode) term. With D_A = D (the whole cohort inside the band)
    it is the full-cohort total of Eq. (25).
    """
    e = link_emission(L, vf, vq, wbar_A, rate)
    return D_A * (e['e0'] + e['Gamma'] * wbar_A + de_tr)
