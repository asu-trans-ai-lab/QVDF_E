"""Finite transitions between free-flow and queued motion (paper Section 5, Eqs. 21-27)."""
import numpy as np

from .emission import cubic_rate_second_derivative

MPH_TO_MS = 0.44704


def kernel(xi):
    """Symmetric transition kernel h(xi) = 3 xi^2 - 2 xi^3 on [0, 1] (Eq. 22)."""
    xi = np.asarray(xi, float)
    return 3 * xi ** 2 - 2 * xi ** 3


def kernel_prime(xi):
    """h'(xi) = 6 xi (1 - xi); its maximum 3/2 sets the peak acceleration."""
    xi = np.asarray(xi, float)
    return 6 * xi * (1 - xi)


def transition_time_s(dv_mph, a_peak_ms2):
    """Duration (s) of a kernel transition with speed change dv (mph) and peak magnitude a (m/s^2): T = 3 dv / (2 a)."""
    return 3 * dv_mph * MPH_TO_MS / (2 * a_peak_ms2)


def speed_only_correction(T_h, v1, v2, coef):
    """Speed-only emission change (g/veh) of one kernel transition of duration T_h (h) between v1 and v2 (mph).

    For a cubic rate: delta_e = -(9/140) T (v2 - v1)^2 r''((v1 + v2)/2) (Eq. 23). The operating-mode
    (acceleration-dependent) part is not included; it needs acceleration-dependent rates.
    """
    dv = v2 - v1
    return float(-(9.0 / 140.0) * T_h * dv ** 2 * cubic_rate_second_derivative(0.5 * (v1 + v2), coef))


def transition_band(L, vf, vq, T_down_h, T_up_h):
    """Delays (h) for which both transitions fit inside the link (Eq. 26).

    w_lo = (vf - vq)(Td + Tu) / (2 vf);  w_hi = L (1/vq - 1/vf) - (vf - vq)(Td + Tu) / (2 vq).
    The band is empty when w_hi < w_lo.
    """
    if not 0 < vq < vf:
        raise ValueError('need 0 < v_q < v_f')
    Th = T_down_h + T_up_h
    w_lo = (vf - vq) * Th / (2 * vf)
    w_hi = L * (1 / vq - 1 / vf) - (vf - vq) * Th / (2 * vq)
    return w_lo, w_hi


def cohort_shares(t, lam, w, w_lo, w_hi):
    """Split a cohort by the band of Eq. (26).

    t (h), lam arrival rate (veh/h/lane), w delay (h) on the same grid. Returns the volume D, the shares inside,
    below and above the band, the admitted volume D_A and the admitted vehicles' own vehicle-weighted mean delay
    wbar_A (h). wbar_A, not the full-episode mean delay, enters the admitted-subset total of Eq. (27).
    """
    t, lam, w = (np.asarray(z, float) for z in (t, lam, w))
    D = float(np.trapezoid(lam, t))
    if D <= 0:
        raise ValueError('the cohort has no volume')
    empty = w_hi < w_lo
    inside = (w >= w_lo) & (w <= w_hi) & (not empty)
    below = (w < w_lo) & (not empty)
    above = (w > w_hi) & (not empty)
    D_A = float(np.trapezoid(lam * inside, t))
    wbar_A = float(np.trapezoid(lam * w * inside, t) / D_A) if D_A > 0 else float('nan')
    return {'D': D, 'band_empty': bool(empty), 'share_inside': D_A / D,
            'share_below': float(np.trapezoid(lam * below, t) / D), 'share_above': float(np.trapezoid(lam * above, t) / D),
            'D_A': D_A, 'wbar_A': wbar_A}
