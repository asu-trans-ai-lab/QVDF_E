"""The closed fluid queue and the reconstructed delay profile (paper Section 2.2, Eqs. 3-4)."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln


def closed_queue(t, lam, mu):
    """Integrate a closed queue with constant service rate mu.

    t time grid (h) from onset to clearance, lam arrival rate on that grid (veh/h/lane), mu service (veh/h/lane).
    Returns a dict with the queue Q(t) (veh/lane), delay w(t) = Q/mu (h), the counted volume D (veh/lane),
    the total delay W_P (veh h/lane), the time-average delay (h), the vehicle-weighted mean delay (h) and the
    closure error |Q(end)| (veh/lane). For a closed queue D = mu*P and W_P = integral of Q = D * mean delay.
    The point queue cannot be negative: arrivals below service at onset raise ValueError.
    """
    t = np.asarray(t, float)
    lam = np.asarray(lam, float)
    if mu <= 0 or len(t) < 2:
        raise ValueError('need mu > 0 and at least two time points')
    inflow = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(t))])
    Q = inflow - mu * (t - t[0])
    if Q.min() < -1e-6 * max(1.0, mu * (t[-1] - t[0])):   # tolerance covers quadrature error
        raise ValueError('the cumulative arrivals fall below the departures: not a point queue starting empty at t[0]')
    w = Q / mu
    D = float(inflow[-1])
    W = float(np.trapezoid(Q, t))
    P = float(t[-1] - t[0])
    return {'Q': Q, 'w': w, 'D': D, 'W_P': W, 'P': P, 'mean_delay_time': float(np.trapezoid(w, t) / P),
            'mean_delay_vehicle': float(np.trapezoid(lam * w, t) / D), 'closure_error': float(abs(Q[-1]))}


@dataclass(frozen=True)
class NewellQueue:
    """Newell's quadratic-inflow closed queue Q(t) = (b/3) t^2 (P - t) (Fig. 4 of the paper).

    mu service rate (veh/h/lane), P duration (h), b amplitude (veh/h^3/lane). The ratio of mean to peak delay
    is 9/16 for this model only; other closed profiles give other ratios.
    """
    mu: float = 1800.0
    P: float = 2.0
    b: float = 675.0

    def Q(self, t):
        t = np.asarray(t, float)
        return self.b / 3 * t * t * (self.P - t)

    def lam(self, t):
        t = np.asarray(t, float)
        return self.mu + self.b / 3 * (2 * self.P * t - 3 * t * t)

    def w(self, t):
        return self.Q(t) / self.mu

    @property
    def t2(self):
        return 2 * self.P / 3

    @property
    def peak_delay(self):
        return 4 * self.b * self.P ** 3 / (81 * self.mu)

    @property
    def mean_delay(self):
        return self.b * self.P ** 3 / (36 * self.mu)

    @property
    def D(self):
        return self.mu * self.P

    @property
    def W(self):
        return self.b * self.P ** 4 / 36


@lru_cache(maxsize=512)
def profile_shape(theta):
    """Exponent a of the symmetric profile h(u) = [4u(1-u)]^a whose integral over [0, 1] equals theta.

    Same root as v15_model.profile_shape: 4^a B(a+1, a+1) = theta. Defined for 0 < theta < 1.
    """
    if not 0 < theta < 1:
        return float('nan')
    return brentq(lambda a: np.exp(a * np.log(4) + betaln(a + 1, a + 1)) - theta, 1e-5, 1e4)


def delay_profile(P_hat, wt2, theta, mu, a=None, n_grid=20001):
    """Reconstructed closed delay profile of a fitted episode (the paper's reconstruction rule).

    P_hat duration (h), wt2 peak delay (h), theta = mean/peak delay, mu service rate (veh/h/lane).
    theta <= 2/3: h(u) = [4u(1-u)]^a with a = profile_shape(theta) unless given; otherwise h = 1 - |2u-1|^b with
    b = theta/(1-theta). The derivative is taken in actual time, w'(t) = (wt2/P_hat) h'(u), and the arrival rate is
    lam(t) = mu [1 + w'(t)]. Returns dict with t (h), w (h), lam (veh/h/lane), D, lambda_min, mean delays.
    A negative lambda_min means the reconstructed profile is not physically feasible.
    """
    if not 0 < theta < 1 or P_hat <= 0 or wt2 < 0:
        raise ValueError('need 0 < theta < 1, P_hat > 0 and wt2 >= 0')
    u = np.linspace(0.0, 1.0, n_grid)
    if theta <= 2.0 / 3.0:
        a = profile_shape(theta) if a is None else float(a)
        base = 4 * u * (1 - u)
        h = base ** a
        with np.errstate(divide='ignore', invalid='ignore'):
            dh = np.where(base > 0, a * base ** (a - 1) * 4 * (1 - 2 * u), 0.0)
        family = 'power'
    else:
        b = theta / (1 - theta)
        s = np.abs(2 * u - 1)
        h = 1 - s ** b
        with np.errstate(divide='ignore', invalid='ignore'):
            dh = np.where(s > 0, -b * s ** (b - 1) * np.sign(2 * u - 1) * 2, 0.0)
        a, family = b, 'plateau'
    t = u * P_hat
    w = wt2 * h
    lam = mu * (1 + (wt2 / P_hat) * dh)
    D = float(np.trapezoid(lam, t))
    return {'t': t, 'u': u, 'w': w, 'lam': lam, 'D': D, 'shape': a, 'family': family,
            'lambda_min': float(lam.min()), 'mean_delay_time': float(np.trapezoid(w, t) / P_hat),
            'mean_delay_vehicle': float(np.trapezoid(lam * w, t) / D)}
