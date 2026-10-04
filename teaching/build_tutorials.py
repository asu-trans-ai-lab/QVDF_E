"""Generate and execute the four tutorials and the instructor solutions.

Run from anywhere:  python teaching/build_tutorials.py [--no-execute]
Every tutorial follows one template: question -> assumptions -> derivation -> worked example -> figure ->
interpretation question. Each imports the shared core package ``qvdfe``; no formula is re-implemented here.
"""
from pathlib import Path
import argparse
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
TUT = HERE / 'tutorials'
SOL = HERE / 'instructor'
TUT.mkdir(exist_ok=True)
SOL.mkdir(exist_ok=True)

PRELUDE = r'''import sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'qvdfe').exists())
sys.path.insert(0, str(ROOT / 'src'))          # works with or without `pip install -e .`
import json
import numpy as np
import matplotlib.pyplot as plt
import qvdfe
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})

def check(name, ok):
    print(('PASS  ' if ok else 'FAIL  ') + name)
    assert ok, name
RATES = qvdfe.load_rates()'''


def md(s):
    return new_markdown_cell(s.strip())


def code(s):
    return new_code_cell(s.strip())


def header(title, prereq, units, outputs, conditions, paper):
    return md(f'''
# {title}

| | |
| --- | --- |
| **Paper** | {paper} |
| **Prerequisites** | {prereq} |
| **Input units** | {units} |
| **Expected outputs** | {outputs} |
| **Conditions** | {conditions} |

Template: question, assumptions, derivation, worked example, figure, interpretation question.
''')


def save(cells, path):
    n = new_notebook()
    n.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    n.metadata['language_info'] = {'name': 'python'}
    n.cells = cells
    path.write_text(nbformat.writes(n), encoding='utf-8')
    return path


T = {}

# ================================================================================================
T['T1_bottleneck_closed_episode'] = [
    header('T1 · Bottleneck and the closed episode',
           'Cumulative arrival and departure curves; integrals.',
           'time h, flow veh/h/lane, delay h (printed in min).',
           'Oblique cumulative curves; D = μP; W_P = ∫Q dt = D·w̄; the 9/16 ratio for the Newell model and a different ratio for another closed profile.',
           'Point queue with a constant service rate μ; the queue is closed (Q = 0 at onset and clearance); first in, first out.',
           'Section 2.2, Eqs. (3)–(4); Fig. 4.'),
    md(r'''
## 1. Question
How do the inflow, the service rate, the queue and the delay connect over one bottleneck episode, and which summary of the delay profile is enough for totals?

## 2. Assumptions
A bottleneck serves at a constant rate $\mu$ from onset $t_0$ to clearance $t_3$; arrivals $\lambda(t)$ exceed $\mu$ early and fall below it later; the queue is empty at $t_0$ and $t_3$; vehicles leave in arrival order.

## 3. Derivation
Queue balance $\dot Q=\lambda-\mu$ and FIFO give the delay $w(t)=Q(t)/\mu$. Integrating over the episode with $Q(t_0)=Q(t_3)=0$:
$$D=\int\lambda\,dt=\mu P,\qquad W_P=\int\lambda w\,dt=\int Q\,dt=D\,\bar w_P,\qquad \bar w_P=\frac1P\int w\,dt.$$
The middle identity uses $\lambda w=\mu w+\tfrac{\mu}{2}\tfrac{d}{dt}w^2$, whose last term integrates to zero on a closed queue. So the time-average delay equals the vehicle-average delay, which is what emission totals need.

Newell's quadratic inflow, $Q=\tfrac b3(t-t_0)^2(t_3-t)$, gives a peak at $t_2=t_0+2P/3$ and $\bar w_P/w_{t_2}=9/16$. **The 9/16 ratio belongs to this model only**; another closed profile gives another ratio.
'''),
    code(PRELUDE),
    md('## 4. Worked example\nThe inputs are those of `examples/01_synthetic_episode/config.json` (synthetic values).'),
    code(r'''
cfg = json.loads((ROOT / 'examples/01_synthetic_episode/config.json').read_text())
mu, P, b = cfg['mu_vphpl'], cfg['P_h'], cfg['newell_b_veh_per_h3']
n = qvdfe.NewellQueue(mu=mu, P=P, b=b)
t = np.linspace(0, P, 4001)
q = qvdfe.closed_queue(t, n.lam(t), mu)
print(f"D = {q['D']:.3f} veh/lane (mu P = {mu*P:.0f});  W_P = {q['W_P']:.4f} veh h/lane;  D * mean delay = {q['D']*q['mean_delay_vehicle']:.4f}")
print(f"time-average delay {q['mean_delay_time']*60:.4f} min, vehicle-average delay {q['mean_delay_vehicle']*60:.4f} min")
check('vehicle conservation D = mu P', np.isclose(q['D'], mu * P, rtol=1e-8))
check('closed queue Q(t3) = 0', q['closure_error'] < 1e-6 * mu * P)
check('W_P = integral of Q = D * mean delay', np.isclose(q['W_P'], q['D'] * q['mean_delay_vehicle'], rtol=1e-4))
check('Newell model: mean/peak delay = 9/16', np.isclose(n.mean_delay / n.peak_delay, 9 / 16))
# a second closed profile with the same mu and P: sinusoidal surplus
lam2 = mu + 120 * np.sin(2 * np.pi * t / P)
q2 = qvdfe.closed_queue(t, lam2, mu)
print(f"sinusoidal profile: mean/peak = {q2['mean_delay_time'] / q2['w'].max():.4f}  (not 9/16)")
check('second profile is closed too', q2['closure_error'] < 1e-6 * mu * P)
'''),
    md('## 5. Figure'),
    code(r'''
ref = 0.9 * mu
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for lam_, Q_, c, lab in ((n.lam(t), q['Q'], '#2a78d6', 'Newell'), (lam2, q2['Q'], '#8C1D40', 'sinusoidal')):
    ax[0].plot(t, mu * t + Q_ - ref * t, color=c, label=f'arrivals ({lab})')
ax[0].plot(t, mu * t - ref * t, color='k', label='departures')
ax[0].set(xlabel='time from onset (h)', ylabel=f'N(t) - {ref:.0f} t (veh/lane)', title='Oblique cumulative curves'); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(t, q['w'] * 60, color='#2a78d6', label='Newell'); ax[1].plot(t, q2['w'] * 60, color='#8C1D40', label='sinusoidal')
ax[1].set(xlabel='time from onset (h)', ylabel='delay (min)', title='Delay profiles'); ax[1].legend(frameon=False)
plt.tight_layout()
'''),
    md(r'''
## 6. Interpretation question
Both profiles serve the same number of vehicles in the same duration. Which one has the larger total delay $W_P$, and why does $\bar w_P$ (not the peak delay) carry that information into an emission total?
'''),
]

# ================================================================================================
T['T2_two_speeds_and_emissions'] = [
    header('T2 · Two speeds and emissions',
           'T1; emission rate as a function of speed.',
           'speed mph, length mi, time h, rate g/(veh h), emissions g.',
           'The time–distance split; Eq. (10) equal to the VMT/VHT form; three activity patterns with equal VMT and VHT but different emissions; the finite-link condition.',
           'One link; a common two-speed state (v_f, v_q); common zero-acceleration rates; delay within the finite-link allowance.',
           'Section 3, Eqs. (6)–(12); Table F1.'),
    md(r'''
## 1. Question
Vehicle-miles (VMT) and vehicle-hours (VHT) give how far and how long. Why are they not enough for emissions, and what extra state information does a bottleneck episode supply?

## 2. Assumptions
A traversal of length $L$ consists of a free-flow portion at $v_f$ and a queued portion at $v_q$; the rate $r(v)$ is a zero-acceleration rate (speed-only); transitions between the two speeds are instantaneous here (T4 relaxes this).

## 3. Derivation
Time and distance balances, Eqs. (6)–(7): $T_{\rm free}+T_{\rm queue}=T_f+w$ and $v_fT_{\rm free}+v_qT_{\rm queue}=L$, so $T_{\rm queue}=\frac{v_f}{v_f-v_q}w$. Then
$$e=e_0+\Gamma(v_q)\,w,\qquad e_0=r(v_f)\frac{L}{v_f},\qquad \Gamma(v_q)=\frac{v_fr(v_q)-v_qr(v_f)}{v_f-v_q}.$$
$\Gamma$ (g per vehicle-hour of delay) can be **negative**: it is the change in emission per mile between the two speeds divided by the change in travel time per mile, so $\Gamma<0$ when a mile in the queue emits less than a mile at $v_f$. With the archived CO$_2$ rates and $v_f=60$ mph this happens for $v_q$ above about 27 mph, and for most of the paper's admitted episodes (computed below).
Summed over a closed episode (T1): $E_P=D[e_0+\Gamma\bar w_P]$ (Eq. 10). With ${\rm VMT}=DL$, ${\rm VHT}=D(T_f+\bar w_P)$:
$$E_P=\frac{r(v_f)}{v_f}{\rm VMT}+\Gamma(v_q)\Big({\rm VHT}-\frac{\rm VMT}{v_f}\Big)\quad\text{(Eq. 12)}.$$
The free-flow portion must be nonnegative: $w\le L(1/v_q-1/v_f)$ (Eq. 9, the finite-link condition).
'''),
    code(PRELUDE),
    md('## 4. Worked example\nFirst the time–distance split and the two equivalent forms, then three activity patterns with identical VMT and VHT.'),
    code(r'''
coef = RATES['CO2']['coef']
L, vf, vq, w = 1.0, 60.0, 20.0, 1 / 60          # mi, mph, mph, h (one minute of delay)
e = qvdfe.link_emission(L, vf, vq, w, coef)
print(f"T_free = {e['T_free']*60:.3f} min, T_queue = {e['T_queue']*60:.3f} min; e0 = {e['e0']:.2f} g, Gamma = {e['Gamma']:.2f} g/(veh h), e = {e['e']:.2f} g")
check('time balance', np.isclose(e['T_free'] + e['T_queue'], L / vf + w))
check('distance balance', np.isclose(vf * e['T_free'] + vq * e['T_queue'], L))
check('finite-link condition holds', bool(qvdfe.finite_link_ok(w, L, vf, vq)))
D = 3000
E10 = qvdfe.episode_total(D, w, L, vf, vq, coef)
E12 = qvdfe.vmt_vht_form(D * L, D * (L / vf + w), vf, vq, coef)
print(f'Eq. (10): {E10/1000:.4f} kg;  VMT/VHT form: {E12/1000:.4f} kg')
check('Eq. (10) equals the VMT/VHT form', np.isclose(E10, E12, rtol=1e-12))
'''),
    md('**Table F1 of the paper.** The same function at the detector-78 reference state of Appendix F. The paper rounds each column separately, so the displayed components need not add to the displayed total.'),
    code(r'''
from qvdfe.data import load_reference_state
ref = load_reference_state()
shown = {'CO2': (294.6, 31.7, 326.2, 1), 'NOx': (0.4138, -0.0658, 0.3480, 4), 'CO': (1.9527, 0.6141, 2.5669, 4), 'HC': (0.0819, 0.0263, 0.1082, 4)}
for p, (s0, s1, st, d) in shown.items():
    ef = qvdfe.link_emission(ref['L_mi'], ref['vf_mph'], ref['vq_mph'], ref['mean_delay_h'], RATES[p]['coef'])
    inc = ef['Gamma'] * ref['mean_delay_h']
    print(f"{p:4s} e0 = {ef['e0']:.6g}, Gamma*wbar = {inc:.6g}, total = {ef['e']:.6g} g/veh  -> rounded {round(ef['e0'], d)}, {round(inc, d)}, {round(ef['e'], d)} (paper {s0}, {s1}, {st})")
    check(f'{p}: Table F1 reproduced', round(ef['e0'], d) == s0 and round(inc, d) == s1 and round(ef['e'], d) == st)
E = qvdfe.load_episodes(); A = E[E.matched_ladder_pass]
print(f"admitted episodes with Gamma_CO2 < 0: {int((A.Gamma_cubic_CO2 < 0).sum())} of {len(A)}; Gamma_NOx < 0: {int((A.Gamma_cubic_NOx < 0).sum())} of {len(A)}")
'''),
    code(r'''
# Three traversals with the same distance (1 mi) and the same time (2 min): equal VMT and VHT per vehicle.
Ttot = L / vf + w
tt = np.linspace(0, Ttot, 20001)
patterns = {
    'steady at the average speed': np.full_like(tt, L / Ttot),
    'free flow then queue (two-speed)': np.where(tt < e['T_free'], vf, vq),
    'oscillating around the average': L / Ttot + 20 * np.sin(2 * np.pi * 4 * tt / Ttot),
}
rows = []
for name, v in patterns.items():
    dist, em = np.trapezoid(v, tt), np.trapezoid(qvdfe.cubic_rate(v, coef), tt)
    rows.append((name, dist, Ttot * 60, em))
    print(f'{name:34s} distance {dist:.4f} mi, time {Ttot*60:.2f} min, CO2 {em:7.2f} g  (speed range {v.min():.0f}-{v.max():.0f} mph)')
check('all three have the same distance', np.allclose([r[1] for r in rows], L, rtol=1e-4))
print('Speed-only rates are used for all three; the oscillating pattern would also carry acceleration effects that this rate omits.')
'''),
    md('## 5. Figure'),
    code(r'''
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for (name, v), c in zip(patterns.items(), ('grey', '#8C1D40', '#2a78d6')):
    ax[0].plot(tt * 60, v, color=c, label=name)
ax[0].set(xlabel='time on link (min)', ylabel='speed (mph)', title='Same VMT and VHT'); ax[0].legend(frameon=False, fontsize=8)
ax[1].bar(range(3), [r[3] for r in rows], color=['grey', '#8C1D40', '#2a78d6'])
ax[1].set_xticks(range(3), ['steady', 'two-speed', 'oscillating']); ax[1].set(ylabel='CO2 per vehicle (g)', title='Different emissions')
plt.tight_layout()
'''),
    md(r'''
## 6. Interpretation question
The three traversals have the same VMT and VHT. Which property of $r(v)$ makes their emissions differ, and which quantity in Eq. (12) carries the bottleneck's state information that VMT and VHT lack?
'''),
]

# ================================================================================================
T['T3_demand_dependent_response'] = [
    header('T3 · Demand-dependent response',
           'T1, T2; elasticities.',
           'loading x = D/C in hours (episode volume per lane divided by capacity per lane: the time the bottleneck would need to serve the episode volume at capacity); capacity veh/h/lane; speeds mph; emissions g/veh.',
           'Γ(x) and the increment elasticity for one calibrated card, compared with fixed service anchored at the same state.',
           'Admissible power branch of the calibrated QVDF; finite-link condition satisfied; Γ ≠ 0 where an elasticity is reported.',
           'Section 4, Eqs. (13)–(20); Fig. 12.'),
    md(r'''
## 1. Question
When demand grows, the episode lengthens and the delay grows. Does the emission increment grow at the same rate as the delay?

## 2. Assumptions
The calibrated queueing-based volume-delay function (QVDF) of each corridor and period:

| Symbol (card column) | Meaning |
| --- | --- |
| $x=D/C$ (`x_h`) | loading in hours (see header) |
| $f_d$, $n$ (`fd`, `n`) | duration $P(x)=\max\{f_dx^n,x\}$ (h); the power branch is $f_dx^n>x$ |
| $f_p$, $s$ (`fp`, `s`) | peak delay $w_{t_2}=T_ff_pP^s$, with $T_f=L/v_f$ the free-flow travel time (h) |
| $\theta$ (`theta`) | mean-to-peak delay ratio, $\bar w_P=\theta\,w_{t_2}$ |
| $\alpha=\theta f_pf_d^s$, $\beta=ns$ (`alpha`, `beta`) | so that $\bar w_P=T_f\alpha x^\beta$ on the power branch; $\beta$ is the delay elasticity |
| $C$, $v_f$, $L$, $k_j$ | capacity, free-flow speed, link length of the card's detector; jam density 220 veh/mi/lane (the paper's value) |

The service rate is $\mu(x)=Cx/P(x)$ and the queue speed follows it on the triangular FD (Section 2.1, Eq. 2). Capacity and the calibration are held fixed. A state is **admissible** when it is on the power branch, $5\le v_q<v_f\le75$ mph (the range of the rate table) and the peak delay satisfies the finite-link condition (Eq. 9); `qvdfe.response` reports this as `valid`.

## 3. Derivation
The emission increment per vehicle is $\Delta e(x)=\bar w_P(x)\,\Gamma[v_q(x)]$. Its loading elasticity is
$$\frac{x\,\Delta e'(x)}{\Delta e(x)}=\beta+\varepsilon_{\Gamma,x},\qquad \varepsilon_{\Gamma,x}=\frac{x\,\Gamma'(x)}{\Gamma(x)}\quad(\Gamma\neq0).$$
**Result.** The elasticity of the emission increment equals the delay elasticity $\beta$ when $\Gamma$ is locally constant, including fixed service ($\mu=\mu_0$, i.e. $n=1$ with $f_d=C/\mu_0$, so $v_q$ does not change with $x$); otherwise it includes the additional state-dependent term $\varepsilon_{\Gamma,x}$. It is defined only where $\Gamma\neq0$.

*Note.* The total per-vehicle emission $e_0+\Delta e$ includes the free-flow baseline $e_0$, which does not grow with $x$; its elasticity is therefore different, and the result above does not apply to it directly.
'''),
    code(PRELUDE),
    md('## 4. Worked example\nThe Arizona I-10 midday card is a corridor-wide calibration (`calibration_scope = corridor_all_periods`, so AM, MD and NT share it), evaluated with the detector-78 fundamental diagram. We use the loading of the Appendix F reference state, $x$ = 1.574 h, which is admissible; the card\'s median training loading (3.50 h) violates the finite-link condition for this short link. For comparison, fixed service is anchored at the same state: $n=1$ and $f_d=P(x_0)/x_0$, so that $\\mu_0$, $v_q$, $\\Gamma$ and the delay equal the calibrated values at $x_0$ and only their change with loading differs.'),
    code(r'''
cards = qvdfe.load_cards()
card = cards[(cards.corridor == 'AZ_I10_W') & (cards.period == 'MD')].iloc[0].to_dict()
print({k: round(card[k], 4) for k in ('fd', 'n', 'fp', 's', 'theta', 'beta', 'C', 'vf', 'L')})
coef = RATES['CO2']['coef']
from qvdfe.data import load_reference_state
ref = load_reference_state()
x0 = ref['x_h']
r0 = qvdfe.response(card, x0, coef)
check(f"reference state reproduced: P = {r0['P']:.4f} h, v_q = {r0['vq']:.2f} mph", np.isclose(r0['P'], ref['P_h'], rtol=1e-6) and np.isclose(r0['vq'], ref['vq_mph'], rtol=1e-6))
fixed = {**card, 'n': 1.0, 'fd': r0['P'] / x0}       # fixed service mu0 = mu(x0): same state at x0
for name, c in (('calibrated n', card), ('fixed service n = 1', fixed)):
    r = qvdfe.response(c, x0, coef)
    print(f"{name:20s} valid={r['valid']}  v_q={r['vq']:.2f} mph  Gamma={r['Gamma']:.1f} g/(veh h)  beta={r['beta']:.3f}  "
          f"eps_Gamma={r['eps_Gamma']:+.3f}  increment elasticity={r['increment_elasticity']:.3f}")
rf = qvdfe.response(fixed, x0, coef)
check('both states are admissible at this loading', r0['valid'] and rf['valid'])
check('same state at x0 (mu, v_q, Gamma, mean delay)', all(np.isclose(r0[k], rf[k], rtol=1e-12) for k in ('mu', 'vq', 'Gamma', 'mean_delay')))
check('fixed service: increment elasticity equals beta', np.isclose(rf['increment_elasticity'], rf['beta'], atol=1e-12))
rc = qvdfe.response(card, x0, coef)
h = 1e-5
num = x0 * (qvdfe.response(card, x0 + h, coef)['increment_g'] - qvdfe.response(card, x0 - h, coef)['increment_g']) / (2 * h) / rc['increment_g']
check('calibrated: beta + eps_Gamma equals the numerical elasticity of the increment', np.isclose(rc['increment_elasticity'], num, rtol=1e-6))
tot = x0 * (qvdfe.response(card, x0 + h, coef)['e_g'] - qvdfe.response(card, x0 - h, coef)['e_g']) / (2 * h) / rc['e_g']
print(f'elasticity of the total per-vehicle emission (includes e0): {tot:.4f}  -- not beta + eps_Gamma')
'''),
    md('## 5. Figure'),
    code(r'''
xs = np.linspace(card['x_min_h'], card['x_max_h'], 121)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
for name, c, col in (('calibrated n', card, 'k'), ('fixed service', fixed, '#2a78d6')):
    R = [qvdfe.response(c, x, coef) for x in xs]
    ok = np.array([r['valid'] for r in R])
    for k, key in enumerate(('vq', 'Gamma', 'increment_elasticity')):
        y = np.array([r[key] for r in R], float)
        ax[k].plot(xs, np.where(ok, y, np.nan), color=col, label=name if k == 0 else None)
        ax[k].plot(xs, np.where(ok, np.nan, y), color=col, ls=':', lw=.8)   # dotted: outside the admissible branch
    ax[2].axhline(R[0]['beta'], color=col, ls=':', lw=.8)
for a_ in ax: a_.axvline(x0, color='grey', lw=.6)
ax[0].set(xlabel='x = D/C (h)', ylabel='v_q (mph)', title='Queue speed (dotted: not admissible)'); ax[0].legend(frameon=False)
ax[1].set(xlabel='x (h)', ylabel='Gamma CO2 (g/(veh h))', title='Conversion factor')
ax[2].set(xlabel='x (h)', ylabel='elasticity', title='Increment elasticity (dotted: beta)')
plt.tight_layout()
'''),
    md(r'''
## 6. Interpretation question
On this card, is $\varepsilon_{\Gamma,x}$ positive or negative, and what does its sign say about whether a delay-based congestion charge would under- or over-state the emission increment caused by an added vehicle?
'''),
]

# ================================================================================================
T['T4_finite_transitions'] = [
    header('T4 · Finite transitions',
           'T2; Taylor expansion; quadrature.',
           'speed mph, acceleration m/s² (converted inside), duration s for reporting and h inside emission integrals.',
           'Time and distance preservation; the 9/140 coefficient by quadrature; the admissible band; the speed-only correction kept separate from the operating-mode term.',
           'Symmetric kernel; cubic speed-only rate; both transitions fit inside the link (Eq. 26). The operating-mode (acceleration) part needs acceleration-dependent rates and is not computed from the speed-only table.',
           'Section 5, Eqs. (21)–(27); Appendix F.'),
    md(r'''
## 1. Question
Real vehicles decelerate into the queue and accelerate out of it. How does a finite transition change emissions, and when do both transitions fit inside the link?

## 2. Assumptions
A transition of duration $T$ between $v_1$ and $v_2$ follows $v=v_1+(v_2-v_1)h(\xi)$ with $h(\xi)=3\xi^2-2\xi^3$, peak magnitude $a=\tfrac32|\Delta v|/T$. Travel time and distance of the traversal are preserved.

## 3. Derivation
$\int_0^1h\,d\xi=\tfrac12$, so the transition covers $(v_1+v_2)T/2$, the same as $T/2$ at each speed. Replacing the instantaneous switch therefore changes only the emission. For a cubic rate (Eq. 23):
$$\delta e=-\tfrac{9}{140}\,T\,(\Delta v)^2\,r''(\bar v),\qquad \bar v=\tfrac{v_1+v_2}2 .$$
Each transition takes half its duration from each portion, so both portions must last at least $(T_\downarrow+T_\uparrow)/2$ (Eq. 26):
$$\frac{(v_f-v_q)(T_\downarrow+T_\uparrow)}{2v_f}\le w\le L\Big(\frac1{v_q}-\frac1{v_f}\Big)-\frac{(v_f-v_q)(T_\downarrow+T_\uparrow)}{2v_q}.$$
This is the speed-only part. The operating-mode part (acceleration-dependent emission) requires acceleration-dependent rates and is not computed from the speed-only table. Section 6.6 of the paper compares the kernel with measured platoon transitions at low speeds (about 8.5–21 mph); the 46.5 mph reference traversal lies outside that measured support.
'''),
    code(PRELUDE),
    md('## 4. Worked example\nThe detector-78 reference state of Appendix F with peak magnitudes 1.5 and 1.0 m/s².'),
    code(r'''
from qvdfe.data import load_reference_state
ref = load_reference_state()
L, vf, vq = ref['L_mi'], ref['vf_mph'], ref['vq_mph']
Td_s, Tu_s = qvdfe.transition_time_s(vf - vq, 1.5), qvdfe.transition_time_s(vf - vq, 1.0)
print(f'T_down = {Td_s:.2f} s, T_up = {Tu_s:.2f} s')
xi = np.linspace(0, 1, 100001); h = qvdfe.kernel(xi)
check('distance preserved: integral of h = 1/2', np.isclose(np.trapezoid(h, xi), 0.5, rtol=1e-9))
check('peak of h-prime is 3/2', np.isclose(qvdfe.kernel_prime(xi).max(), 1.5, rtol=1e-9))
for p in qvdfe.POLLUTANTS:
    c = RATES[p]['coef']; T = Td_s / 3600
    v = vf + (vq - vf) * h
    quad = T * np.trapezoid(qvdfe.cubic_rate(v, c), xi) - T / 2 * (qvdfe.cubic_rate(vf, c) + qvdfe.cubic_rate(vq, c))
    closed = qvdfe.speed_only_correction(T, vf, vq, c)
    check(f'{p}: quadrature {quad:.6g} g equals -(9/140) T dv^2 r\'\' = {closed:.6g} g', np.isclose(quad, closed, rtol=1e-6, atol=1e-12))
lo, hi = qvdfe.transition_band(L, vf, vq, Td_s / 3600, Tu_s / 3600)
print(f'band {lo*60:.3f} to {hi*60:.3f} min; mean delay {ref["mean_delay_h"]*60:.3f} min; peak delay {ref["peak_delay_h"]*60:.3f} min')
check('mean-delay vehicle is admitted', lo <= ref['mean_delay_h'] <= hi)
check('peak-delay vehicle is not admitted', not (lo <= ref['peak_delay_h'] <= hi))
print('Baseline (Eq. 9) admits the peak-delay vehicle:', bool(qvdfe.finite_link_ok(ref['peak_delay_h'], L, vf, vq)),
      '-> baseline feasibility and transition feasibility are different conditions.')
'''),
    md('## 5. Figure'),
    code(r'''
ts = np.linspace(0, Td_s, 200)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].plot(ts, vf + (vq - vf) * qvdfe.kernel(ts / Td_s), color='k', label='kernel transition')
ax[0].step([0, Td_s / 2, Td_s], [vf, vq, vq], where='post', color='grey', ls='--', label='instantaneous switch at T/2')
ax[0].set(xlabel='time (s)', ylabel='speed (mph)', title='Same time, same distance'); ax[0].legend(frameon=False, fontsize=8)
ws = np.linspace(0, qvdfe.finite_link_allowance(L, vf, vq) * 1.1, 200)
ax[1].axvspan(lo * 60, hi * 60, color='#8C1D40', alpha=.12, label='transition band (Eq. 26)')
ax[1].axvline(qvdfe.finite_link_allowance(L, vf, vq) * 60, color='k', ls='--', label='finite-link allowance (Eq. 9)')
ax[1].axvline(ref['mean_delay_h'] * 60, color='#2a78d6', label='mean delay'); ax[1].axvline(ref['peak_delay_h'] * 60, color='#2a78d6', ls=':', label='peak delay')
ax[1].set(xlabel='delay w (min)', yticks=[], title='Which vehicles fit both transitions'); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout()
'''),
    md(r'''
## 6. Interpretation question
For a closed episode the delay is zero at onset and clearance, while the band needs $w\ge w_{\rm lo}>0$. Why can the admitted share never reach 100 %, and why should the admitted subset use its own $D_{\mathcal A}$ and $\bar w_{\mathcal A}$ rather than the episode mean delay? (Example 3 works this out on a long average-weekday episode whose cohort is reconstructed from its fitted delay profile.)
'''),
]

# ------------------------------------------------------------------------------------------------
# Instructor solutions: one notebook with worked answers to the exercises.
SOLUTIONS = [
    md('# Instructor solutions\n\nWorked answers to `teaching/exercises/`. Keep this folder apart from the student materials when distributing.'),
    code(PRELUDE),
    md('## T1-E1: total delay of the two closed profiles\nNewell: $W_P=bP^4/36$. The sinusoidal surplus $\\lambda-\\mu=A\\sin(2\\pi t/P)$ gives $Q=\\frac{AP}{2\\pi}(1-\\cos(2\\pi t/P))$ and $W_P=AP^2/(2\\pi)$.'),
    code(r'''
mu, P, b, A = 1500, 2.0, 100, 120
t = np.linspace(0, P, 8001)
print('Newell W_P', qvdfe.NewellQueue(mu=mu, P=P, b=b).W, ' numerical', qvdfe.closed_queue(t, qvdfe.NewellQueue(mu=mu, P=P, b=b).lam(t), mu)['W_P'])
print('sine   W_P', A * P**2 / (2 * np.pi), ' numerical', qvdfe.closed_queue(t, mu + A * np.sin(2 * np.pi * t / P), mu)['W_P'])
'''),
    md('## T1-E2: amplitude that makes the peak-delay vehicle reach the finite-link allowance\n$w_{t_2}=4bP^3/(81\\mu)=L(1/v_q-1/v_f)$, so $b^*=81\\mu L(1/v_q-1/v_f)/(4P^3)$.'),
    code(r'''
L, vf, vq = 1.0, 60.0, 20.0
bstar = 81 * mu * qvdfe.finite_link_allowance(L, vf, vq) / (4 * P**3)
print(f'b* = {bstar:.2f} veh/h^3; check peak delay {qvdfe.NewellQueue(mu=mu, P=P, b=bstar).peak_delay*60:.4f} min = allowance {qvdfe.finite_link_allowance(L, vf, vq)*60:.4f} min')
'''),
    md('## T2-E1: queue speed at which Γ changes sign (CO₂, v_f = 60 mph)'),
    code(r'''
from scipy.optimize import brentq
c = RATES['CO2']['coef']
vs = np.linspace(5, 59, 541); g = qvdfe.gamma_cubic(vs, 60, c)
roots = [brentq(lambda v: qvdfe.gamma_cubic(v, 60, c), vs[i], vs[i + 1]) for i in range(len(vs) - 1) if g[i] * g[i + 1] < 0]
print('sign changes of Gamma(v_q) at', [round(r, 2) for r in roots], 'mph' if roots else '(none in 5-59 mph)')
for p in qvdfe.POLLUTANTS:
    g = qvdfe.gamma_cubic(vs, 60, RATES[p]['coef']); print(p, 'Gamma range', round(g.min(), 3), 'to', round(g.max(), 3))
'''),
    md('## T2-E2: VMT/VHT form for an episode\n$E_P=\\frac{r(v_f)}{v_f}{\\rm VMT}+\\Gamma({\\rm VHT}-{\\rm VMT}/v_f)$; the excess vehicle-hours ${\\rm VHT}-{\\rm VMT}/v_f=D\\bar w_P$ are converted at $\\Gamma(v_q)$, not at $r(v_f)$ or at an average-speed rate.'),
    md('## T3-E1: where the increment elasticity departs most from β'),
    code(r'''
cards = qvdfe.load_cards(); cards = cards[cards.C.notna()]
rows, seen = [], set()
for c in cards.to_dict('records'):
    key = (c['corridor'], round(c['fd'], 9), round(c['n'], 9), round(c['fp'], 9), round(c['s'], 9))
    if key in seen:          # corridor-wide calibrations repeat across periods
        continue
    seen.add(key)
    r = qvdfe.response(c, c['x_median_h'], RATES['CO2']['coef'])
    if r['valid'] and np.isfinite(r['eps_Gamma']):
        rows.append((c['corridor'], c['period'], round(float(c['n']), 3), round(r['beta'], 3), round(r['Gamma'], 1), round(float(r['eps_Gamma']), 3)))
rows.sort(key=lambda z: -abs(z[5]))
print('corridor, period, n, beta, Gamma (g/(veh h)), eps_Gamma at the card median loading (admissible cards only)')
for z in rows: print(z)
print("Large |eps_Gamma| occurs where Gamma is close to zero: the ratio x Gamma'/Gamma is then large, and it is undefined at Gamma = 0.")
'''),
    md('## T3-E2: marginal emission at the reference loading, split into the average and the external part'),
    code(r'''
from qvdfe.data import load_reference_state
ref = load_reference_state()
card = cards[(cards.corridor == 'AZ_I10_W') & (cards.period == 'MD')].iloc[0].to_dict()
r0 = qvdfe.response(card, ref['x_h'], RATES['CO2']['coef'])
for name, c in (('calibrated n', card), ('fixed service mu0 = mu(x0)', {**card, 'n': 1.0, 'fd': r0['P'] / ref['x_h']})):
    r = qvdfe.response(c, ref['x_h'], RATES['CO2']['coef'])
    print(f"{name:20s} e = {r['e_g']:.2f} g/veh, external x e'(x) = {r['external_g']:.2f} g/veh, marginal dE/dD = {r['marginal_g']:.2f} g/veh (valid={r['valid']})")
print('Both rows share e (same state at x0). Under fixed service the external part is beta * Gamma * wbar only;')
print('the calibrated card adds wbar * x dGamma/dx, the queue-state term, because v_q falls as loading rises.')
'''),
    md('## T4-E1: sensitivity of the band to the peak magnitudes'),
    code(r'''
from qvdfe.data import load_reference_state
ref = load_reference_state(); L, vf, vq = ref['L_mi'], ref['vf_mph'], ref['vq_mph']
for ad, au in [(1.0, 0.67), (1.5, 1.0), (2.0, 1.5), (3.0, 2.0)]:
    lo, hi = qvdfe.transition_band(L, vf, vq, qvdfe.transition_time_s(vf - vq, ad) / 3600, qvdfe.transition_time_s(vf - vq, au) / 3600)
    print(f'({ad}, {au}) m/s^2: band {lo*60:.3f}-{hi*60:.3f} min; peak delay {ref["peak_delay_h"]*60:.3f} min admitted: {lo <= ref["peak_delay_h"] <= hi}')
'''),
]


def stale_notebooks():
    """Names of committed notebooks whose cell sources differ from what this file generates."""
    out = []
    targets = [(TUT / f'{name}.ipynb', cells) for name, cells in T.items()] + [(SOL / 'solutions.ipynb', SOLUTIONS)]
    for path, cells in targets:
        if not path.exists():
            out.append(path.name)
            continue
        have = [c.source for c in nbformat.read(path, as_version=4).cells]
        if have != [c.source for c in cells]:
            out.append(path.name)
    return out


def execute(paths):
    from nbclient import NotebookClient
    bad = []
    for p in paths:
        nb = nbformat.read(p, as_version=4)
        try:
            NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(p.parent)}}).execute()
            nbformat.write(nb, p)
            print('executed', p.relative_to(HERE))
        except Exception as exc:  # noqa: BLE001
            bad.append((p.name, str(exc)[-2500:]))
            print('FAILED', p.relative_to(HERE))
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-execute', action='store_true')
    a = ap.parse_args()
    paths = [save(cells, TUT / f'{name}.ipynb') for name, cells in T.items()]
    paths.append(save(SOLUTIONS, SOL / 'solutions.ipynb'))
    print('generated', len(paths), 'notebooks')
    if not a.no_execute:
        bad = execute(paths)
        for n, e in bad:
            print('----', n, chr(10), e)
        sys.exit(1 if bad else 0)
