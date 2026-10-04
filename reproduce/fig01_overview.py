# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Fig. 1: one congestion episode at three resolutions.
(a) closed fluid queue as cumulative arrival and departure curves, drawn after Arnott, de Palma & Lindsey
    (1990, p. 117) in the paper's notation: Q(t) vertical, w(t) horizontal, peak delay w_t2 at t2, P = t3 - t0;
(b) calibrated duration -> peak-delay relationships w_t2 = T_f f_p P^s for the four daily AZ periods at
    detector 78, with the s = 1 reference through the midday reference state (square);
(c) one vehicle's KS1 transition v_f -> v_q against the two-speed path (detector 78 reference state).
Black/grey line work with one blue accent, direct labels, no notes inside the figure; overlap-checked."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcheck import plt, INK, GRAY, LIGHT, BLUE, check_overlaps, arrow_axes

HERE = _OUT
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.07, 2.55), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})
fig.subplots_adjust(left=0.05, right=0.985, top=0.88, bottom=0.25, wspace=0.52)

# ---------------- (a) Arnott et al. (1990) style queueing diagram, schematic units (P = 1, mu = 1)
t0, t2, t3, mu = 0.0, 0.42, 1.0, 1.0
lam1 = 2.2                                   # arrival rate before t2 (> mu)
A2 = lam1 * t2                               # cumulative arrivals at t2
tt = 0.25                                    # a generic arrival time t
A = lambda t: np.where(t <= t2, lam1 * t, A2 + (mu * t3 - A2) * (t - t2) / (t3 - t2))
w2 = (A2 - mu * t2) / mu                     # peak delay
At = float(A(tt)); wt = (At - mu * tt) / mu
xs = np.linspace(t0, t3, 200)
ax1.fill_between(xs, mu * xs, A(xs), color=LIGHT, alpha=0.55, lw=0)
ax1.plot(xs, A(xs), color=INK, lw=1.4)
ax1.plot([t0, t3], [0, mu * t3], color=INK, lw=1.2, ls=(0, (4, 2)))
for x, top in ((tt, At), (t2, A2), (t3, mu * t3)):
    ax1.plot([x, x], [0, top], color=GRAY, lw=0.6, ls=(0, (2, 2)), gid="ref")
ax1.annotate("", xy=(tt, mu * tt), xytext=(tt, At), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0))
ax1.text(tt + 0.012, mu * tt + 0.72 * (At - mu * tt), "$Q(t)$", ha="left", va="center", fontsize=8.5)
ax1.annotate("", xy=(tt, At), xytext=(tt + wt, At), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0))
ax1.text(tt + 0.58 * wt, At + 0.015, "$w(t)$", ha="center", va="bottom", fontsize=8.5)
ax1.annotate("", xy=(t2, A2), xytext=(t2 + w2, A2), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0))
ax1.text(t2 + 0.5 * w2, A2 - 0.025, "$w_{t_2}$", ha="center", va="top", fontsize=8.5)
ax1.text(0.57, 0.70, "$W_P$", ha="center", va="center", fontsize=9)
ax1.annotate("Cumulative\narrivals $A(t)$", xy=(0.36, float(A(0.36))), xytext=(0.02, 0.97), fontsize=7.5, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax1.annotate("Cumulative departures\n$D(t)$, slope $\\mu$", xy=(0.80, 0.80), xytext=(0.99, 0.22), fontsize=7.5, ha="right", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax1.text(t3 - 0.01, mu * t3 + 0.04, "$D=\\mu P$", ha="right", va="bottom", fontsize=8.5)
ax1.set_xticks([t0, tt, t2, t3]); ax1.set_xticklabels(["$t_0$", "$t$", "$t_2$", "$t_3$"])
ax1.set_yticks([])
ax1.annotate("", xy=(t0, -0.2), xytext=(t3, -0.2), xycoords="data", arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0),
             annotation_clip=False)
ax1.text((t0 + t3) / 2, -0.23, "$P=t_3-t_0$", ha="center", va="top", fontsize=8.5, clip_on=False)
ax1.set_xlim(t0, 1.08); ax1.set_ylim(0, 1.15)
arrow_axes(ax1, 1.08, 1.15)
ax1.set_ylabel("Cumulative vehicles")
ax1.set_title("(a)  Congestion episode", loc="left")

# ---------------- (b) calibrated duration -> peak delay (Table A.1, daily AZ, detector 78)
vf, L = 64.16, 1.040
Tf = L / vf * 60
periods = [("MD", 1.724, 0.846, INK, "-"), ("PM", 2.317, 0.755, INK, (0, (4, 2))),
           ("NT", 1.638, 0.746, GRAY, "-"), ("AM", 0.685, 0.502, GRAY, (0, (4, 2)))]
Pg = np.linspace(0.3, 8.0, 300)
P_ref, w_ref = 1.378, 2.20                    # daily MD calibration at x = 1.052 h (midday reference state)
ends = []
for name, fp, s, col, ls in periods:
    ax2.plot(Pg, Tf * fp * Pg**s, color=col, lw=1.4, ls=ls)
    ends.append([Tf * fp * 8.0**s, f"{name}, $s={s:.2f}$"])
ends.append([w_ref * 8.0 / P_ref, "$s=1$"])
ends.sort()                                   # spread end labels vertically (at least 1.15 min apart)
for k in range(1, len(ends)):
    ends[k][0] = max(ends[k][0], ends[k - 1][0] + 1.15)
for y, lab in ends:
    ax2.text(8.15, y, lab, fontsize=7.5, ha="left", va="center")
ax2.plot(Pg, w_ref * Pg / P_ref, color=GRAY, lw=0.9, ls=":")
ax2.plot([P_ref], [w_ref], "s", color=BLUE, ms=4.5, mfc="white", mew=1.1)
ax2.set_xlim(0, 10.9); ax2.set_ylim(0, 14); ax2.set_xticks([0, 2, 4, 6, 8]); ax2.spines["bottom"].set_bounds(0, 8)
ax2.set_xlabel("Congestion duration $P$ (h)"); ax2.set_ylabel("Peak delay $w_{t_2}$ (min)")
ax2.set_title(r"(b)  $w_{t_2}=T_f f_p P^{\,s}$", loc="left")

# ---------------- (c) one vehicle: KS1 transition against the two-speed path
vq, a_dn = 17.67, 1.5
dv = (vf - vq) * 0.44704; T_dn = 3 * dv / (2 * a_dn)
t = np.linspace(0, 45, 600); v = np.full_like(t, vf)
t1 = 10.0; m = (t >= t1) & (t <= t1 + T_dn); xi = (t[m] - t1) / T_dn
v[m] = vf + (vq - vf) * (3 * xi**2 - 2 * xi**3); v[t > t1 + T_dn] = vq
sw = t1 + T_dn / 2
ax3.axvspan(t1, t1 + T_dn, color=LIGHT, alpha=0.35, lw=0)
ax3.plot([0, sw, sw, 45], [vf, vf, vq, vq], color=INK, lw=1.1, ls=(0, (4, 2)))
ax3.plot(t, v, color=BLUE, lw=1.8)
ax3.annotate("", xy=(t1, 80), xytext=(t1 + T_dn, 80), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
ax3.text(t1 + T_dn / 2, 82, r"$T_\downarrow$", ha="center", va="bottom", fontsize=9)
ax3.annotate("KS1", xy=(24.0, float(np.interp(24.0, t, v))), xytext=(34, 44), fontsize=8, ha="center", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax3.annotate("Two-speed\npath", xy=(sw, 30), xytext=(8.6, 36), fontsize=8, ha="center", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax3.text(t1 + T_dn / 2, 4, r"$\Delta x=\frac{v_f+v_q}{2}T_\downarrow$", ha="center", va="bottom", fontsize=8)
ax3.text(0.8, vf + 2, r"$v_f$", fontsize=9, va="bottom"); ax3.text(44, vq + 2, r"$v_q$", fontsize=9, ha="right", va="bottom")
ax3.set_xlim(0, 45); ax3.set_ylim(0, 92)
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Speed (mph)")
ax3.set_title("(c)  Vehicle transition", loc="left")

issues = check_overlaps(fig)
fig.savefig(HERE / "figs" / "fig1_overview.pdf"); fig.savefig(HERE / "figs" / "fig1_overview.png", dpi=220)
print(f"(a) schematic: w_t2 = {w2:.2f} P, Q(t) = {At - mu*tt:.2f}; (b) Tf = {Tf:.3f} min; (c) T_dn = {T_dn:.1f} s")
sys.exit(1 if issues else 0)
