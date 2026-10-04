# Reproduction entry; reads the frozen inputs through qvdfe.paths and writes to reproduce/output.
"""Fig. 5 (file fig13_transition): one vehicle at the detector 78 reference state; KS1 finite transitions
against the two-speed path, and the acceleration along the kernel (black/grey with one blue accent,
direct labels, no notes inside the figure; overlap-checked)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / 'lib'))  # reproduce/lib
from _paths import ROOT as _ROOT, FROZEN as _FROZEN, OUT as _OUT  # noqa: E402
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcheck import plt, INK, GRAY, LIGHT, BLUE, check_overlaps

HERE = _OUT
vf, vq = 64.16, 17.67                     # mph, detector 78 reference state
a_dn, a_up = 1.5, 1.0                     # m/s^2, illustrative
mph2ms = 0.44704
dv = (vf - vq) * mph2ms
T_dn, T_up = 3 * dv / (2 * a_dn), 3 * dv / (2 * a_up)
T_queue_mean = 69.2                       # s, v_f w/(v_f - v_q) at the mean delay 0.84 min
T_plateau = T_queue_mean - (T_dn + T_up) / 2
t1 = 15.0; t2 = t1 + T_dn; t3 = t2 + T_plateau; t4 = t3 + T_up; t_end = t4 + 15.0


def h(xi): return 3 * xi**2 - 2 * xi**3
def hp(xi): return 6 * xi * (1 - xi)


t = np.linspace(0, t_end, 2000)
v = np.full_like(t, vf); a = np.zeros_like(t)
m = (t >= t1) & (t <= t2); xi = (t[m] - t1) / T_dn
v[m] = vf + (vq - vf) * h(xi); a[m] = (vq - vf) * mph2ms / T_dn * hp(xi)
m = (t > t2) & (t < t3); v[m] = vq
m = (t >= t3) & (t <= t4); xi = (t[m] - t3) / T_up
v[m] = vq + (vf - vq) * h(xi); a[m] = (vf - vq) * mph2ms / T_up * hp(xi)
s1, s2 = t1 + T_dn / 2, t3 + T_up / 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.07, 2.55), gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.2, wspace=0.3)

for ax in (ax1, ax2):
    ax.axvspan(t1, t2, color=LIGHT, alpha=0.35, lw=0)
    ax.axvspan(t3, t4, color=LIGHT, alpha=0.35, lw=0)

# (a) speed
ax1.plot([0, s1, s1, s2, s2, t_end], [vf, vf, vq, vq, vf, vf], color=INK, lw=1.1, ls=(0, (4, 2)))
ax1.plot(t, v, color=BLUE, lw=1.8)
ax1.axhline(vf, color=GRAY, lw=0.5, ls=":", gid="ref"); ax1.axhline(vq, color=GRAY, lw=0.5, ls=":", gid="ref")
ax1.text(1.0, vf + 2.0, r"$v_f$", fontsize=9, va="bottom")
ax1.text(1.0, vq + 2.0, r"$v_q$", fontsize=9, va="bottom")
for lo, hi, lab in ((t1, t2, r"$T_\downarrow$"), (t3, t4, r"$T_\uparrow$")):
    ax1.annotate("", xy=(lo, 80), xytext=(hi, 80), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
    ax1.text((lo + hi) / 2, 82, lab, ha="center", va="bottom", fontsize=9)
ax1.text((t2 + t3) / 2, vq - 3, "queued state", ha="center", va="top", fontsize=8)
ax1.annotate("KS1", xy=(t1 + 0.75 * T_dn, float(np.interp(t1 + 0.75 * T_dn, t, v))), xytext=(t1 + T_dn + 12, 44),
             fontsize=8, ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax1.annotate("Two-speed path,\nswitch at $T_\\downarrow/2$", xy=(s1, 50), xytext=(40.0, 72), fontsize=8, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax1.set_xlim(0, t_end); ax1.set_ylim(0, 92)
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Speed (mph)")
ax1.set_title("(a)  One vehicle: speed", loc="left")

# (b) acceleration
ax2.plot(t, a, color=BLUE, lw=1.8)
ax2.axhline(0, color=INK, lw=0.6, gid="ref")
ax2.hlines(-dv / T_dn, t1, t2, color=INK, lw=1.0, ls=(0, (4, 2)))
ax2.hlines(dv / T_up, t3, t4, color=INK, lw=1.0, ls=(0, (4, 2)))
ax2.text(t2 + 2, -a_dn, r"peak $a_\downarrow$", va="center", fontsize=8)
ax2.text(t3 - 2, a_up, r"peak $a_\uparrow$", va="center", ha="right", fontsize=8)
ax2.annotate(r"mean $=\frac{2}{3}$ peak", xy=(t2 - 3, -dv / T_dn), xytext=(t2 + 8, -0.45), fontsize=8, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax2.set_xlim(0, t_end); ax2.set_ylim(-2.0, 1.5)
ax2.set_xlabel("Time (s)"); ax2.set_ylabel(r"Acceleration $\dot v$ (m/s$^2$)")
ax2.set_title("(b)  Acceleration along the KS1 kernel", loc="left")

issues = check_overlaps(fig)
fig.savefig(HERE / "figs" / "fig13_transition.pdf"); fig.savefig(HERE / "figs" / "fig13_transition.png", dpi=220)
print(f"T_dn={T_dn:.1f} s, T_up={T_up:.1f} s, plateau={T_plateau:.1f} s")
sys.exit(1 if issues else 0)
