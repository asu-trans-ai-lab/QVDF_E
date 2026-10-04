# Teaching materials

Four short tutorials that follow the paper's theory in order. Each tutorial follows the same pattern: **question → assumptions → derivation → worked example → figure → interpretation question**. Each one states its prerequisites, input units, expected outputs and the conditions under which its result holds. All computations call the shared package `qvdfe` (`src/qvdfe/`), which the reproduction scripts also use.

| Tutorial | Question | Checkable product | Paper |
| --- | --- | --- | --- |
| [T1 Bottleneck and closed episode](tutorials/T1_bottleneck_closed_episode.ipynb) | How do inflow, service rate, queue and delay connect? | Oblique cumulative curves; D = μP; W_P = ∫Q dt = D·w̄; 9/16 for the Newell model only | Sec. 2.2 |
| [T2 Two speeds and emissions](tutorials/T2_two_speeds_and_emissions.ipynb) | Why do equal VMT and VHT still need operating-state information? | Time–distance split; Eq. (10) = VMT/VHT form; three equal-VMT/VHT patterns with different emissions | Sec. 3 |
| [T3 Demand-dependent response](tutorials/T3_demand_dependent_response.ipynb) | How does a changing service rate alter the delay-to-emission conversion? | Fixed service versus calibrated n; increment elasticity β + ε_Γ,x (Γ ≠ 0) | Sec. 4 |
| [T4 Finite transitions](tutorials/T4_finite_transitions.ipynb) | How do finite decelerations and accelerations change emissions, and when do they fit? | Time and distance preserved; 9/140 by quadrature; admissible band; speed-only versus operating-mode terms | Sec. 5 |

- `exercises/` contains two exercises per tutorial.
- `instructor/` contains the worked solutions. Keep this folder apart from student copies.
- `advanced/` contains optional readings on scheduling (Appendix D), pricing and the two-route illustration (Appendix E), and the reference state (Appendix F).

**Running.** Install with `pip install -e ".[teaching]"` from the repository root (this includes JupyterLab), or skip installation: each notebook finds `src/qvdfe` itself. Open a notebook in Jupyter from the repository root or from `teaching/`. To regenerate and execute all notebooks, run:

```bash
python teaching/build_tutorials.py
```

**Order.** Before T1, run `examples/01_synthetic_episode` (synthetic values). It works through the five checks of T1–T2 on one page.
