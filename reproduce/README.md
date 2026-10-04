# Reproducing the paper's results from the frozen data

```bash
python reproduce/run_all.py                 # every entry below, about one minute
python reproduce/run_all.py --skip-figures  # calculations and audit only
python reproduce/run_all.py --strict        # also fail when a figure differs from a published copy
```

Each entry reads the frozen inputs in `data/frozen/` through `qvdfe.paths` and writes only to `reproduce/output/`. `run_all.py` runs every entry and then compares each calculation and audit file it writes with the frozen file, value by value. When a copy of the published figures is available, set `QVDFE_PAPER_FIGS` to that folder and the regenerated figures are compared with it by rendering both.

The test suite (`pytest`) separately checks the mathematical identities and the published values (Tables 8, F1 and F2, the Appendix F reference values and the episode counts) with the core package `qvdfe`.

## Supported here

| Entry | Paper item | Comparison with the frozen data |
| --- | --- | --- |
| `fig01_overview.py` | Fig. 1 | regenerated |
| `fig03_shockwave.py` | Fig. 3 (shock waves at the bottleneck) | regenerated |
| `fig04_figD1_queue_and_scheduling.py` | Fig. 4 (closed fluid queue) and Fig. D1 (scheduling) | regenerated with an earlier layout; geometry checks pass (see the to-do in the main README) |
| `fig05_transition.py` | Fig. 5 (finite transition) | regenerated |
| `fig07_paired_episode.py` | Fig. 7 and its audit (12 weekdays reproduce the monthly profile bin by bin) | audit identical |
| `fig09_fig10_fig11_fig12_evidence.py` | Figs. 9 (calibration), 10 (emission comparison), 11 (Γ at calibrated states), 12 (elasticity) | regenerated |
| `figE1_two_route.py` | Fig. E1 (two-route illustration) | regenerated |
| `calc_cohort_admissibility.py` | Cohort transition admissibility; Appendix F, Table F2 | all outputs identical |
| `calc_corridor_admissibility.py` | Corridor-scale geometric admissibility | all outputs identical |
| `calc_daytime_period.py` | One daytime accounting window | all outputs identical |
| `audit_numerics.py` | Table F1 at full precision, exclusion ledger, Table 8 metrics, recorded-speed recalculation, later-date diagnostic with a null benchmark | all outputs identical |

Matplotlib and font versions change pixels slightly; `requirements.txt` lists the versions the figures were checked with.

## Not supported here (original data needed)

| Item | Needs |
| --- | --- |
| Episode detection, QVDF calibration and the emission ladder O/M/S behind `data/frozen/calibration` | The original ADOT and PeMS panels and the authors' experiment pipeline |
| Fig. 2 (emission rate and operating states) | The original MOVES lookup run |
| Fig. 6 (study map) | Map layers and the upstream sensor inventory |
| Fig. 13 (measured platoon transitions) | Raw platoon trajectories (not redistributed) |
| Fig. 8 (corridor delay) | The corridor time-space series; the derived records are in `data/frozen/transitions/corridor/` |
| Emission rates | Results start from the supplied MOVES-derived rate tables; the original MOVES run settings were not recorded |
