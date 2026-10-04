# Examples

Three runnable examples. Run each from any folder with `python examples/<name>/run.py`, or open the notebook in its folder. Each example reads its inputs from `config.json` and writes `figure.png` and `results.json`, which are not tracked by git; the reference figure is `expected/figure.png`. It then compares the results with `expected/results.json`, and its exit status is nonzero if any check or comparison fails. Every `results.json` records the paper version and the repository commit.

| Example | Data | What it shows | Runtime |
| --- | --- | --- | --- |
| [01_synthetic_episode](01_synthetic_episode/) | **Synthetic** (round values in `config.json`) | Queued state on the FD; D = μP and W_P for a Newell queue; the finite-link condition; e = e₀ + Γw; the episode total in the delay form and the VMT/VHT form | < 5 s |
| [02_weekdays_vs_average](02_weekdays_vs_average/) | Real: detector 78, Nov 2016 (extract in `data/examples/`) | Twelve weekdays reproduce the average-weekday profile bin by bin; the averaged-profile episode is not the average of the daily episodes (paper Fig. 7) | < 5 s |
| [03_long_episode_applicability](03_long_episode_applicability/) | Real: frozen average-weekday results | Baseline feasibility (Eq. 9) versus common-transition feasibility (Eq. 26) on the long episode of Appendix F. The admitted subset's own D_𝒜 and w̄_𝒜 give the **admitted-subset speed-only emissions** of Eq. (27). The output also shows the error from using the episode mean delay | < 10 s |

The values in example 1 are illustrative. The rate is the archived MOVES-derived CO₂ cubic, with r in g/(veh·h) and v in mph. If you change `config.json`, the comparison with `expected/results.json` will report the changed values. That report is expected; the physical checks still apply.
