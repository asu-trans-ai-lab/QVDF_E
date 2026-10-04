# Exercises

Two exercises per tutorial. Work in a copy of the tutorial notebook; every exercise can be answered with the functions of `qvdfe`. Solutions are in `teaching/instructor/` (keep them apart when distributing).

## T1 · Bottleneck and the closed episode

**T1-E1.** For the two closed profiles of T1 (Newell with b = 100 veh/h³ and the sinusoidal surplus with amplitude 120 veh/h/lane, both with μ = 1500 veh/h/lane and P = 2 h), derive the total delay W_P in closed form and confirm it numerically with `qvdfe.closed_queue`. Which profile has the larger W_P, and why, although both serve the same D?

**T1-E2.** Using the link of `examples/01_synthetic_episode` (L = 1 mi, v_f = 60 mph, v_q = 20 mph), find the Newell amplitude b* at which the peak-delay vehicle exactly reaches the finite-link allowance L(1/v_q − 1/v_f). Copy the example folder, change `newell_b_veh_per_h3` in the copy's `config.json` to a value above b* and rerun: which check fails, and what does it mean physically? (The comparison with `expected/results.json` will also list MISMATCH lines and exit with status 1; that is expected after changing an input.)

## T2 · Two speeds and emissions

**T2-E1.** For CO₂ with v_f = 60 mph, find the queue speed at which Γ(v_q) changes sign. Explain the sign in terms of the emission per mile f(v) = r(v)/v. *Hint:* Γ < 0 means a mile driven at the queue speed emits less than a mile at v_f, so each hour of queued delay lowers the total. Repeat for NOx, CO and HC and report whether Γ changes sign in 5–59 mph.

**T2-E2.** Rewrite Eq. (12) for a link where the episode adds 500 vehicle-hours of excess time. Which rate converts those excess vehicle-hours into emissions, and why is it neither r(v_f) nor the rate at the average speed?

## T3 · Demand-dependent response

**T3-E1.** Evaluate `qvdfe.response` for every card of `qvdfe.load_cards()` at its median training loading (column `x_median_h`). Cards with `calibration_scope = corridor_all_periods` share one calibration across periods, so keep one row per distinct calibration; cards without a supported fit (empty `n`) are skipped; keep only states with `valid = True`. Where does the increment elasticity differ most from β? Relate the largest values of |ε_Γ,x| to the size of Γ.

**T3-E2.** For the Arizona midday card at the reference loading, compute the marginal emission dE/dD and split it into the average emission e and the external part x e′(x). How does the external part change under fixed service anchored at the same state (n = 1 and f_d = P(x)/x, as in T3)?

## T4 · Finite transitions

**T4-E1.** Recompute the transition band of the reference state for the peak magnitudes (1.0, 0.67), (1.5, 1.0), (2.0, 1.5) and (3.0, 2.0) m/s². For which pairs is the peak-delay vehicle admitted?

**T4-E2.** Run `examples/03_long_episode_applicability`. Explain why the admitted subset's total must use D_𝒜 and w̄_𝒜, and quantify the error from using the full-episode mean delay instead. Why is the reported total a speed-only quantity?
