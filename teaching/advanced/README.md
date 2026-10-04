# Advanced readings (optional)

These readings follow the paper's appendices. They assume T1–T4 and point to the reproduction entries that regenerate the corresponding figures and tables from the frozen outputs.

## A1 · Scheduling reference (paper Appendix D)

The classical homogeneous bottleneck equilibrium (Vickrey; Arnott, de Palma and Lindsey; Small) gives a peak delay proportional to the duration, w_t2 = c_E c_L / [c_w (c_E + c_L)] · P, so s = 1 and θ = 1/2 in the QVDF notation (Eq. D2). The calibrated exponents s are an empirical analogue, not an equilibrium result.

*Try:* with `qvdfe.qvdf`, set s = 1 and θ = 1/2 and compare the mean delay with the calibrated card of T3 over the card's loading range.

## A2 · Pricing and the two-route illustration (paper Appendix E)

With fixed background demand, flexible travelers choose between a congested bottleneck route and an uncongested alternative. Time-only user equilibrium and system optimum differ. The time optimum shifts flexible traffic toward the bypass, which is 1.86 times the length of the congested route, and raises total CO₂ by 17.5 % relative to time-only user equilibrium; a CO₂-weighted objective therefore moves traffic back toward the shorter congested route. The cause is the extra free-flow distance on the bypass, not a change in Γ. The paper quotes the standard price-of-anarchy bound only as a no-background benchmark, not as a theorem for this network.

*Reproduce:* `python reproduce/figE1_two_route.py` redraws the Appendix E figure and table from the frozen assignment results.

## A3 · The reference state and long episodes (paper Appendix F)

Table F1 gives the per-vehicle emissions of the detector-78 reference state. Its rounded components (294.6 + 31.7) do not add to the rounded total (326.2): the total is rounded from full precision. Table F2 works one long PM episode through the transition band; `examples/03_long_episode_applicability` reproduces it and computes the admitted-subset speed-only emissions.

*Reproduce:* `pytest tests/test_reproduction.py -k "F1 or F2"`.
