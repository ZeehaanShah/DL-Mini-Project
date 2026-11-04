# Battery Dispatch Validation — Phase 3

Date: 2026-10-27. No model training.

## Equations and metric denominator

Per interval (kWh): `R_used + G_import + B_discharge + U = D`; `R_available = R_used + B_charge + curtailment`; `SOC_t = SOC_(t-1) + eta_charge*B_charge - B_discharge/eta_discharge`. SOC is bounded by capacity-scaled min/max; power-energy bounds use capacity × C-rate × interval duration. Renewable share is `(direct renewable used + renewable-origin battery discharge) / served demand`; initial carry-in discharge is reported separately and excluded.

Cyclic mode is a finite-horizon MILP, fixes terminal SOC to target, uses one binary charge/discharge mode per interval, prohibits grid charging, and lexicographically minimizes unmet energy, grid imports, curtailment, then battery throughput. Non-cyclic mode can end at a different SOC.

## Evidence

A deterministic synthetic 240-hour check completed in 0.714 s: 250 kWh battery, initial/target 125 kWh, final 125.0 (delta 0); SOC range 25–125 kWh; unmet 0; grid imports 4,419.6825 kWh; curtailment 0; no simultaneous charge/discharge; SOC balance residual 0.0. Served demand 9,168 kWh; direct renewable 4,210.2557 kWh; renewable-origin battery discharge 423.0637 kWh; initial carry-in discharge 114.9981 kWh; renewable share 50.54% of served supply excluding carry-in. Requesting a 75% terminal SOC from 50% with no renewable supply/load is infeasible under renewable-only charging and correctly errors.

Battery tests passed in the full suite, though suite overall is failing (215 passed, 5 failed). SciPy MILP deployment dependency and lexicographic priority assumptions require review; repeatable operation is only modeled over the chosen horizon/target.
