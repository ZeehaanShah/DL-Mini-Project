# Reproduced Metrics and Artifact Fingerprints

Metrics were reproduced in the isolated copy with frozen checkpoints; no model was trained. Hashes identify the exact data/checkpoint files.

| Model | Test ID | N | MAE | RMSE | R² | Unit |
|---|---|---:|---:|---:|---:|---|
| ResNet18_fine_tune | EXP_SOLAR_DL_S3_TEST | 14,003 | 2.3008 | 3.3736 | 0.8052 | kW |
| WeatherLSTM | EXP_NASA_DL_W1_TEST | 1,242 | 16.2575 | 30.1767 | 0.9844 | Wh/m^2 |
| WindLSTM | EXP_WIND_DL_C2_TEST | 1,242 | 0.5908 | 0.8216 | 0.7576 | m/s |
| DemandLSTM | EXP_DEMAND_DL_D2_TEST | 2,561 | 2.6152 | 4.0673 | 0.9234 | kWh |

## Artifact SHA-256

- SKIPP'D held-out test parquet (data/raw/skippd/test-00000-of-00001.parquet): 848fb838c533b185ee3d3b07c0739c742e4dfa5e4e9fe1ab1734b91478147982
- NASA POWER hourly source (data/raw/nasa_power/POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv): 5fb24f559cc2e2b4b8978e74ed214b3a27b326ad203c6b100c34f05792ee2e93
- Wind processed series (data/processed/wind/wind_hourly_pilot.csv): 511e1dd704be3c82859b2a20b6a0e83a05b50ee3cf761b685993eada5d9be18e
- Demand processed series (data/processed/demand/demand_hourly_pilot.csv): 8945003e4d27e9148b78fbeabaad3ac6b8223d2b2f2314892b6e0df4e5936e18
- Solar S3 checkpoint (checkpoints/EXP_SOLAR_DL_S3/best_model.pt): d5daf03479e989e306fbdcd18b49237522174b6bc0b7c729fb7f2a850839cb53
- NASA W1 checkpoint (checkpoints/EXP_NASA_DL_W1/best_model.pt): 0904c0fa86c0a47023e71e6d6eed39284cc1e1672fdd06b1705d6d9bd5e7fffe
- Wind C2 checkpoint (checkpoints/EXP_WIND_DL_C2/best_model.pt): 1a7f55723421fb60bddbb6084cd297616e2689c48782d6e386583d4c8329f9c5
- Demand D2 checkpoint (checkpoints/EXP_DEMAND_DL_D2/best_model.pt): fa733c9f6c4d0a35b21aa111f88e13f9aafd846e19a9e98e1fae95c23a884f62

## Conformal held-out coverage
- Wind C2, N=1,242: 80% nominal 84.22%; 90% nominal 92.83%.
- Demand D2, N=2,561: 80% nominal 66.03%; 90% nominal 85.47%. Undercoverage persists across the observed test months; diagnostics do not establish a single cause.

## Decision-support output after fixes

- cost_optimal_configuration: solar 150.0 kW, wind 0.0 kW, battery 200.0 kWh; renewable share 37.54%; grid import 7,271.02 kWh; modeled cost 1,121.14 USD; initial carry-in discharge 92.00 kWh; terminal SOC delta -80.00 kWh.
- grid_optimal_configuration: solar 150.0 kW, wind 300.0 kW, battery 200.0 kWh; renewable share 56.33%; grid import 5,056.18 kWh; modeled cost 1,390.33 USD; initial carry-in discharge 92.00 kWh; terminal SOC delta -80.00 kWh.

Pipeline command scripts/run_decision_support.py exited 0. Cost parameters are illustrative; this synthetic, index-paired scenario is not a site estimate.

## Phase 2 scientific checks — 2026-10-27

### NASA forecast score parity and nonnegative display projection

The app's historical metric cards use the registered, raw-output W1 test scores: N=1,242, MAE=16.2575 Wh/m², RMSE=30.1767 Wh/m², R²=0.9844. A full held-out replay reproduced those values when scoring raw predictions. The display path clips negative irradiance to zero; on the same full test partition, this projected series has MAE=15.4667 Wh/m², RMSE=30.1125 Wh/m², R²=0.9845, with 339/1,242 raw outputs below zero (27.29%). The dashboard now reports raw-score cards and separately labels the projected chart score, avoiding a silent metric mismatch. The projection improves these aggregate scores but must not replace the frozen raw-output benchmark.

### Demand interval diagnostics

| Test month | N | 80% target coverage | 90% target coverage | MAE (kWh) |
|---|---:|---:|---:|---:|
| 2017-09 | 353 | 69.12% | 84.42% | 2.7872 |
| 2017-10 | 744 | 63.84% | 84.81% | 2.6810 |
| 2017-11 | 720 | 65.42% | 86.39% | 2.4942 |
| 2017-12 | 744 | 67.34% | 85.75% | 2.5850 |

All four observed months fell below both nominal targets. Held-out residual autocorrelation was 0.1243 at lag 1 hour, 0.1488 at lag 24 hours, and 0.0753 at lag 168 hours. This is consistent with modest temporal dependence, but does not prove that dependence or seasonality caused undercoverage. No calibration or model tuning was performed on the test set.

### Matched-timestamp baseline checks

| Dataset/model | N and held-out period | Model MAE / RMSE / R² | Persistence MAE / RMSE / R² | Daily seasonal MAE / RMSE / R² |
|---|---|---|---|---|
| Wind C2 | 1,242; 2014-11-10 06:30–2014-12-31 23:30 | 0.5908 / 0.8216 / 0.7576 m/s | 0.6002 / 0.8679 / 0.7295 m/s | 1.2661 / 1.6807 / -0.0142 m/s |
| Demand D2 | 2,561; 2017-09-16 07:00–2017-12-31 23:00 | 2.6152 / 4.0673 / 0.9234 kWh | 2.5336 / 4.0823 / 0.9228 kWh | 6.7775 / 9.9977 / 0.5369 kWh |
| NASA W1 | 1,242; 2023-11-10 06:00–2023-12-31 23:00 | 16.2575 / 30.1767 / 0.9844 Wh/m² | 172.2827 / 296.0681 / -0.5027 Wh/m² | 172.2822 / 296.0324 / -0.5024 Wh/m² |

Wind C2 modestly beats persistence on all three metrics. Demand D2 has worse MAE than persistence, with slightly lower RMSE and slightly higher R²; do not claim blanket superiority. The NASA baseline comparison is on identical timestamps, but the all-hour aggregate includes many low/night observations and does not alone establish daytime operational value. A 168-hour weekly-naive baseline could not be evaluated under the model's 24-hour available-history contract and is not reported.

### Data completeness and cyclic battery check

The raw SKIPP'D directory contains the held-out test shard and only `train-00000-of-00005.parquet`; four training shards are absent. No full-data training reproducibility claim is supported. The optional cyclic terminal-SOC control was exercised in the dashboard: for 175 kW solar / 250 kWh storage / 240 hours, the current greedy dispatch ended at 25 kWh against the 125 kWh starting target and surfaced an infeasibility message. This check enforces/reports the terminal condition; it does not re-optimize the dispatch schedule to recover the starting charge.

## Phase 3 follow-up — 2026-10-27

The previous NASA baseline values near 172.28 MAE/296.07 RMSE are withdrawn and superseded by exact 24-hour-history comparisons in `baseline_comparison_report.md`; prior cause was not established. NASA raw test scores are 16.2575/30.1767/0.9844; projected 15.4668/30.1125/0.9845, N=1,242; 339 raw negatives (27.29%).

Demand fixed split remains undercovered (66.03%/85.47% at nominal 80%/90%). Validation-only prequential comparison favored rolling-168 among tested windows; its one-pass online test evaluation was 88.72% coverage at nominal 90%, width 11.5810 kWh, Winkler 20.3040, N=2,561. It is separate from the fixed split score and is not deployed in the dashboard.

240-hour synthetic cyclic dispatch reached 125.0 kWh target SOC from 125.0 kWh; unmet 0, no simultaneous charge/discharge, SOC residual 0.0. Full test suite: 215 passed, 5 failed; not a passing acceptance check.

## Phase 3 resumption correction and final verification — 2026-10-27

**Withdrawn:** The earlier one-pass demand paragraph reporting 88.72% coverage, 11.5810 kWh width, and 20.3040 Winkler as “rolling 168-hour” did not truncate residual history after initialization and therefore described an expanding window. Do not use those figures as rolling-window evidence. The fixed-window calculation and method are documented in `demand_uncertainty_analysis.md`.

**Correct rolling-168 result:** Starting with the latest 168 calibration residuals, each test interval is formed using only the current trailing 168 residuals; the current actual is appended only after that interval is generated. Full test N=2,561; valid=2,561; skipped=0; 90% coverage=90.16%; mean width=12.8404 kWh; Winkler=20.2323. Fixed split remains separately reported at 85.47% coverage, 9.5254 kWh width, and 20.6045 Winkler on those same rows. This is one prequential historical replay and does not provide a distribution-free future guarantee.

Final replay checks: NASA W1 raw N=1,242, MAE/RMSE/R²=16.26/30.18/0.9844 Wh/m²; physically projected scores=15.47/30.11/0.9845; raw negative predictions=339 (27.29%). The 240-hour cyclic MILP check (SciPy 1.17.1) ended at 125 kWh from 125 kWh, with 0 unmet demand, 0 SOC-balance residual, and no simultaneous charge/discharge. Final `python -m pytest -q`: 265 passed, 14 warnings, 0 failed.

