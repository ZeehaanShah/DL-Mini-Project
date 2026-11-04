# Demand Uncertainty Analysis — Phase 3

Frozen `EXP_DEMAND_DL_D2`; BDG2 Panther_education_Rosalie, Orlando; hourly kWh, one-hour horizon. No training. Scalers fit on training; chronological calibration/test with 24-hour embargo. Calibration residuals are calculated in kWh after matching target inverse-scaling. Split conformal uses `ceil((n+1)(1-alpha))`; intervals use inclusive `lower <= actual <= upper`, lower-clipped at zero; MPIW is mean upper-minus-lower and Winkler miss penalty is `2/alpha` times miss distance. No negative predictions occurred in the D2 calibration/test sets.

Calibration N=2,607, 2017-05-28 16:00–2017-09-14 06:00. Test N=2,561, 2017-09-16 07:00–2017-12-31 23:00.

| Fixed split nominal | Coverage | Mean width kWh | Winkler | N |
|---:|---:|---:|---:|---:|
| 80% | 66.03% | 4.9561 | 15.4839 | 2,561 |
| 90% | 85.47% | 9.5254 | 20.6045 | 2,561 |

90% monthly coverage: Sep 84.42% (N=353), Oct 84.81% (744), Nov 86.39% (720), Dec 85.75% (744). By actual demand quartile low-to-high: 82.63%, 88.73%, 89.32%, 81.33% (N=662,621,646,632); upper quartile high-side misses 14.56%. Point MAE/RMSE/R2=2.6152/4.0673/0.9234. Residual ACF lag 1/24/168=0.1243/0.1488/0.0753. Descriptive only; not causal proof.

## Validation-only prequential window comparison

First 70% of calibration predictions seed the chronological validation tail (N=783); each interval uses prior residuals only, then updates after outcome. Candidate selection used lowest validation Winkler at both 80% and 90%:

| Method | 80% coverage / width / score | 90% coverage / width / score | N |
|---|---|---|---:|
| Expanding | 80.72% / 5.1389 / 10.2391 | 89.53% / 9.7311 / 13.4496 | 783 |
| Rolling 168 h | **82.89% / 5.9867 / 10.1498** | **92.72% / 10.4135 / 13.0160** | 783 |
| Rolling 336 h | 82.38% / 5.8630 / 10.1889 | 91.57% / 10.5363 / 13.1550 | 783 |
| Rolling 720 h | 81.74% / 5.5885 / 10.3429 | 90.80% / 10.4747 / 13.5445 | 783 |

### Correction to the earlier Phase 3 test calculation

The earlier attempted held-out calculation did not truncate residual history after the initial 168 calibration observations, so it became expanding instead of rolling. Its rolling-window metric paragraph is withdrawn and must not be used. The window choice itself was selected from the validation comparison above, without test selection.

### One-pass rolling-168 evaluation after the configuration was fixed

The corrected evaluator begins with the latest 168 calibration residuals. At each hourly test timestamp it calculates bounds from only the current trailing 168 residuals, then appends that target's absolute error for use at subsequent timestamps. There is no warm-up loss because all 168 initial residuals are required before the test begins. Evaluation partition: N=2,561, 2017-09-16 07:00 through 2017-12-31 23:00; valid intervals 2,561; skipped 0; nominal 90%.

| Method | Coverage | Mean width (kWh) | Winkler score | N |
|---|---:|---:|---:|---:|
| Fixed split baseline, same rows | 85.47% | 9.5254 | 20.6045 | 2,561 |
| Rolling 168-hour prequential | 90.16% | 12.8404 | 20.2323 | 2,561 |

Rolling 90% coverage by test month was 86.40% (Sep, N=353), 90.73% (Oct, N=744), 90.83% (Nov, N=720), and 90.73% (Dec, N=744). The intervals are wider than fixed split; the test-level result is one sequential historical replay and gives no future-time or exchangeability guarantee. No window selection or tuning was performed on these test results.

The dashboard now shows both methods distinctly. Its demand page uses rolling intervals for the chart, derives displayed metrics from the full evaluator output, and labels the result as prequential; the fixed split cards on the uncertainty page remain the static baseline.
