# Fix Log

## 2026-10-27 — Isolated copy audit and repair

| Finding | Disposition | Main files |
|---|---|---|
| Dashboard/report metric drift and stale stress values | Fixed: model cards read completed test registry rows, uncertainty cards read generated JSON, and the stress chart reads generated scenario JSON. Corrected stale Model A/B and Stage 5 report values; reproduced all four models. | app.py, src/frontend/model_loaders.py, reports, registry outputs |
| Battery carry-in misclassified and unpriced | Fixed: renewable-origin SOC/discharge tracked separately, carry-in discharge priced at grid rate, SOC residual and initial/final/delta reported. Phase 2 adds an optional strict cyclic terminal-SOC feasibility check; the current greedy dispatcher does not re-optimize to satisfy the target. | src/decision_support/energy_balance.py, src/decision_support/optimizer.py, app.py, tests |
| Time-series forecasts may violate nonnegative domain | Fixed at inference: reject non-finite predictions, log and project negative predictions to zero, retain raw predictions and count, reject materially negative labels while tolerating inverse-scaler round-off. No architecture changes or retraining. | src/frontend/model_loaders.py, tests |
| NASA evaluator architecture mismatch | Fixed: infer hidden dimension and layer count from checkpoint tensors; mismatching overrides fail. Default W1 run reproduced. | scripts/evaluate_test_set.py, README.md |
| Chronological splitter ignored test_ratio | Fixed in generic splitter: retained policy apportions requested fractions across rows after embargo. Added legacy_source policy; current dataset builders select it to preserve frozen checkpoint/scaler compatibility. | src/data/splits.py, src/data/datasets.py, src/data/wind.py, src/data/demand.py, tests |
| Non-hour timestep conversion | Fixed: generation and inverter energy scale by interval duration; optimizer and uncertainty calls propagate dt; finite input checks and regression tests added. | src/decision_support/scenarios.py, src/decision_support/optimizer.py, src/decision_support/uncertainty_eval.py, tests |
| Conformal claims overstate validity under demand shift | Fixed in UI and summaries: exchangeability condition disclosed, coverage is sourced from JSON, seasonal undercoverage identified. Remaining: test coverage 66.03%/85.47%. | app.py, reports |
| Synthetic Model E described as synchronized | Fixed wording: independent profiles are paired by row index; synthetic-only scope is clear. | app.py, src/frontend/model_loaders.py, src/decision_support/scenarios.py |
| Missing run guide / stale project status | Fixed: README and audit/project status docs added or refreshed with run evidence. | README.md, PROJECT_CONTEXT.md, CHANGELOG.md, reports/audit |
| Human-centred dashboard and result explanations | Implemented: eight user-facing sections use clearer navigation, plain-language results, data-derived simulations, same-horizon grid-only comparisons, visible synthetic-data scope, and uncertainty/sample-count disclosures. Added reusable explanation helpers and user-facing tests. | app.py, src/frontend/explanations.py, src/frontend/charts.py, tests |
| NASA dashboard metric mismatch after physical projection | Fixed: benchmark metric cards remain scored on raw held-out outputs; nonnegative display projection now has separately named projected MAE/RMSE. Full test replay found 339/1,242 (27.29%) negative raw predictions, so the distinction is disclosed on the forecast page. | src/frontend/model_loaders.py, app.py, tests/test_frontend.py |
| Demand undercoverage causal wording | Corrected: monthly coverage and residual autocorrelation are documented; visible disclosure states that dependence/distribution change are possible explanations, not proven causes. No test-set tuning was done. | app.py, reports/audit/verified_metrics.md |
| Cyclic SOC option and greedy dispatch limitation | Implemented and browser-checked strict terminal-SOC feasibility reporting. Current dispatch is greedy and reports infeasible rather than re-optimizing to hit the starting SOC; the limitation is explicit. | app.py, src/decision_support/energy_balance.py, src/frontend/model_loaders.py, tests |
| Methodology equation rendered with invalid escape sequences | Fixed conformal-equation HTML to use a raw string with valid LaTeX escape sequences. | app.py |
| Fair baseline comparison gap | Added same-test-window comparisons for NASA W1, Wind C2, and Demand D2; demand's MAE tradeoff and NASA's all-hour/daylight limitation are stated. Weekly-naive demand was not evaluated because 168 hours of lookback are outside the model's 24-hour input contract. | reports/audit/verified_metrics.md, reports/audit/final_validation_report.md |
| Circular wind direction in C3/C4 | Deferred: selected C2 is univariate. C3/C4 need feature-contract/config migration and matched retraining/reevaluation before use. | Future feature/data migration |
| Missing four SKIPP'D train shards and no co-located sources | Open research/data limitation: acquire and audit missing shards and co-located profiles before benchmark-completeness or real-site claims. | Data/report follow-up |


## 2026-10-27 — Phase 3 validation note

The cyclic MILP reached a 125 kWh terminal SOC in a 240-hour synthetic check (0 unmet demand, no simultaneous charge/discharge). The demand validation-only comparison favored a rolling 168-hour residual window, but the dashboard integration edit was rejected by the automatic reviewer because of the protected Desktop-project path and test-outcome update risk. The full suite currently has 215 passed and 5 failed: the wind evaluator has an undefined q_90, the demand evaluator references undefined rolling interval variables, and three downstream integration tests cascade from the wind failure. The former NASA baseline values around 172.28 MAE are withdrawn; exact 24-hour model-history baselines are documented in baseline_comparison_report.md. No Phase 3 completion claim is made.

**Correction to Phase 3 note:** The attempted rolling helper was partially written into `src/uncertainty/conformal.py` and `src/frontend/model_loaders.py` before the follow-up was rejected. The loader integration is incomplete and causes the reported test regression. Do not assume the dashboard source was unchanged. No source repair was attempted after reviewer rejection.

## 2026-10-27 — Phase 3 resumption repairs and verification

| Finding | Disposition | Main files |
|---|---|---|
| Wind dashboard evaluator referenced an undefined fixed conformal quantile | Fixed: compute q90 from the calibration residuals, form same-test-row 80%/90% intervals, and return interval metrics. | `src/frontend/model_loaders.py`, `tests/test_frontend.py` |
| Demand uncertainty dashboard lacked a causal rolling result and mixed interval interpretations | Fixed: fixed split metrics remain available separately; 90% chart intervals use a trailing 168-residual prequential window. Each interval is issued before that timestamp's target residual updates the window; missing outcomes do not update it. | `src/uncertainty/conformal.py`, `src/frontend/model_loaders.py`, `app.py`, tests |
| Prior “rolling” test numbers were actually from an expanding history | Corrected and withdrawn: 88.72%/11.5810/20.3040 are not fixed-window evidence. Correct rolling-168 full-test result is 90.16% coverage, 12.8404 kWh width, Winkler 20.2323, N=2,561. | `reports/audit/demand_uncertainty_analysis.md`, `verified_metrics.md`, `final_validation_report.md`, `PROJECT_CONTEXT.md` |
| Regression acceptance and app startup needed re-verification | Verified: full suite 265 passed/14 warnings/0 failed; in-memory syntax compilation passed; Streamlit landing page rendered. Browser interactions for pages 5 and 6 remain unverified. | tests, app, audit reports |

