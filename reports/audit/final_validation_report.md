# Final Validation Report — Isolated Copy

## Phase 2 completion update — 2026-10-27

Working copy: `C:/Users/shahz/Downloads/DL-MiniProject-Fixed-20261027`, branch `codex/dashboard-redesign-phase2`, HEAD `92d5674685854c48d3b2070fbbb04922e49ad4ec`. This section supersedes the earlier baseline-only test count below. The original Desktop working tree was not modified. No commits, pushes, model training, package installs, dataset downloads, or checkpoint changes were made.

### Verified implementation and tests

- Dashboard now has eight plain-language sections with reusable forecast, uncertainty, and simulation explanations. Simulation values come from the active scenario and are compared with a same-horizon grid-only run. Independent locations/years and synthetic row pairing are disclosed.
- The solar-irradiance forecast cards use the frozen raw-output test metrics. The chart clips negative irradiance for physical display and labels the separate projected error. Full W1 replay: raw N=1,242, MAE=16.2575, RMSE=30.1767, R²=0.9844; projected N=1,242, MAE=15.4667, RMSE=30.1125, R²=0.9845; 339 raw predictions were negative.
- Monthly demand coverage remains below nominal in all test months. Residual autocorrelation is reported as a diagnostic, without claiming it proves the cause. Matched-timestamp NASA, wind, and demand baseline results and tradeoffs are recorded in `verified_metrics.md`.
- The cyclic SOC control displays infeasibility when the current greedy dispatcher cannot meet terminal SOC. Browser check at 175 kW solar / 250 kWh battery / 240 hours: terminal SOC 25 kWh vs. 125 kWh target. This option does not re-optimize dispatch.
- Automated checks cover absent model/data files, malformed and too-small image uploads, energy-balance constraints, and explanation behavior.
- `python -m py_compile app.py src/frontend/model_loaders.py tests/test_frontend.py`: passed.
- `python -m pytest -q`: **216 passed, 14 warnings**, exit 0. Warnings are Matplotlib/PyParsing deprecations; the earlier app invalid-escape warning was corrected.
- Streamlit was run from the isolated copy and its eight sections rendered in the browser. Solar, wind, demand, reliability, energy-mix controls, and methodology views were checked; capacity changes updated both summaries and results. Browser console error query returned no entries. Server output showed `use_container_width` deprecation notices and expected negative-output clipping diagnostics, no traceback observed.

### Known limitations after Phase 2

1. Demand interval coverage (66.03% at nominal 80%; 85.47% at nominal 90%) is still under target. The test months and autocorrelation suggest time dependence/distribution-change concerns but do not isolate a cause. No held-out test tuning was performed.
2. Cyclic SOC is a strict feasibility check over greedy dispatch; it does not search/re-dispatch to hit terminal SOC.
3. Four of five SKIPP'D training shards are absent. The full training benchmark is incomplete.
4. Selected Wind C2 is speed-only; C3/C4 degree direction features still need a versioned sine/cosine experiment and fair refit.
5. Demand D2 slightly underperforms persistence on MAE, despite slightly better RMSE/R². Avoid an unqualified model-superiority claim. NASA's high aggregate skill is all-hour and should be supplemented with daylight-focused evaluation before operational conclusions.
6. Model E combines independent sources by row position and remains a synthetic software benchmark, not real-site feasibility.
7. Streamlit emits `use_container_width` deprecation notices; this does not block runtime.

Run commands from the working-copy root:

```powershell
python -m streamlit run app.py
python -m pytest -q
```

The first Phase 1 audit record follows for historical context.

Date: 2026-10-27  
Working copy: C:/Users/shahz/Downloads/DL-MiniProject-Fixed-20261027  
Original: C:/Users/shahz/OneDrive/Desktop/DL MiniProject  
Copy branch: codex/isolated-audit-fixes; no commits or pushes.

## Changes and validation
All fixes and generated outputs are in the Downloads copy. The original was rechecked: branch main, HEAD 92d5674685854c48d3b2070fbbb04922e49ad4ec, and same five pre-existing modified/untracked entries.

- Full suite: 205 passed, 16 warnings, exit 0.
- Streamlit started from the copy; local page returned HTTP 200. Process stopped after the check.
- Frozen models, no training: Solar S3 N=14,003, MAE 2.3008 kW, RMSE 3.3736, R² 0.8052; NASA W1 inferred input=4/hidden=32/layers=1, N=1,242, MAE 16.2575 Wh/m², RMSE 30.1767, R² 0.9844; Wind C2 N=1,242, MAE 0.5908 m/s, RMSE 0.8216, R² 0.7576; Demand D2 N=2,561, MAE 2.6152 kWh, RMSE 4.0673, R² 0.9234.
- Uncertainty command succeeded: wind coverage 84.22%/92.83%; demand coverage 66.03%/85.47% at nominal 80%/90%.
- Decision-support command succeeded and regenerated the 120-candidate table and stress scenarios. Cost-optimum: 150 kW solar / 0 kW wind / 200 kWh battery, modeled cost $1,121.14, grid import 7,271.02 kWh, renewable penetration 37.54%. Carry-in discharge is priced at grid rate and excluded from renewable-origin energy.
- Syntax compilation and focused regression tests passed. No model training, package installation, data acquisition, commit, or push.

## Remaining issues and actionable follow-up
1. Demand uncertainty undercoverage: held-out coverage is 66.03% at nominal 80% and 85.47% at nominal 90% following seasonal shift. Evaluate rolling or season-aware calibration under a predeclared protocol; do not present nominal levels as guarantees on this distribution.
2. No cyclic terminal SOC: initial/final SOC and terminal delta are reported, and carry-in discharge is priced, but ending SOC is not constrained to equal starting SOC. Add a cyclic constraint or salvage/penalty before claiming repeatable daily operation.
3. Circular direction feature in C3/C4: selected C2 uses speed only. Encode degrees as sine/cosine, version feature/config metadata, and retrain/reevaluate C3/C4 before selecting them.
4. Independent data sources: Stanford solar, Mumbai irradiance, Gujarat wind, and Orlando demand are not co-located or time-aligned. Model E is a synthetic software benchmark, not site feasibility or investment evidence. Collect and audit aligned site data before site-specific claims.
5. Incomplete SKIPP'D training data: only one of five training shards is present. Acquire/audit missing shards and cross-shard/test day leakage before claiming benchmark completeness.
6. Split compatibility: generic split defaults to retained rows after embargo. NASA/wind/demand builders explicitly default to legacy_source to reproduce frozen checkpoints/scalers. New runs should choose and record retained after refitting scalers/retraining.
7. Baseline comparison scope: this audit reproduced point metrics but did not regenerate every baseline on identical test timestamps. Avoid test-win claims based only on validation baseline rows.

Warnings remaining: Matplotlib/PyParsing deprecations, one invalid escape warning in app.py, and pytest class-scoped fixture deprecation. See verified_metrics.md and fix_log.md for artifact fingerprints and finding dispositions.


## Phase 3 follow-up — 2026-10-27 (INCOMPLETE)

- Cyclic MILP 240-hour synthetic check completed in 0.714 s: start/end SOC 125.0 kWh, delta 0, unmet 0, no simultaneous charging/discharging, SOC range 25–125 kWh, SOC balance residual 0.0. Renewable share 50.54% of served demand, excluding 114.9981 kWh carry-in discharge.
- NASA target, units and inverse-scaling were checked. Raw/projected scores are separated and negatives are prevented downstream. Prior NASA baseline figures near 172.28 MAE are withdrawn; aligned results are in the baseline report.
- Demand uncertainty was decomposed by month and demand quartile. Validation-only rolling-window comparison selected 168 hours by Winkler score; one-pass sequential test result remains below nominal 90% coverage. This method is not wired into the dashboard.
- SKIPP'D inventory contains one of five train Parquets and one complete test Parquet. Four train shards are missing. Active C2 wind model is speed-only; C3/C4 configs use degree-valued direction linearly.
- Full `python -m pytest -q`: **215 passed, 5 failed, 17 warnings, 31.77 s**. Direct failures: undefined `q_90` in wind evaluator and undefined rolling variables in demand evaluator; three simulation/evaluator tests cascade from wind. Battery tests pass. Pytest could not write `.pytest_cache` because of Windows ACL.
- In-memory syntax compilation passed for 8 Python files. Streamlit/browser QA after this regression was not run.
- Automatic approval review rejected a source edit citing Desktop-project protection and test-outcome update risk. No alternate source-edit path was attempted. Normal Windows writes to the Downloads folder were denied; elevated writes were limited to requested Markdown audit files. No training, checkpoint/data changes, commit, or push.

Next: resolve approved source-edit boundary and interval protocol; repair undefined dashboard values; add targeted validation; run targeted and full tests; then syntax/app/browser QA. Add scenario validation for timestamps, values, capacities, aggregate renewable consistency, and metadata hours. Do not claim Phase 3 complete.

### Correction on the attempted source edit and browser check

The rolling-helper attempt partially changed the Downloads-copy source before the reviewer rejected the follow-up: `src/uncertainty/conformal.py` now contains an unintegrated rolling helper, and `src/frontend/model_loaders.py` contains incomplete interval references. These changes are not a valid integration and are the direct cause of the five failures above. I did not retry through a shell or another patch route after rejection. The exact reviewer rationale was that the retry targeted the protected Desktop project and that using test outcomes in the displayed interval path created leakage risk. User approval is needed to repair the specific source targets under a verified copy path and agreed evaluation protocol.

Streamlit server startup succeeded and printed its local URL. Browser page-level verification was not completed: the Computer Use tool reported that it was stopped by the user with the physical Escape key. The server was then stopped. No successful page rendering is claimed.

## Phase 3 resumption — final verified status — 2026-10-27

This section supersedes the earlier “INCOMPLETE / BLOCKED” Phase 3 follow-up and its 215-passed/5-failed regression results. Authorized repairs were completed only in `C:/Users/shahz/Downloads/DL-MiniProject-Fixed-20261027`. Baseline HEAD recorded before edits: `92d5674685854c48d3b2070fbbb04922e49ad4ec`. Desktop original and Git history were not modified. No training, checkpoint/data mutation, package install, data download, commit, or push occurred. Rollback copies of five edited source/test files are at `scratch/phase3-resumption-rollback-20261027/`.

### Repairs and verified outcomes

- Restored the wind fixed split conformal evaluation and its q90/metrics output.
- Completed a causal prequential rolling conformal helper and connected it to demand evaluation/dashboard charts. The calibration history remains a fixed trailing 168 residuals. Intervals precede outcome updates; missing forecasts yield missing intervals, and missing targets do not update residual history. The dashboard labels fixed and rolling metrics separately.
- Fixed-window demand test replay: N=2,561, 90.16% coverage, 12.8404 kWh mean width, Winkler 20.2323. Fixed split comparator on the same rows: 85.47%, 9.5254 kWh, 20.6045. The 88.72%/11.5810/20.3040 historical paragraph was an expanding-window calculation mislabeled as rolling and is withdrawn.
- NASA W1 evaluator replay: raw N=1,242, MAE/RMSE/R²=16.26/30.18/0.9844 Wh/m²; projected=15.47/30.11/0.9845; 339 raw outputs negative (27.29%).
- SciPy 1.17.1 cyclic dispatch synthetic check, 240 hours: initial/final SOC 125 kWh, unmet demand 0, SOC residual 0, no simultaneous charge/discharge.
- `python -m pytest -q`: **265 passed, 14 warnings, 0 failed (35.93 s)**. Warnings are Matplotlib/PyParsing deprecations. In-memory compilation passed for six changed runtime/test Python files.
- Streamlit server started and the dashboard landing page rendered with eight navigation sections. Page 5 and 6 interactions were not verified because the computer-control tool detected user input in Chrome and required stopping automation.

### Remaining limitations

Demand fixed split remains under nominal coverage; rolling performance is only a historical sequential replay and not a guarantee. Four SKIPP'D training shards are absent. Wind C2 is speed-only and C3/C4 retain a linear direction feature. Model E profiles remain independent and unsynchronized, so the scenario is synthetic and not site feasibility evidence.

## Phase 4 — final verification and acceptance status — 2026-10-28

Working copy remained `C:/Users/shahz/Downloads/DL-MiniProject-Fixed-20261027`; HEAD is still `92d5674685854c48d3b2070fbbb04922e49ad4ec`. The existing modified/untracked Git state was inspected and preserved; this phase made no source-code changes. The Desktop original was not touched. No training, downloads, package installation, checkpoint changes, commit, or push.

### Checks executed in Phase 4

- Required final command `python -m pytest -q`: **265 passed, 0 failed, 0 skipped, 14 warnings** (25.96 s). Warnings are Matplotlib/PyParsing deprecations.
- In-memory syntax compilation passed for nine relevant runtime/test files: `app.py`, frontend model loaders, conformal helper, dispatch/scenario/optimizer modules, and their regression tests.
- Started Streamlit from this copy. Local `/_stcore/health` returned **200 ok**; server was stopped afterward. This confirms startup/health, not page interaction.
- Reviewed the current rolling helper and its tests: intervals use the latest trailing 168 calibration residuals, are calculated before the current target residual is appended, skip missing outcomes, and do not use held-out outcomes to choose the window. The documented comparison selects the window on the validation tail; test results are a single subsequent replay.
- Read the current cyclic MILP and regression tests: each lexicographic solve checks proven success or explicit infeasibility; constraints encode demand and renewable balances, SOC recurrence/bounds, charge/discharge limits, grid limit, terminal SOC, and mutually exclusive charge/discharge. The full suite passed the cyclic feasible/infeasible and accounting tests.

### Results with prior direct replay versus Phase 4 reproduction

- The Phase 3 resumption directly executed the full demand and NASA frozen-checkpoint evaluators and the 240-hour cyclic check. Those recorded results remain: demand rolling-168 N=2,561, coverage 90.16%, width 12.8404 kWh, Winkler 20.2323; NASA raw N=1,242 MAE/RMSE/R²=16.26/30.18/0.9844 Wh/m² and projected=15.47/30.11/0.9845 with 339 negative raw outputs. Phase 4 inspected their code paths/reports and regression coverage but did not repeat NASA or the exact battery numeric replay.
- The exact input arrays and seed/profile recipe for the battery report's 240-hour numeric summary are not recorded there, so that exact metric row cannot be reconstructed from the report alone. Source and tests verify model constraints; the documented earlier 125 kWh terminal SOC, zero unmet energy and zero aggregate SOC residual are retained as prior direct-run evidence, not represented as a new Phase 4 replay.
- A Phase 4 direct full-demand replay exceeded the expected runtime without producing metrics and was stopped. The existing Phase 3 direct replay and full-suite demand regressions remain the available numerical evidence. No reported result was fabricated or hard-coded.

### Dashboard acceptance scope

The Phase 3 browser check rendered the landing page and showed all eight navigation labels: Project Overview, Solar PV Nowcasting, Solar Irradiance Forecasting, Wind Forecasting, Electricity-Demand Forecasting, Uncertainty Analysis, Energy-Mix Optimizer, and Methodology and Disclosures. Phase 4 confirmed the local app health endpoint, but did not navigate the pages or exercise controls/inference. Pages 5 and 6 had already been left unverified after the computer-control interruption; Phase 4 did not attempt to bypass that interruption. Therefore interaction criteria—page-level inference/input errors, rolling explanations, capacity changes, grid-only comparison, dispatch charts, and raw/projected NASA presentation—remain **unverified in browser**.

### Remaining limitations and readiness

The project is test-green and the dashboard starts, but this is not full user-acceptance sign-off. Demand fixed split coverage remains below the nominal 80%/90% targets (66.03%/85.47%); rolling coverage is one historical sequential replay, not a guarantee. Four SKIPP'D training shards are absent. Wind C2 is speed-only and C3/C4 use linear degree direction. NASA all-hours aggregate scores do not establish daylight operating performance. Energy-mix profiles are from independent, unsynchronized locations/periods and are a synthetic software demonstration, not site feasibility evidence. Battery report metrics lack the input profile recipe needed for exact replay.

Launch: `python -m streamlit run app.py`. Test: `python -m pytest -q`. No source fix was warranted in Phase 4.

