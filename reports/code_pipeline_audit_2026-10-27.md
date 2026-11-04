# Code and Pipeline Audit — 2026-10-27

## Scope and method

Read-only static review of the current project tree, project context, supplied implementation brief, data/configuration code, model/training/evaluation entry points, frontend, decision-support modules, reports, experiment registry, saved histories, and asset availability. Existing CSV/JSON artifacts and checkpoint metadata were inspected. No training, test suite, application launch, or code execution was performed. Binary checkpoints and all diagnostic image pixels were not independently re-evaluated.

The checkout already contained user edits in `requirements.txt` and `src/frontend/model_loaders.py` at audit start. They were preserved. No code, data, checkpoints, Git metadata, commits, or branches were changed.

## Executive assessment

The project has working-looking implementations for four separate prediction tasks, saved checkpoints for all four, and valid-looking data pipelines for the available datasets. The saved reports and selected point metrics are not consistently synchronized, however, and the decision-support battery accounting has a material attribution issue. Treat the Streamlit metric cards and combined feasibility figures as unverified until the results are regenerated from one reproducible evaluation path.

The data sources remain geographically and temporally independent: Stanford SKIPP'D solar images, NASA POWER Mumbai weather, Gujarat wind, and Orlando building demand. Model E labels its composite scenario synthetic, which is appropriate, but its outputs cannot support a real location-specific feasibility claim.

## Findings and actionable fixes

### P1 — Results and dashboard values do not have one consistent source of truth

**Evidence**

- The saved registry row for `EXP_SOLAR_DL_S3_TEST` reports 14,003 samples, MAE 2.3008 kW, RMSE 3.3736 kW, and R² 0.8052. The Streamlit cards show MAE 2.301, RMSE 3.766, and a 3.842 kW persistence comparator. `reports/stage3_forecasting_summary.md` instead says 13,998 samples and 1.13 kW MAE. The data audit and registry agree on 2.30 kW MAE and 14,003 rows.
- The saved registry row for `EXP_NASA_DL_W1_TEST` reports 1,242 samples, MAE 16.2575, RMSE 30.1767, R² 0.9844 (Wh/m²). The dashboard shows MAE 16.26 but RMSE 30.68 and R² 0.989; it also displays a 112.43 Wh/m² seasonal-naive baseline not found in the registry. The Stage 3 summary says h=64, MAE 18.57, and 1,290 test windows. The registry identifies W1 (h=32) and 1,242 test windows.
- For current NASA data (8,760 hourly rows), the implemented 24-hour embargo split produces 1,290 validation windows and 1,242 test windows. The test-set evaluator docstring and Stage 3 report call 1,290 the test count; the registry’s 1,242 is consistent with the current builder.
- The Stage 5 report claims 82.16% demand coverage at nominal 90%; `reports/tables/uncertainty_summary.json`, the dashboard, and the uncertainty report say 85.47%. The saved 80% demand coverage is 66.03%.
- The dashboard’s fixed conservative stress row (renewable 2,795.3 kWh and grid 10,136.3 kWh) does not match the saved `decision_support_summary.json` conservative scenario (renewable 4,299.8928 kWh and grid 8,765.2138 kWh).

**Impact:** Users can be shown incorrect model quality, selected architecture, baseline advantage, sample count, or optimizer stress results. Some existing conclusions are plausible, but the repository does not currently establish all displayed values from the same run.

**Fix plan:** Treat registry rows plus a freshly reproducible evaluation as canonical; regenerate the dashboard cards, report tables, stress-scenario table, and plot labels from those outputs instead of hand-maintained literals. Add dataset/checkpoint/config identifiers and evaluation timestamp to each generated report. Keep a separate validation baseline comparison and held-out test baseline comparison, with identical timestamps and sample counts. Correct or remove stale statements (especially 1.13 kW, 13,998, 18.57, 1,290-as-test, 82.16%, and the unmatched stress row).

### P1 — Battery energy is free at the start and renewable penetration can misattribute it

**Evidence:** `BatteryConfig` defaults to 50% initial SOC. Dispatch can discharge that inventory immediately, but `renewable_penetration` counts all battery discharge as renewable energy. The simulator checks demand and renewable-side conservation, but does not report the initial/final stored-energy delta as a separate source or enforce a terminal SOC condition.

**Impact:** A battery can supply energy that was present before the evaluated window without charging it to the scenario, and that energy is counted as renewable. Ending with a depleted battery can make a short-horizon configuration look better than a sustainable repeatable system. The stored optimizer JSON has zero unmet demand, but that alone does not validate storage attribution or repeatability.

**Fix plan:** Make initial SOC an explicit scenario input and either start at minimum SOC or account for initial stored energy as a non-renewable carry-in. Add terminal/cyclic SOC requirements or include the value/cost of the ending SOC difference. Report direct renewable supply, battery discharge, initial carry-in, and renewable-origin battery energy separately. Add accounting checks for SOC change and efficiency losses before relying on the mix results.

### P1 — Time-series model outputs are unconstrained despite nonnegative physical targets

**Evidence:** Wind, demand, and weather LSTM heads are linear. The data validators and downstream physical conversion/dispatch APIs reject negative target-domain inputs; solar-image inference clips predictions at zero, but the time-series inference path does not apply equivalent range handling.

**Impact:** An out-of-range model prediction can result in negative displayed demand/wind/irradiance or make the Model E path fail validation. Existing saved aggregate metrics do not demonstrate that every future input will avoid this case.

**Fix plan:** Define output-domain handling explicitly per target. Prefer a nonnegative output transform or a documented inference-time constraint with validation and diagnostics; if changing architecture, preserve old-checkpoint compatibility or retrain intentionally. Bound wind speed by a defensible data/physical range. Test negative, NaN, and infinite predictions through the end-to-end conversion and dispatch path.

### P1 — Model E is a synchronized synthetic composite, not a co-located installation

**Evidence:** `build_synthetic_hybrid_benchmark_scenario` takes the first N rows from independent solar, wind, and demand arrays and pairs them by index. Those arrays represent Mumbai 2023, Gujarat 2014, and Orlando 2016–2017. The code and reports disclose this and mark provenance synthetic.

**Impact:** The energy balance and optimizer are useful for software demonstrations, but their renewable penetration, storage adequacy, and cost savings are not estimates for any real site. Pairing row zero across unrelated years/seasons does not preserve joint weather-demand behavior or correlation.

**Fix plan:** Keep the current path explicitly labeled as synthetic and avoid site-specific or investment claims. For real feasibility, acquire co-located or defensibly aligned irradiance, wind, load, and time-zone data; audit common timestamps and units; then re-run dispatch and optimization on that aligned series. Preserve dependence structure when constructing uncertainty scenarios.

### P2 — NASA evaluator defaults can instantiate the wrong architecture for the saved selected checkpoint

**Evidence:** `EXP_NASA_DL_W1_TEST` is the recorded selected test checkpoint (hidden size 32, one layer). `scripts/evaluate_test_set.py` defaults to hidden size 64 and two layers. Loading W1 without overriding both CLI values will create an incompatible model before loading the state dict.

**Fix plan:** Store model architecture/config metadata in the checkpoint and reconstruct from it, or require and validate an explicit matching config. Make the command help show the exact selected checkpoint/config pair. Do not silently default to a different architecture.

### P2 — Generic chronological splitter ignores the requested test ratio when allocating rows

**Evidence:** `split_timeseries_chronological` asserts that train/validation/test ratios sum to one, calculates train and validation lengths, inserts two embargoes, then assigns every remaining row to test. The `test_ratio` argument does not otherwise determine the test length.

**Impact:** The documented ratios are approximate pre-embargo proportions, not exact final partition proportions. Changing `test_ratio` while preserving the sum has no effect. Existing counts can be reproduced for current inputs, but config fields overstate control.

**Fix plan:** Decide whether ratios refer to pre-embargo allocation or final retained partitions. Implement and document that rule, validate nonempty partitions and expected sizes, and record exact post-embargo row/window counts and timestamp ranges in each run artifact.

### P2 — Circular wind direction is fed as a linear scalar in multivariate experiments

**Evidence:** C3/C4 list `wind_direction` in degrees as a direct feature. Directions near 0° and 360° are physically close but numerically far apart.

**Impact:** The multivariate models can learn an artificial discontinuity at north, reducing reliability or making the extra-feature comparison misleading.

**Fix plan:** Encode direction as sine/cosine (and update `input_dim` and config); compare on the same validation and held-out test periods. Keep old checkpoint/config combinations intact if they remain supported.

### P2 — Non-hourly timestep arguments are not reflected in resource conversion outputs

**Evidence:** The conversion functions return hourly kWh assuming a one-hour interval, while optimizer and dispatch APIs accept arbitrary `time_step_hours`. Dispatch scales the grid and battery limits by dt, but solar/wind generation conversion does not scale by dt.

**Impact:** Non-1-hour runs mix energy from a 1-hour generation profile with a different interval duration.

**Fix plan:** Either enforce a one-hour input contract throughout and reject other values, or pass timestep duration into both generation conversion functions and scale kWh consistently. Verify units at conversion and dispatch boundaries.

### P2 — Some claims overstate conformal validity under observed time-series shift

**Evidence:** The app contains generic text saying conformal intervals have guaranteed finite-sample/marginal coverage. The code’s conformal module correctly documents exchangeability assumptions, but the demand test set empirically covers only 66.03% at nominal 80% and 85.47% at nominal 90% after a seasonal shift.

**Impact:** A reader may interpret the displayed interval as a guarantee on this winter test distribution or the full 240-hour horizon.

**Fix plan:** Present nominal level, calibration period, test coverage, and known shift together. State the exchangeability condition next to the guarantee, avoid calling the demand intervals valid on this test period, and retain the marginal-versus-joint disclosure. Consider rolling or season-aware calibration only after a pre-registered evaluation protocol.

### P3 — Solar training data are only one fifth of the available train partition

**Evidence:** The current solar train file is shard 1 of 5 (69,875 rows); the other four train shards are not present in the noted dataset inventory. The test parquet is present with 14,003 rows.

**Impact:** This is adequate for a limited proof of concept but weakens generalization and benchmark claims. A held-out test parquet does not correct training-data incompleteness.

**Fix plan:** Acquire and audit the remaining official training shards if permitted, verify duplicate days/timestamps and leakage across all train shards and test, then train/select using documented full-data splits. Until then, state the shard limitation prominently and do not claim benchmark comparability.

### P3 — Project guidance and runtime metadata are stale/inconsistent

**Evidence:** `PROJECT_CONTEXT.md` describes older dependency versions/status and some earlier validation counts; the supplied brief and current `.venv` metadata report torch 2.11.0+cpu, torchvision 0.26.0+cpu, and Streamlit 1.66.1. The root has no README, despite project instructions naming it as the first run guide. Existing report headers claim different test counts (140, 162, 177, 192) at different stages.

**Fix plan:** Refresh the context to match the current environment and distinguish historical test runs from current verification. Add a short root run guide with environment activation, app launch, training/evaluation commands, expected inputs, and output paths. Do not present old test counts as a current passing status.

## Functionality and assets observed

- Checkpoint and `history.json` files were present for all 16 listed experiments across solar, NASA weather, wind, and demand.
- Processed wind data were present with 8,760 rows; processed demand data were present with 17,544 rows and no missing `demand_kwh` values.
- NASA POWER, SKIPP'D, wind, and BDG2 files are present in the repository inventory; the raw NASA CSV includes a metadata header and should be read through its project validator rather than a naive `read_csv` assumption.
- The `requirements.txt` and `src/frontend/model_loaders.py` edits predated this audit. Current `.venv` metadata matches the Torch/Torchvision versions in the edited requirements file.
- Energy balance equations account for per-step demand and renewable inflow, and the solar image predictor clips negative PV output. These are useful foundations; battery initial/terminal state attribution remains unresolved.

## Recommended order of work

1. Reconcile all test metrics and counts from the existing frozen checkpoints, dataset builders, and saved registry; regenerate dashboard/report artifacts from one output source.
2. Correct battery SOC/carry-in attribution and terminal SOC handling; re-evaluate optimizer comparisons.
3. Constrain and validate point predictions before presenting them or passing them to Model E.
4. Fix evaluator architecture reconstruction and define exact split-ratio semantics.
5. Rephrase conformal claims and preserve synthetic provenance; then improve circular wind features and non-hour timestep handling.
6. Decide whether to acquire missing SKIPP'D training shards and co-located data before expanding real-world claims.

## Verification limits

No automated tests, model evaluations, Streamlit launch, checkpoint reload, or training run were performed for this audit. Existing saved metrics are reported as artifacts, not independently reproduced here. No code/data/checkpoint edits or Git operations that alter history were made.
