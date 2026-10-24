# PROJECT_CONTEXT.md — Persistent Project Memory

**Last updated:** 2026-10-24T16:25 IST (Stage 8 UI/UX Accessibility & Readability Repair Completed; Presentation-Only Changes; Not Runtime-Tested per User Instruction)

---

## 1. Project Overview

| Field | Value |
|---|---|
| **Title** | A Multimodal, Uncertainty-Aware Decision-Support Framework for Location-Specific Renewable Energy Feasibility and Energy-Mix Optimization |
| **Type** | Academic Deep Learning Mini-Project |
| **Objective** | Multimodal, uncertainty-aware decision-support PoC for renewable energy forecasting |
| **Status** | Project Complete; Interactive Multi-Page Streamlit Dashboard (`app.py`); Models A–E Fully Integrated; 192 Passed Tests (100%) |

### Research Objectives
1. Demonstrate CNN transfer learning on sky images for solar PV output forecasting (SKIPP'D)
2. Build weather/radiation forecasting from NASA POWER hourly data
3. Wind speed forecasting (pending data acquisition)
4. Electricity demand forecasting (pending data acquisition)
5. Decision-support module combining compatible predictions with uncertainty

### Key Constraint
Datasets are from independent sources (Stanford, NASA POWER, NREL, BDG2). **Do NOT claim** they describe the same location, timestamp, or physical installation unless verified.

---

## 2. Environment

| Component | Value | Status |
|---|---|---|
| OS | Windows 10 (10.0.26200), AMD64 | VERIFIED |
| Python | 3.11.9 (system install, no venv) | VERIFIED |
| PyTorch | 2.14.1+cpu | VERIFIED |
| torchvision | 0.29.1+cpu | VERIFIED |
| CUDA/GPU | Not available (CPU only) | VERIFIED |
| pandas | 2.2.2 | VERIFIED |
| NumPy | 2.4.6 | VERIFIED |
| scikit-learn | 1.5.1 | VERIFIED |
| PyArrow | 17.0.0 | VERIFIED |
| Pillow | 10.4.0 | VERIFIED |
| matplotlib | 3.9.2 | VERIFIED |
| PyYAML | 6.0.2 | VERIFIED |
| SciPy | 1.17.1 | VERIFIED |
| tqdm | 4.67.3 | VERIFIED |
| Disk | 574.2 GB free / 927.1 GB | VERIFIED |
| RAM | ~15.7 GB total | VERIFIED |
| Git | Initialized (new repo, no commits) | VERIFIED |

### Important Notes
- **CPU-only training**: All experiments will run on CPU. Batch sizes and model complexity must account for this.
- **Windows PyTorch DLL registration**: Handled automatically in `conftest.py` via `os.add_dll_directory` for reliable pytest execution.
- **No virtual environment**: Using system Python. Recommend creating a venv for isolation.

---

## 3. Repository Structure

```
DL MiniProject/
├── AGENTS.md                    # Agent instructions
├── PROJECT_CONTEXT.md           # This file
├── CHANGELOG.md                 # Change log
├── app.py                       # Master Streamlit Interactive Dashboard
├── conftest.py                  # Pytest configuration & Windows DLL loader
├── .gitignore                   # Git exclusions
├── requirements.txt             # Environment dependencies (including streamlit, plotly)
├── checkpoints/                 # Checkpoints for Models A, B, C, D (16 experiments)
├── configs/
│   ├── skippd_solar.yaml        # SKIPP'D nowcasting dataset & loader config
│   ├── nasa_weather.yaml        # NASA POWER sequence forecasting config
│   ├── wind_resource.yaml       # NREL wind dataset & loader config
│   ├── demand_resource.yaml     # BDG2 demand dataset & loader config
│   ├── decision_support.yaml    # Stage 5 decision-support configuration
│   └── experiments/             # 16 individual experiment hyperparameter configs
├── data/
│   ├── raw/
│   │   ├── skippd/              # SKIPP'D solar dataset
│   │   ├── nasa_power/          # NASA POWER hourly 2023 Mumbai
│   │   ├── wind/                # NREL India Wind Toolkit Gujarat
│   │   └── bdg2/                # BDG2 Panther_education_Rosalie Orlando
│   ├── processed/
│   │   ├── wind/wind_hourly_pilot.csv
│   │   └── demand/demand_hourly_pilot.csv
│   └── sample_images/           # 4 verified real sky test images for interactive inference
├── experiments/
│   └── experiment_registry.csv  # 19-column reproducible experiment tracking registry (51 entries)
├── reports/
│   ├── figures/                 # 112 diagnostic, trajectory, and calibration plots
│   ├── tables/                  # uncertainty_summary.json, energy_feasibility_scenarios.csv, decision_support_summary.json
│   ├── data_audit.md            # Comprehensive data audit report
│   ├── leakage_audit.md         # Leakage risk register and verification
│   ├── baseline_report.md       # Empirical baseline benchmark report
│   ├── stage2_pipeline_report.md# Stage 2 pipeline validation report
│   ├── stage3_forecasting_summary.md # Stage 3 sequence forecasting summary
│   ├── uncertainty_report.md    # Stage 4 Conformal Prediction uncertainty report
│   └── stage5_decision_support_report.md # Stage 5 Decision Support & Energy-Mix Optimization report
├── scripts/
│   ├── check_env.py             # Environment audit
│   ├── audit_data.py            # Data audit (main)
│   ├── audit_skippd_deep.py     # SKIPP'D deep audit
│   ├── verify_skippd_alignment.py  # Timestamp alignment checks
│   ├── evaluate_baselines.py    # Baseline model evaluation & registry logger
│   ├── train_stage3_models.py   # Master Stage 3 training and test evaluation
│   ├── evaluate_uncertainty.py  # Stage 4 Split Conformal Prediction pipeline
│   └── run_decision_support.py  # Stage 5 Master Decision Support & Optimization orchestration
├── src/
│   ├── __init__.py              # Package init with early Windows PyTorch DLL registration
│   ├── data/
│   │   ├── splits.py            # Leakage-safe splitting & windowing
│   │   ├── datasets.py          # PyTorch Datasets & DataLoader builders
│   │   ├── wind.py              # NREL wind ingestion, validator, TurbinePowerCurve
│   │   └── demand.py            # BDG2 demand ingestion & meter validator
│   ├── decision_support/        # Stage 5 Model E module
│   │   ├── __init__.py          # Public exports
│   │   ├── energy_balance.py    # Physical energy dispatch, BESS dynamics, strict conservation check
│   │   ├── scenarios.py         # Provenance schemas, STC solar conversion, turbine proxy
│   │   ├── optimizer.py         # Deterministic grid-search optimizer, CostModel with horizon amortization
│   │   ├── uncertainty_eval.py  # Conformal uncertainty stress testing & joint coverage disclosure
│   │   └── plotting.py          # Publication figure generators
│   ├── evaluation/
│   │   ├── metrics.py           # Regression metrics (MAE, RMSE, R2)
│   │   └── registry.py          # ExperimentRegistry tracker
│   ├── frontend/                # Streamlit UI modules & cached loaders
│   │   ├── __init__.py
│   │   ├── model_loaders.py     # Cached PyTorch inference & evaluation data accessors
│   │   ├── charts.py            # Interactive Plotly chart builders
│   │   └── components.py        # Reusable styling & provenance callouts
│   ├── models/
│   │   ├── demand/              # DemandLSTM sequence models & baselines
│   │   ├── nasa_weather/        # WeatherLSTM & baselines
│   │   ├── solar/               # ResNet18 nowcaster & baselines
│   │   └── wind/                # WindLSTM sequence models & baselines
│   ├── uncertainty/
│   │   ├── __init__.py
│   │   ├── conformal.py         # SplitConformalCalibrator, metrics, finite-sample adjustment
│   │   └── plotting.py          # Calibration curve & trajectory interval plotting
│   └── utils/
│       ├── __init__.py
│       └── torch_loader.py      # Windows PyTorch DLL compatibility helper
└── tests/
    ├── __init__.py
    ├── test_baselines.py        # Baseline models, metrics & registry tests (15 tests)
    ├── test_datasets.py         # PyTorch dataset & loader pipeline tests (19 tests)
    ├── test_decision_support.py # Stage 5 energy conservation, BESS, optimizer, uncertainty (15 tests)
    ├── test_demand_pipeline.py  # BDG2 demand ingestion & model tests (19 tests)
    ├── test_frontend.py         # Streamlit loaders, inference, charts & app integrity (15 tests)
    ├── test_leakage.py          # Leakage prevention & embargo tests (10 tests)
    ├── test_nasa_integrity.py   # NASA POWER schema, quality, coverage (21 tests)
    ├── test_skippd_integrity.py # SKIPP'D schema, quality, temporal (19 tests)
    ├── test_uncertainty.py      # Conformal prediction & interval tests (15 tests)
    └── test_wind_pipeline.py    # Wind ingestion, turbine & model tests (44 tests)
```

---

## 4. Dataset Audit Findings

### 4.1 SKIPP'D Solar Dataset

| Property | Value | Status |
|---|---|---|
| Source | HuggingFace `torchgeo/skippd` (Parquet) | VERIFIED |
| Original source | Stanford SLAC, Nie et al. | VERIFIED |
| Location | Stanford, California, USA (30 kW rooftop array) | VERIFIED (Nie et al.) |
| Train parquet | `train-00000-of-00005.parquet`, 69,875 rows, **shard 1 of 5** | VERIFIED |
| Test parquet | `test-00000-of-00001.parquet`, 14,003 rows, **complete** | VERIFIED |
| Total available | 69,875 train + 14,003 test = 83,878 rows | VERIFIED |
| Full trainval size | ~349,372 (inferred from npy; 4 shards missing) | INFERRED |
| Schema | 3 fields: `image` (struct{bytes,path}), `time` (timestamp[ms, tz=US/Pacific]), `pv` (float32) | VERIFIED |
| Image format | 64×64 RGB PNG, embedded as bytes in struct | VERIFIED |
| PV target | Float32, range [0.00016, 29.56] kW, verified from 30 kW system | VERIFIED (kW) |
| Temporal resolution | Mostly 1-minute, variable (10s gaps to 63-day gaps) | VERIFIED |
| Date range (train shard) | 2017-03-09 to 2017-09-04 (92 calendar days) | VERIFIED |
| Date range (test parquet) | 2017-06-24 to 2019-10-19 (20 calendar days) | VERIFIED |
| Shared days (train ↔ test) | 0 days (0 exact timestamps) | VERIFIED |
| Missing values | None (0 NaN in PV, 0 nulls) | VERIFIED |
| Negative PV | None | VERIFIED |
| Duplicates (timestamps) | 0 in train shard, 0 in test parquet | VERIFIED |
| Timezone | US/Pacific (parquet); naive (npy, non-monotonic) | VERIFIED |

#### CRITICAL FINDINGS — SKIPP'D

1. **Target Task Definition (Contemporaneous Nowcasting, Horizon $H = 0$)**:
   - The dataset in this workspace contains the official **nowcast** partition (single 64×64 sky image at time $t$ mapped to contemporaneous PV output in kW at time $t$).
   - This was verified from TorchGeo (`torchgeo.datasets.SKIPPD task="nowcast"` has 349,372 trainval and 14,003 test samples, exactly matching our parquet/npy row counts). The separate `forecast` task requires 16-step image sequences (130,412 samples).
   - Therefore, Model A performs **contemporaneous PV estimation from sky imagery**, NOT multi-step forward forecasting. We do not invent an ungrounded future horizon.

2. **Persistence Baseline Validity Analysis**:
   - Time-series persistence ($\hat{P}_t = P_{t-1}$) assumes access to an active physical PV power meter stream. A CNN predicting PV from imagery does not use past power as input.
   - Irregular gaps (overnight, multi-day) make naive index-lag persistence invalid across morning boundaries.
   - Defensible baselines for this task:
     * **Baseline 1: Global Mean / Median Regressor**: Predicts $\bar{P}_{\text{train}}$ (minimum baseline floor).
     * **Baseline 2: Diurnal Hourly Climatology**: Predicts $\bar{P}_{\text{train}}(\text{hour}(t))$ (tests whether the CNN learns cloud patterns beyond the sun's diurnal cycle).
     * **Baseline 3: Conditional 1-Minute Lagged Persistence**: Evaluated only for consecutive samples where $\Delta t \le 2\text{ min}$.

3. **Partial train data**: We have shard 1 of 5 (69,875 / ~349,372 rows = ~20%). This is sufficient for a proof-of-concept but NOT for benchmark comparability. Must document this clearly.

4. **Test npy ↔ test parquet MISALIGNED**: `times_test.npy` is non-monotonic, and 13,464/14,003 timestamps differ by 1 hour due to DST handling. The Parquet embedded `time` column is authoritative and aligned row-for-row with images and PV targets. Never use `.npy` files for row labeling.

5. **Train shard ↔ Test parquet day isolation**: Empirically verified that train shard 1 (92 days) and test parquet (20 days) share **0 calendar days** and **0 timestamps**.

6. **PV units**: VERIFIED as **kW** from the 30 kW rooftop installation at Stanford University (Nie et al., 2020). Target values max out at ~29.57 kW.

#### Recommended Actions — SKIPP'D
- Use `SkippdDataset` and `build_skippd_datasets` in `src/data/datasets.py`.
- Split train shard 1 into train/val using `src/data/splits.py:split_skippd_by_day` (calendar-day split).
- Evaluate on the 14,003-sample test parquet as the held-out test set.
- Evaluate against Global Mean, Diurnal Climatology, and Conditional Persistence baselines.

### 4.2 NASA POWER

| Property | Value | Status |
|---|---|---|
| Source | NASA POWER API | VERIFIED |
| File | `POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv` | VERIFIED |
| Location | Lat 19.076, Lon 72.8777 (Mumbai, India) | VERIFIED (from file header) |
| Elevation | 97.15m (MERRA-2 average) | VERIFIED |
| Time zone | LST (Local Standard Time) | VERIFIED |
| Date range | 2023-01-01 00:00 to 2023-12-31 23:00 | VERIFIED |
| Rows | 8,760 (= 365 days × 24 hours, complete year) | VERIFIED |
| Columns | YEAR, MO, DY, HR, ALLSKY_SFC_SW_DWN, T2M, RH2M, WS10M | VERIFIED |
| Missing values | 0 NaN | VERIFIED |
| Sentinel value | -999 (documented in header, but 0 found in data) | VERIFIED |
| Duplicates | 0 | VERIFIED |

#### Parameter Details

| Parameter | Description | Unit | Min | Max | Mean |
|---|---|---|---|---|---|
| ALLSKY_SFC_SW_DWN | All Sky Surface Shortwave Downward Irradiance | Wh/m² | 0.00 | 1008.53 | 211.63 |
| T2M | Temperature at 2 Meters | °C | 12.82 | 42.76 | 27.00 |
| RH2M | Relative Humidity at 2 Meters | % | 7.95 | 100.00 | 66.91 |
| WS10M | Wind Speed at 10 Meters | m/s | 0.07 | 11.28 | 3.61 |

#### Important Notes — NASA POWER
- **Irradiance is Wh/m², NOT kW/m²** — it's hourly accumulated energy, not instantaneous power.
- **This is satellite-derived reanalysis data**, not ground measurements.
- **WS10M is at 10m height**, not hub height for wind turbines.
- **Location is Mumbai** — completely unrelated to SKIPP'D (Stanford, California).
- **No sentinel values found** — -999 was documented as possible but didn't appear (good data quality).
- Complete temporal coverage: all 8,760 hours of 2023.

### 4.3 Wind Dataset

| Property | Value | Status |
|---|---|---|
| Directory | `data/raw/wind/` | VERIFIED |
| Contents | EMPTY | VERIFIED |
| Status | **NOT DOWNLOADED** | CONFIRMED |

**Action Required**: Must acquire wind data before Model C. Potential source: NREL India Wind Toolkit. Do NOT download without checking location, variables, temporal coverage, licensing, and storage.

### 4.4 BDG2 Demand Dataset (Model D)

| Property | Value | Status |
|---|---|---|
| Source | Building Data Genome Project 2 (BUDS Lab, Miller et al., 2020) | VERIFIED |
| Building ID | `Panther_education_Rosalie` (Education/Research, UCF Orlando, FL) | VERIFIED |
| Area & Year | 690.5 m² (7,432 sq ft), built 2008 | VERIFIED |
| Raw Extract | `data/raw/bdg2/bdg2_Panther_education_Rosalie_raw.csv` (17,544 rows) | ACQUIRED |
| Processed File | `data/processed/demand/demand_hourly_pilot.csv` (17,544 rows) | VALIDATED |
| Timezone | `US/Eastern` Local Standard Time (EST / UTC-5, no DST shifts) | VERIFIED |
| Date range | 2016-01-01 00:00:00 to 2017-12-31 23:00:00 (17,544 hours) | VERIFIED |
| Raw missingness | 20 nulls (0.11%): 4 gaps (5h, 1h [DST shift], 6h, 8h) | AUDITED |
| Early 2016 zeros | 3,373 inactive zeros (Jan 1 – May 20, 2016, nulled in author clean) | AUDITED |
| Active period | May 20, 2016 to Dec 31, 2017 (mean: 64.98 kWh, min: 11.0 kWh, max: 131.3 kWh, 0 zeros) | AUDITED |
| Imputation policy | Bounded causal forward-fill (limit=8h, zero lookahead bias) | AUDITED & LOGGED |
| Physical Bounds | Min 0.00 kWh, Max 131.30 kWh $\le 1000.0$ kWh threshold | VERIFIED |

---

## 5. Split Policies

### SKIPP'D (Model A)
- **Source**: HuggingFace parquet test file IS the test set (14,003 samples).
- **Train/Val split**: From train shard (69,875 samples), split chronologically by day.
- **Proposed**: 80% train / 20% validation by chronological day boundary.
- **Leakage risk**: Same-day images in both train/val could leak (similar sky conditions). Split by full calendar day to mitigate.
- **Status**: PLANNED

### NASA POWER (Model B)
- **Proposed**: Chronological 70/15/15 split of 8,760 hourly records.
  - Train: Jan 1 – Sep 10 (~6,132 hours)
  - Val: Sep 11 – Nov 7 (~1,314 hours)
  - Test: Nov 8 – Dec 31 (~1,314 hours)
- **Embargo**: At least 24-hour gap between splits (no overlapping windows).
- **Status**: PLANNED

### Wind (Model C) — ACQUIRED & VALIDATED
- **Source**: NREL India Wind Toolkit / IRIS (2014, Gujarat coastal grid cell 23.25°N, 69.67°E, 100m AGL).
- **Records**: 8,760 hourly continuous observations (2014-01-01 00:30:00 to 2014-12-31 23:30:00).
- **Splits**: Chronological 70/15/15 with 24-hour embargo buffers:
  - Train: Jan 1, 2014 00:30 to Sep 13, 2014 11:30 (6,132 hours)
  - Embargo: 24-hour purge (Sep 13 12:30 to Sep 14 11:30)
  - Val: Sep 14, 2014 12:30 to Nov 7, 2014 17:30 (1,314 hours)
  - Embargo: 24-hour purge (Nov 7 18:30 to Nov 8 17:30)
  - Test: Nov 8, 2014 18:30 to Dec 31, 2014 23:30 (1,314 hours)
- **Status**: COMPLETE & VERIFIED

### BDG2 (Model D) — ACQUIRED & VALIDATED
- **Source**: Building Data Genome Project 2 (BUDS Lab, Miller et al., 2020).
- **Building Profile**: `Panther_education_Rosalie` (Education / Research facility, 690.5 m², built 2008, Orlando, FL).
- **Records**: Exactly 17,544 hourly observations (2016 leap year: 8,784h + 2017 non-leap year: 8,760h).
- **Chronological Splits (70/15/15 with 24-hour embargo buffers)**:
  - Train: 12,280 hours (`2016-01-01 00:00:00` to `2017-05-26 15:00:00`). Contains early 2016 unmetered zeros and 6 NaNs.
  - Embargo: 24-hour purge (`2017-05-26 16:00:00` to `2017-05-27 15:00:00`). Zero sample overlap.
  - Val: 2,631 hours (`2017-05-27 16:00:00` to `2017-09-14 06:00:00`). 0 zero values (100% active load) and 6 NaNs.
  - Embargo: 24-hour purge (`2017-09-14 07:00:00` to `2017-09-15 06:00:00`). Zero sample overlap.
  - Test: 2,585 hours (`2017-09-15 07:00:00` to `2017-12-31 23:00:00`). 0 zero values (100% active load) and 8 NaNs.
- **Scaler Isolation**: `StandardScaler` fitted strictly on Train partition.
- **Status**: COMPLETE & VERIFIED (All 140 tests passing).

---

## 6. Leakage Risks

| Risk | Severity | Mitigation | Status |
|---|---|---|---|
| SKIPP'D: Same-minute images in train/val | HIGH | Split by full calendar day (`split_skippd_by_day`) | VERIFIED & TESTED |
| SKIPP'D: npy-parquet misalignment for test | HIGH | Use parquet timestamps only; ignore `.npy` for row matching | ENFORCED & TESTED |
| NASA: Future weather used as input | HIGH | Window construction uses only $t_{\text{input}} \le t < t_{\text{target}}$ | VERIFIED & TESTED |
| NASA: Scaler fit on full data | MEDIUM | Fit `StandardScaler` strictly on training split only | VERIFIED & TESTED |
| Cross-dataset join without verification | HIGH | Keep models independent; document incompatibility | ENFORCED & TESTED |

---

## 7. Model Architecture Plans & Baseline Benchmarks

| Model | Dataset | Architecture | Baselines (Fitted on Train, Evaluated on Val) | Status |
|---|---|---|---|---|
| A: Solar Image Nowcasting | SKIPP'D (shard 1) | ResNet-18 (frozen + fine-tune) | **Climatology**: MAE 1.88 kW, RMSE 2.71 kW, R² 0.8841<br>**Cond. Persistence** (<= 2m): MAE 0.14 kW, RMSE 0.44 kW, R² 0.9969<br>**Global Mean**: MAE 7.10 kW, RMSE 8.00 kW, R² -0.0088<br>**Global Median**: MAE 7.06 kW, RMSE 8.22 kW, R² -0.0646 | BASELINES & S1-S4 COMPLETE |
| B: NASA Weather Forecasting | NASA POWER | LSTM or TCN | **Seasonal Naive (24h)**: MAE 23.02 Wh/m², RMSE 53.79 Wh/m², R² 0.9598<br>**1-Hour Persistence**: MAE 61.16 Wh/m², RMSE 96.52 Wh/m², R² 0.8707 | BASELINES & W1-W4 COMPLETE |
| C: Wind Resource & Power | NREL Wind Toolkit | WindLSTM (C1-C4) | **1-Hour Persistence**: MAE 0.7644 m/s (86.06 kW), RMSE 1.2845 m/s, R² 0.6665<br>**24h Seasonal Naive**: MAE 1.4190 m/s (152.72 kW), RMSE 1.9749 m/s, R² 0.2117 | BASELINES & C1-C4 COMPLETE; C2 Selected (Val MAE: 0.7128 m/s, Test MAE: 0.5908 m/s, Test Power MAE: 47.89 kW)<br>**Conformal Intervals (Test)**: 80% Cov: 84.22% (MPIW: 2.09 m/s, Power: 181.14 kW), 90% Cov: 92.83% (MPIW: 3.06 m/s, Power: 270.23 kW) |
| D: Building Demand | BDG2 | DemandLSTM (D1-D4) | **1-Hour Persistence**: MAE 1.8052 kWh, RMSE 3.6647 kWh, R² 0.8107<br>**24h Seasonal Naive**: MAE 5.6797 kWh, RMSE 9.1857 kWh, R² -0.1892<br>**168h Weekly Seasonal Naive**: MAE 6.5699 kWh, RMSE 9.7644 kWh, R² -0.3793 | BASELINES & D1-D4 COMPLETE; D2 Selected (Val MAE: 2.0003 kWh, RMSE: 3.6014 kWh; Test RMSE: 4.0673 kWh, Test R²: 0.9234)<br>**Conformal Intervals (Test)**: 80% Cov: 66.03% (MPIW: 4.96 kWh), 90% Cov: 85.47% (MPIW: 9.53 kWh; summer-to-winter distribution shift documented) |
| E: Decision Support | All compatible | Modular aggregator | N/A | PLANNED (Stage 5 PoC; combines point forecasts and conformal intervals) |

---

## 8. Experiment Registry

Active registry maintained at `experiments/experiment_registry.csv`.
Currently logged: **48 registered records**:
- 17 completed baseline benchmarks (Solar, NASA POWER, BDG2 Demand, NREL Wind)
- 16 completed deep learning model runs (NASA W1-W4, Solar S1-S4, Wind C1-C4, Demand D1-D4)
- 4 completed held-out test evaluations (Solar S3_TEST, NASA W1_TEST, Wind C2_TEST, Demand D2_TEST)
- 4 completed conformal prediction uncertainty evaluations (`EXP_WIND_CP_80`, `EXP_WIND_CP_90`, `EXP_DEMAND_CP_80`, `EXP_DEMAND_CP_90`)
- 5 historical BLOCKED baseline pipeline entries (preserved from Stage 2)
- 2 historical PLANNED deep learning model entries (preserved from Stage 2)

---

## 9. Confirmed Facts vs. Assumptions

### Confirmed Facts
- SKIPP'D images are 64×64 RGB PNGs from Stanford, CA (ground fisheye camera ~125m from array).
- SKIPP'D PV target is in **kW** (kilowatts), verified from 30 kW rooftop array at Stanford University (Nie et al., 2020).
- SKIPP'D train shard 1 (92 calendar days) and test parquet (20 calendar days) share **0 calendar days** and **0 exact timestamps**.
- SKIPP'D memory footprint: `train-00000-of-00005.parquet` occupies 427.03 MB in RAM (~2.7% of 15.7 GB total system RAM). Images are decoded lazily on-demand in batches.
- NASA POWER data is hourly, complete for 2023 (8,760 hours), Mumbai region (19.076°N, 72.878°E). Monotonic continuity verified (zero missing hours, zero duplicates, zero NaNs, zero sentinels).
- NASA POWER 1h-ahead target alignment: for all samples across train, val, and test, target timestamp is strictly origin timestamp + 1h (`t_target - t_origin == 1h`). Embargo gap between train/val and val/test is >= 24h.
- **Real Wind Data Acquired & Validated**:
  - Raw original file saved immutably: `data/raw/wind/nrel_india_wind_2014_23.25N_69.67E.csv` (389,951 bytes).
  - Metadata sidecar: `data/raw/wind/nrel_india_wind_2014_23.25N_69.67E_metadata.json` (SiteID: 42395, Lat: 23.24346924°N, Lon: 69.6620636°E).
  - Standardized processed file: `data/processed/wind/wind_hourly_pilot.csv` (260,256 bytes, 8,760 rows).
  - Temporal span: 2014-01-01 00:30:00 to 2014-12-31 23:30:00 (continuous hourly timestamps, 0 gaps, 0 duplicates).
  - Physical ranges: `wind_speed` [0.08, 18.64] m/s (mean 6.93 m/s), `wind_direction` [0.0, 360.0] deg, `temperature` [10.26, 40.25] °C, `pressure` [970.70, 995.10] hPa.
  - Nature of data: Meso-scale atmospheric numerical simulation (WRF 3km grid reanalysis), NOT physical tower anemometer observations.
- Wind ingestion interface (`src/data/wind.py`), schema validator (`validate_wind_dataframe`), turbine power curve conversion (`TurbinePowerCurve`), dataset (`WindDataset`), and dataset builder (`build_wind_datasets`) are implemented and verified against real data.
- Wind baseline models (`WindPersistenceBaseline`, `WindSeasonalNaiveBaseline`), power estimation, and `WindLSTM` sequence models (C1-C4) are trained, benchmarked, and evaluated on test data (`EXP_WIND_DL_C2_TEST`: Test MAE 0.5908 m/s, Power MAE 47.89 kW).
- Demand ingestion interface (`src/data/demand.py`), schema validator (`validate_demand_dataframe`), dataset (`DemandDataset`), and dataset builder (`build_demand_datasets`) are implemented and verified against real BDG2 data.
- Demand baseline models (`DemandPersistenceBaseline`, `DemandSeasonalNaiveBaseline`, `DemandWeeklySeasonalNaiveBaseline`), and `DemandLSTM` sequence models (D1-D4) are trained, benchmarked, and evaluated on test data (`EXP_DEMAND_DL_D2_TEST`: Test RMSE 4.0673 kWh, R² 0.9234).
- **Split Conformal Prediction Framework (Stage 4 Complete)**:
  - Reusable module: `src/uncertainty/conformal.py` (`SplitConformalCalibrator`, `calculate_interval_metrics`).
  - Finite-sample quantile index: $k = \min(n, \lceil(n+1)(1-\alpha)\rceil)$.
  - Model C Wind (Test N=1,242): 80% Cov = 84.22% (MPIW: 2.0877 m/s, Winkler: 3.0501), 90% Cov = 92.83% (MPIW: 3.0602 m/s, Winkler: 3.9157).
  - Model C Power Proxy (Test N=1,242): 80% Cov = 87.20% (MPIW: 181.14 kW), 90% Cov = 94.36% (MPIW: 270.23 kW).
  - Model D Demand (Test N=2,561): 80% Cov = 66.03% (MPIW: 4.9561 kWh, Winkler: 15.4839), 90% Cov = 85.47% (MPIW: 9.5254 kWh, Winkler: 20.6045). Undercoverage empirically attributed to summer-to-winter distribution shift in Florida building electricity load.
  - Physical non-negative clipping ($\ge 0$ m/s, $\ge 0$ kWh) strictly preserves or reduces interval width without reducing empirical coverage.
- PyTorch is CPU-only, verified as **2.14.1+cpu**; torchvision is verified as **0.29.1+cpu**.
- Complete test suite: **192 passed, 0 failed, 14 warnings in 11.74s (100% passing)**.

### Assumptions & Guidelines
- Model C forecasts physical wind speed ($v_{100\text{m}}$ in m/s) and converts it to estimated electrical power via `TurbinePowerCurve`. The generic 2.0 MW turbine power curve (cut-in 3.0 m/s, rated 12.0 m/s, cut-out 25.0 m/s) is an *illustrative engineering approximation / power proxy*, NOT a validated turbine specification under IEC 61400-12-1, nor an empirical model of a commercial turbine. Estimated power ($P_{\text{est}}$ in kW) must NEVER be described as measured wind-farm generation or actual SCADA production.
- BDG2 electricity meter readings represent hourly electricity consumption in kWh (equivalent to average active power in kW over 1h). Single building meters (e.g. `Panther_education_Rosalie`) must be maintained with strict meter isolation.

---

## 10. Current Blockers

**None**. All project stages (Stage 1 through Stage 6) are fully implemented, verified, tested, and documented.
- 192/192 pytest tests passing (100%).
- 52 verified experiment registry entries.
- Master orchestration scripts runnable end-to-end.
- Interactive multi-page Streamlit dashboard (`app.py`) fully verified with 8 functional pages.
- Machine-readable tables, summary JSON, and publication figures generated.

---

## 11. Project Roadmap & Status

1. ✅ Complete environment audit (`scripts/check_env.py`)
2. ✅ Complete data audit for all raw directories (`scripts/audit_data.py`, `scripts/audit_skippd_deep.py`, `scripts/verify_skippd_alignment.py`)
3. ✅ Write formal `reports/data_audit.md`
4. ✅ Write `reports/leakage_audit.md`
5. ✅ Implement leakage-safe data split utilities (`src/data/splits.py`)
6. ✅ Implement PyTorch Datasets & DataLoader builders (`src/data/datasets.py:SkippdDataset,NasaWeatherDataset`)
7. ✅ Add experiment configuration files (`configs/skippd_solar.yaml`, `configs/nasa_weather.yaml`, `configs/experiments/`)
8. ✅ Phase 2: Implement baseline models for Models A and B
9. ✅ Phase 2: Experiment tracking registry & evaluation metrics (`src/evaluation/`, `experiments/experiment_registry.csv`, `reports/baseline_report.md`)
10. ✅ Phase 3: Train Model B (NASA WeatherLSTM W1-W4) and Model A (Solar ResNet S1-S4)
11. ✅ Stage 2: Implement Model C (Wind resource pipeline, illustrative turbine power proxy curve, baselines, `WindLSTM`, configs, tests)
12. ✅ Stage 2: Implement Model D (BDG2 electricity demand pipeline, meter validator, baselines, `DemandLSTM`, configs, tests)
13. ✅ Stage 2A: Safely acquire and validate NREL India Wind Toolkit pilot dataset (8,760 hours, 2014, Gujarat, 100m AGL)
14. ✅ Stage 2B: Prepare extraction and verify BDG2 candidate (`Panther_education_Rosalie`)
15. ✅ Stage 2C: Safely acquire, causally clean, audit, and validate BDG2 electricity demand dataset (`Panther_education_Rosalie`); evaluate real demand baselines (140/140 tests passing)
16. ✅ Stage 2C Audit: Final independent audit of Stage 2C implementation, DST verification, causality check, baseline reproductions, split verification (140/140 tests passing)
17. ✅ Stage 3: Neural-network sequence modeling (`WindLSTM` & `DemandLSTM` trained, benchmarked, evaluated; 147/147 tests passing)
18. ✅ Stage 4: Leakage-safe uncertainty estimation (Split conformal prediction, interval metrics, coverage diagnostics; 162/162 tests passing)
19. ✅ Stage 5: Uncertainty-aware energy feasibility and mix optimization module (Model E completed; deterministic optimizer, strict energy conservation, BESS modeling, conformal stress testing; 177/177 tests passing)
20. ✅ Stage 6: Multi-Page Research Frontend (`app.py`, `src/frontend/`), automated integration tests (`tests/test_frontend.py`), and final project integration (192/192 tests passing)
21. ✅ Stage 7: UI/UX Redesign — Polished, modern, light-theme Renewable Energy Intelligence dashboard matching prototype (clean white/grey cards, dark navy typography, prototype donut chart, quick actions, mini forecasts, high-contrast buttons, user-led manual testing)
22. ✅ Stage 8: UI/UX Accessibility & Readability Repair — High-contrast WCAG AA light theme (#F8FAFC canvas, #FFFFFF panels, #172033 text, #047857 primary green), eliminated CommonMark indented code-block `</div>` rendering artefacts by switching to single-string unindented HTML, high-specificity sidebar active/inactive item styling, added rich `render_control_context_card` metadata guides for all sliders across Models B, C, D, and E, replaced blank/unstyled dataframe tables with high-contrast `.styled-table` HTML representations for Methodology & Uncertainty, fixed Plotly chart axes, units, legends, and non-scientific timestamp formatting. Presentation-only changes; no application execution, automated testing, or model training performed per strict instructions.

---

## 12. Commands to Resume & Verify

```powershell
# Launch interactive Streamlit research dashboard
streamlit run app.py

# Verify full repository test suite (192 tests passing, 0 failed, 14 warnings)
python -m pytest tests/ -q

# Run Stage 5 Decision Support & Optimization orchestration
python scripts/run_decision_support.py

# Run Stage 4 Split Conformal Prediction pipeline
python scripts/evaluate_uncertainty.py

# Check experiment registry (52 total verified entries)
python -c "import pandas as pd; df = pd.read_csv('experiments/experiment_registry.csv'); print(f'Total: {len(df)} entries'); print(df[['experiment_id', 'model_name', 'status', 'metric_unit']].tail(5))"
```



  