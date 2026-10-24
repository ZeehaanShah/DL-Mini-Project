# Stage 2 Report: Wind and Electricity-Demand Forecasting Pipelines

**Project:** DL MiniProject — Renewable Energy Forecasting and Location-Specific Feasibility  
**Date:** 2026-10-21  
**Phase:** Stage 2 — Wind (Model C) & Electricity Demand (Model D) Pipeline Implementation  
**Auditor:** Senior Deep Learning Engineer & Pipeline Auditor  

---

## 1. Executive Summary & Repository Audit

In Stage 2, the forecasting pipelines for **Model C (Wind Resource & Turbine Power Estimation)** and **Model D (BDG2 Electricity Demand)** were fully designed, implemented, and verified with zero data leakage.

### Initial Environment & Workspace Audit

| Check | Target | Status | Detail / Result |
|---|---|---|---|
| **Git Working Tree** | Repo root | Clean | All untracked and working files preserved; zero user changes discarded |
| **Existing Test Suite** | `tests/` | **91 passed** | Verified actual baseline test suite before any changes (14 warnings in 14.95s) |
| **Wind Directory** | `data/raw/wind/` | **EMPTY** | No raw wind files present; marked as BLOCKED |
| **BDG2 Directory** | `data/raw/bdg2/` | **EMPTY** | No raw meter files present; marked as BLOCKED |
| **Credentials & .env** | `.env` | **VERIFIED** | `.env` exists; defines `NREL_API_KEY`; key values masked; zero secrets logged |
| **PyTorch & Platform** | System Python 3.11.9 | **VERIFIED** | PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu (CPU-only execution) |

### Policy Compliance
- **No Unapproved Downloads**: In strict compliance with instructions, no large downloads or external API calls were initiated.
- **No Fabricated Real-Data Results**: Ingestion interfaces, validators, and baselines were validated using rigorous, isolated synthetic fixtures. Real data models and baselines are explicitly recorded as `BLOCKED` in documentation and the experiment registry.
- **Isolation of Historical Benchmarks**: Existing SKIPP'D and NASA POWER data, code, test sets, and recorded metrics were kept strictly untouched.

---

## 2. Model C: Wind Resource Forecasting & Turbine Power Estimation

### 2.1 Target Definition & Physical Units
- **Primary Forecast Target**: Hub-height wind speed in **m/s** (meters per second) at 100 meters above ground level ($v_{100\text{m}}$).
- **Physical Bounds**: $0.0 \le v \le 75.0\text{ m/s}$ (rejects negative speeds and unphysical values exceeding hurricane threshold).
- **Meteorological Context**: Modeled or reanalysis wind datasets (such as NREL Wind Toolkit) provide meteorological wind speeds, not turbine power.
- **Forecast Formulation**: Causal sequence regression mapping $L = 24$ past hourly observations to next-step wind speed at horizon $H = 1$ hour ($t+1$).

### 2.2 Turbine Power Curve Conversion Model (Illustrative Engineering Approximation)
Wind power estimation is implemented as an **illustrative engineering approximation and power proxy** using a standard aerodynamic cubic relationship:
- **Primary Forecast Target**: The sequence model strictly forecasts physical meteorological wind speed ($v_{100\text{m}}$ in m/s). The forecasted wind speed is subsequently converted into an estimated electrical power output proxy ($P_{\text{est}}$ in kW).
- **Illustrative Proxy Model**: Generic 2.0 MW utility-scale engineering proxy (an illustrative approximation; NOT a validated commercial turbine under IEC 61400-12-1 unless supported by a documented manufacturer curve and site-specific calibration).
- **Rated Capacity ($P_{\text{rated}}$)**: 2,000.0 kW (2.0 MW).
- **Hub Height**: 100.0 meters.
- **Cut-in Speed ($v_{\text{in}}$)**: 3.0 m/s.
- **Rated Speed ($v_{\text{rated}}$)**: 12.0 m/s.
- **Cut-out Speed ($v_{\text{out}}$)**: 25.0 m/s.
- **Reference Air Density ($\rho_0$)**: $1.225\text{ kg/m}^3$ (standard sea-level atmospheric density).

#### Mathematical Formulation

$$P(v) = \begin{cases} 
0.0 & \text{if } v < v_{\text{in}} \\
P_{\text{rated}} \cdot \left(\frac{v^3 - v_{\text{in}}^3}{v_{\text{rated}}^3 - v_{\text{in}}^3}\right) & \text{if } v_{\text{in}} \le v < v_{\text{rated}} \\
P_{\text{rated}} & \text{if } v_{\text{rated}} \le v < v_{\text{out}} \\
0.0 & \text{if } v \ge v_{\text{out}} \text{ (storm shutdown)}
\end{cases}$$

#### Air Density Normalization
When air density $\rho$ is available, wind speed is adjusted using standard atmospheric cube-root density correction prior to curve evaluation:

$$v_{\text{eff}} = v \cdot \left(\frac{\rho}{\rho_0}\right)^{1/3}$$

> [!IMPORTANT]
> The sequence model forecasts meteorological wind speed (m/s) and converts it to estimated power. This conversion produces an **illustrative electrical power proxy** ($P_{\text{est}}$ in kW) for preliminary renewable-energy feasibility analysis. It must **NEVER** be described or reported as measured wind-farm generation or actual SCADA production data.

### 2.3 Implementation & Baseline Models
- **`src/data/wind.py`**:
  - `TurbinePowerCurve`: Aerodynamic piecewise calculation with vectorization and density adjustment.
  - `validate_wind_dataframe`: Rejects missing columns, non-monotonicity, hourly gaps, NaNs, sentinels ($\le -900$), out-of-bounds speeds ($<0$ or $>75$ m/s), and directions ($[0, 360]$).
  - `WindDataset`: PyTorch dataset with causal timestamp tracking and unscaling.
  - `build_wind_datasets`: Enforces chronological 70/15/15 split, 24h embargo, train-only `StandardScaler`.
- **`src/models/wind/baselines.py`**:
  - `WindPersistenceBaseline`: $\hat{v}_{t+1} = v_t$ (observation at forecast origin). Provides `predict_power` for estimated kW evaluation.
  - `WindSeasonalNaiveBaseline`: $\hat{v}_{t+1} = v_{t - 23}$ (same-hour-yesterday observation).
- **`src/models/wind/sequence_models.py`**:
  - `WindLSTM`: Causal stacked LSTM sequence model for future deep learning experiments.
- **`configs/wind_resource.yaml`**: Complete configuration file with status `BLOCKED_PENDING_DATA`.

---

## 3. Model D: BDG2 Electricity Demand Forecasting

### 3.1 Dataset Provenance & Reading Definition
- **Dataset**: Building Data Genome Project 2 (BDG2) (Miller et al., Scientific Data 2020).
- **License**: Open access, Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Units and Physical Meaning**: BDG2 electricity meter readings represent **hourly energy consumption in kWh** (kilowatt-hours). Because $\Delta t = 1\text{ hour}$, the numerical consumption in kWh is mathematically equal to the average active electrical power in **kW** during that hour ($1\text{ kWh} / 1\text{ h} = 1\text{ kW}$).
- **Meter Candidate**: Single institutional building meter `Panther_education_babette` (education facility with regular occupation and floor area ~6,250 $m^2$).
- **Meter Isolation Policy**: Independent meters must NEVER be aggregated across disparate campuses or building types without explicit metadata normalization.

### 3.2 Implementation & Baseline Models
- **`src/data/demand.py`**:
  - `validate_demand_dataframe`: Validates timestamp monotonicity, 1-hour continuity, non-negativity ($y \ge 0$), plausibility ceiling ($y \le 10,000\text{ kWh}$), and absence of NaNs/sentinels.
  - `DemandDataset`: PyTorch dataset for sliding windows $(X, y)$ with target unscaling.
  - `build_demand_datasets`: Chronological 70/15/15 split with 24-hour temporal embargo.
- **`src/models/demand/baselines.py`**:
  - `DemandPersistenceBaseline`: $\hat{y}_{t+1} = y_t$ (last observed hourly demand).
  - `DemandSeasonalNaiveBaseline`: $\hat{y}_{t+1} = y_{t - 23}$ (same hour yesterday).
  - `DemandWeeklySeasonalNaiveBaseline`: $\hat{y}_{t+1} = y_{t - 167}$ (same hour, same day last week for sequence length $L \ge 168$).
- **`src/models/demand/sequence_models.py`**:
  - `DemandLSTM`: Causal stacked LSTM sequence model for future deep learning experiments.
- **`configs/bdg2_demand.yaml`**: Complete configuration file with status `BLOCKED_PENDING_DATA`.

---

## 4. Leakage-Safe Time-Series Protocol

Both pipelines strictly enforce the project's zero-leakage protocol:

1. **Chronological Partitioning Prior to Windowing**: The continuous hourly time series is partitioned into Train (70%), Validation (15%), and Test (15%) partitions before constructing any sliding windows.
2. **Temporal Embargo Gap**: A 24-hour embargo interval is purged between train/val and val/test boundaries:
   $$\min(t_{\text{val, target}}) - \max(t_{\text{train, target}}) > 24\text{ hours}$$
3. **Causal Forward-Only Windows**: Window construction ensures no lookahead bias:
   $$\max(t_{\text{input}}) < t_{\text{target}}, \quad t_{\text{target}} - t_{\text{origin}} = H = 1\text{ hour}$$
4. **Train-Only Preprocessing**: `StandardScaler` is fitted strictly on `train_df`. Validation and test partitions are transformed out-of-sample with fixed parameters.
5. **No Cross-Dataset Joins**: Wind and demand pipelines operate completely independently.

---

## 5. Verification & Automated Test Suite

A comprehensive test suite was executed covering unit, functional, and pipeline integration tests:

```bash
python -m pytest tests/ -v
```

### Test Suite Results

| Test File | Tests Run | Result | Scope / Description |
|---|---|---|---|
| `tests/test_nasa_integrity.py` | 21 | **PASSED** | NASA POWER schema, quality, coverage, sentinels, continuity |
| `tests/test_skippd_integrity.py` | 19 | **PASSED** | SKIPP'D Parquet schema, kW target, timestamps, image decoding |
| `tests/test_leakage.py` | 10 | **PASSED** | Day splitting, chronological embargoes, lookahead prevention |
| `tests/test_baselines.py` | 15 | **PASSED** | Solar & NASA baseline models, metric mathematical verification |
| `tests/test_datasets.py` | 19 | **PASSED** | SKIPP'D and NASA PyTorch Dataset and DataLoader pipelines |
| `tests/test_deep_learning.py` | 7 | **PASSED** | ResNet-18, WeatherLSTM, trainer loops, checkpoint restoration |
| `tests/test_wind_pipeline.py` | **22** | **PASSED** | Wind schema validator, physical bounds, illustrative power proxy curve, embargo, baselines, ValueError input guards, `WindLSTM` smoke training |
| `tests/test_demand_pipeline.py` | **19** | **PASSED** | Demand schema validator, non-negative demand, 24h & 168h seasonal naive baselines, embargo, ValueError input guards, `DemandLSTM` smoke training |
| **TOTAL** | **132** | **132 PASSED** | **100% passing (0 failed, 14 deprecation warnings in 16.19s)** |

---

## 6. Experiment Registry Updates

The experiment registry (`experiments/experiment_registry.csv`) was updated to track the new pipelines with explicit distinction between completed, planned, and blocked experiments:

| Experiment ID | Dataset | Model | Partition | Status | Target Unit | Notes |
|---|---|---|---|---|---|---|
| `EXP_WIND_BASE_PERSIST` | NREL_WIND_TOOLKIT | `WindPersistenceBaseline` | val | **BLOCKED** | m/s | Awaiting NREL raw data; pipeline verified |
| `EXP_WIND_BASE_SEASONAL` | NREL_WIND_TOOLKIT | `WindSeasonalNaiveBaseline` | val | **BLOCKED** | m/s | Awaiting NREL raw data |
| `EXP_WIND_DL_LSTM_PLANNED` | NREL_WIND_TOOLKIT | `WindLSTM` | val | **PLANNED** | m/s | Architecture implemented; pending data |
| `EXP_DEMAND_BASE_PERSIST` | BDG2 | `DemandPersistenceBaseline` | val | **BLOCKED** | kWh | Awaiting BDG2 raw data; pipeline verified |
| `EXP_DEMAND_BASE_SEASONAL_24H` | BDG2 | `DemandSeasonalNaiveBaseline` | val | **BLOCKED** | kWh | Awaiting BDG2 raw data |
| `EXP_DEMAND_BASE_WEEKLY_168H` | BDG2 | `DemandWeeklySeasonalNaiveBaseline` | val | **BLOCKED** | kWh | Awaiting BDG2 raw data |
| `EXP_DEMAND_DL_LSTM_PLANNED` | BDG2 | `DemandLSTM` | val | **PLANNED** | kWh | Architecture implemented; pending data |

*(Total records in registry: 24, including 17 previously completed benchmarks and runs).*

---

## 7. Remaining Blockers & Proposed Next Actions

### Remaining Blockers
1. **Model C Real Data**: `data/raw/wind/` is empty. The pipeline is fully ready to ingest and validate NREL India Wind Toolkit data once acquired.
2. **Model D Real Data**: `data/raw/bdg2/` is empty. The pipeline is fully ready to ingest and validate the selected BDG2 meter once acquired.

### Proposed Next Actions (Pending User Approval)
1. **Wind Pilot Extract Acquisition**:
   - Proposed source: NREL Wind Toolkit API using available `NREL_API_KEY`.
   - Proposed site: Western/coastal India pilot site (e.g. 23.25°N, 69.67°E in Gujarat or 19.076°N, 72.878°E in Mumbai).
   - Pilot volume: 1 calendar year (8,760 hourly records) to minimize bandwidth and request quota.
   - Target variables: 100m wind speed (m/s), wind direction (°), surface temperature (°C), surface pressure (hPa).
2. **BDG2 Meter Acquisition**:
   - Proposed source: BDG2 official repository (`buds-lab/building-data-genome-project-2`).
   - Proposed extract: `electricity.csv` column for `Panther_education_babette` (1-2 years, ~8,760–17,520 hourly kWh readings) plus corresponding metadata row from `metadata.csv`.
3. **Execution of Real-Data Baselines & LSTM Training**:
   - Once raw data files are in place, run validation through `validate_wind_dataframe` and `validate_demand_dataframe`.
   - Evaluate empirical baseline benchmarks on validation partitions and record in registry.
   - Train `WindLSTM` and `DemandLSTM` models with early stopping and learning curve diagnostics.
