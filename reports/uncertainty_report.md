# Stage 4: Leakage-Safe Uncertainty Estimation for WindLSTM and DemandLSTM

**Author**: Senior Deep Learning Engineer & Research Auditor  
**Date**: 2026-10-21  
**Scope**: Split Conformal Prediction Intervals for Model C (`EXP_WIND_DL_C2`) and Model D (`EXP_DEMAND_DL_D2`)  
**Status**: COMPLETE (All 162 unit tests passing; zero model retraining; strict chronological partitioning; zero test leakage)

---

## 1. Executive Summary & Mandatory Safeguards Audit

In Stage 4, split conformal prediction intervals were calibrated and evaluated for the selected best-performing sequence models:
- **Wind Resource Forecasting**: `EXP_WIND_DL_C2` (`WindLSTM`, wider architecture with $h=64$, $L=1$, dropout=0.1)
- **Building Electricity Demand Forecasting**: `EXP_DEMAND_DL_D2` (`DemandLSTM`, wider architecture with $h=64$, $L=1$, dropout=0.1)

Both forecasting models were loaded directly from their frozen checkpoints (`checkpoints/EXP_WIND_DL_C2/best_model.pt` and `checkpoints/EXP_DEMAND_DL_D2/best_model.pt`). **Neither model was retrained**.

### Compliance Checklist

| Safeguard Requirement | Implementation Verification | Status |
|---|---|:---:|
| **1. Checkpoint & Scaler Inspection** | Inspected `WindLSTM`, `DemandLSTM`, `StandardScaler`, sequence configs, and registry prior to execution. Loaded frozen checkpoints without retraining. | VERIFIED |
| **2. Historical Preservation** | Historical baseline, solar, NASA POWER, and Stage 3 experiment records and predictions in `experiments/experiment_registry.csv` preserved completely without modification. | VERIFIED |
| **3. Test Leakage Prevention** | Scalers, models, and conformal calibration quantiles $q$ were fitted strictly on training and dedicated calibration sets. The held-out test partition was never accessed during calibration. | VERIFIED |
| **4. Dedicated Chronological Calibration** | Used the chronological validation partitions (isolated by 24h embargos) as dedicated calibration sets. Exact timestamps and sample counts documented below. | VERIFIED |
| **5. Training Precedes Calibration** | Training sets strictly precede calibration partitions in time. Models were trained on data chronologically prior to calibration sets. | VERIFIED |
| **6. No Test Tuning** | Test residuals were never used to tune interval widths or select models; evaluated strictly once in a confirmatory evaluation pass. | VERIFIED |
| **7. Units & Engineering Representation** | Wind speed in **m/s**, demand in **kWh**, and estimated turbine power in **kW**. Generic 2 MW power curve is explicitly treated as an illustrative power proxy, not measured generation. | VERIFIED |
| **8. Exchangeability Caveat** | Formal acknowledgement that time-series autocorrelation and seasonal distribution shift violate the exchangeability requirement of ordinary split conformal prediction. | VERIFIED |
| **9. Empirical Diagnostics vs Test** | Calibration diagnostics and final held-out test evaluations are reported separately, including undercoverage and overcoverage breakdown. | VERIFIED |
| **10. Geographic & Physical Isolation** | Maintained strict separation between Gujarat wind data (WRF reanalysis at 100m AGL) and Florida building demand data (Orlando education facility). | VERIFIED |

---

## 2. Methodology & Conformal Calibration Formulation

### 2.1 Split Conformal Framework
Given a dedicated calibration partition of $n$ samples $\{(X_i, y_i)\}_{i=1}^n$ and point predictions $\hat{y}_i = \hat{f}(X_i)$, the nonconformity score is defined by the absolute prediction residual:
$$R_i = |y_i - \hat{y}_i|$$

Under the assumption of exchangeability, split conformal prediction guarantees that for a new observation $(X_{n+1}, Y_{n+1})$ and miscoverage level $\alpha \in (0, 1)$ (nominal coverage $1 - \alpha$), the finite-sample adjusted critical nonconformity score is:
$$q_{1-\alpha} = R_{(k)}, \quad \text{where } k = \min\left(n, \left\lceil (n + 1)(1 - \alpha) \right\rceil\right)$$
where $R_{(1)} \le R_{(2)} \le \dots \le R_{(n)}$ are the sorted calibration nonconformity scores.

For any test forecast $\hat{y}_{\text{test}}$, the symmetric two-sided prediction interval is given by:
$$C(X_{\text{test}}) = [\hat{y}_{\text{test}} - q_{1-\alpha}, \; \hat{y}_{\text{test}} + q_{1-\alpha}]$$

### 2.2 Physically Justified Non-Negative Clipping
Both physical quantities have a strict physical floor at zero:
- Wind speed: $v \ge 0.0$ m/s
- Building electricity demand: $d \ge 0.0$ kWh

Lower bounds were clipped at $0.0$:
$$\hat{y}_l = \max(0.0, \; \hat{y}_{\text{test}} - q_{1-\alpha})$$
Because the true physical values are non-negative ($y_{\text{true}} \ge 0.0$), clipping the lower bound at zero **never reduces empirical coverage**, but strictly **reduces or maintains interval width** ($\text{MPIW}_{\text{clipped}} \le \text{MPIW}_{\text{raw}}$). Both raw and clipped metrics are reported to quantify the exact effect.

### 2.3 Estimated Turbine Power Intervals
For wind forecasting, point predictions and interval bounds are converted to estimated electrical power (in kW) through the illustrative `TurbinePowerCurve` $P(v)$:
$$[P_l, P_u] = [P(\hat{v}_l), \; P(\hat{v}_u)]$$
where $P(v) = 0$ for $v < v_{\text{cut-in}}$ (3.0 m/s), cubic aerodynamic progression up to $v_{\text{rated}}$ (12.0 m/s, 2000 kW), and rated power until $v_{\text{cut-out}}$ (25.0 m/s).

### 2.4 Interval Quality Metrics
1. **Empirical Coverage**:
   $$\text{Coverage} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(l_i \le y_i \le u_i)$$
2. **Mean Prediction Interval Width (MPIW)**:
   $$\text{MPIW} = \frac{1}{N} \sum_{i=1}^N (u_i - l_i)$$
3. **Winkler Interval Score** (Gneiting & Raftery, 2007):
   A strictly proper scoring rule penalizing both interval width and miscoverage:
   $$W_\alpha(l_i, u_i, y_i) = (u_i - l_i) + \frac{2}{\alpha}(l_i - y_i)\mathbb{I}(y_i < l_i) + \frac{2}{\alpha}(y_i - u_i)\mathbb{I}(y_i > u_i)$$
   Lower Winkler scores indicate superior interval quality (tighter intervals with minimal penalty for misses).

---

## 3. Dedicated Chronological Partitions

To ensure zero temporal leakage, both datasets enforce strict chronological order with 24-hour embargo buffers between partitions:

```
[ Train Partition ] ---> [ 24h Embargo ] ---> [ Calibration (Val) ] ---> [ 24h Embargo ] ---> [ Held-Out Test ]
```

### Partition Metadata

| Domain | Partition | Window Count | Start Timestamp | End Timestamp | Role |
|---|---|:---:|---|---|---|
| **Wind (`EXP_WIND_DL_C2`)** | Calibration | 1,290 | 2014-09-15 12:30:00 | 2014-11-08 05:30:00 | Nonconformity score quantile fitting ($q$) |
| **Wind (`EXP_WIND_DL_C2`)** | Held-Out Test | 1,242 | 2014-11-10 06:30:00 | 2014-12-31 23:30:00 | Confirmatory interval evaluation |
| **Demand (`EXP_DEMAND_DL_D2`)** | Calibration | 2,607 | 2017-05-28 16:00:00 | 2017-09-14 06:00:00 | Nonconformity score quantile fitting ($q$) |
| **Demand (`EXP_DEMAND_DL_D2`)** | Held-Out Test | 2,561 | 2017-09-16 07:00:00 | 2017-12-31 23:00:00 | Confirmatory interval evaluation |

---

## 4. Empirical Results & Findings

### 4.1 Model C: Wind Resource & Estimated Power Uncertainty

On the dedicated calibration set ($n = 1,290$), the critical nonconformity quantiles are:
- $q_{80} = 1.0439$ m/s ($k = 1,033$)
- $q_{90} = 1.5316$ m/s ($k = 1,162$)

#### Held-Out Test Interval Performance (1,242 Samples)

| Nominal Level | Empirical Coverage | Nominal Cov. | Coverage Gap | Raw MPIW | Clipped MPIW | Width Reduction | Winkler Score | Under-Cov. | Over-Cov. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **80% ($\alpha = 0.20$)** | **84.22%** | 80.00% | +4.22% | 2.0879 m/s | **2.0877 m/s** | 0.0002 m/s | **3.0501** | 8.05% | 7.73% |
| **90% ($\alpha = 0.10$)** | **92.83%** | 90.00% | +2.83% | 3.0633 m/s | **3.0602 m/s** | 0.0031 m/s | **3.9157** | 3.46% | 3.70% |

#### Estimated Turbine Electrical Power Proxy Intervals (kW)

| Nominal Level | Power Coverage | Nominal Cov. | Coverage Gap | Power MPIW | Power Winkler Score | Under-Cov. | Over-Cov. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **80% ($\alpha = 0.20$)** | **87.20%** | 80.00% | +7.20% | **181.14 kW** | **257.30 kW** | 5.07% | 7.73% |
| **90% ($\alpha = 0.10$)** | **94.36%** | 90.00% | +4.36% | **270.23 kW** | **342.52 kW** | 1.93% | 3.70% |

> [!NOTE]
> For wind speed, empirical coverage on the held-out test partition exceeded nominal targets (84.22% vs 80%, 92.83% vs 90%). Non-linear mapping through `TurbinePowerCurve` compressed intervals at near-zero speeds (cut-in threshold) and high speeds (rated plateau), resulting in power coverage of 87.20% and 94.36%.

---

### 4.2 Model D: Building Electricity Demand Uncertainty

On the dedicated calibration set ($n = 2,607$), the critical nonconformity quantiles are:
- $q_{80} = 2.4781$ kWh ($k = 2,087$)
- $q_{90} = 4.7627$ kWh ($k = 2,348$)

#### Held-Out Test Interval Performance (2,561 Samples)

| Nominal Level | Empirical Coverage | Nominal Cov. | Coverage Gap | Raw MPIW | Clipped MPIW | Width Reduction | Winkler Score | Under-Cov. | Over-Cov. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **80% ($\alpha = 0.20$)** | **66.03%** | 80.00% | -13.97% | 4.9561 kWh | **4.9561 kWh** | 0.0000 kWh | **15.4839** | 17.02% | 16.95% |
| **90% ($\alpha = 0.10$)** | **85.47%** | 90.00% | -4.53% | 9.5254 kWh | **9.5254 kWh** | 0.0000 kWh | **20.6045** | 7.26% | 7.26% |

---

## 5. Research Analysis: Time-Series Exchangeability & Distribution Shift

A central contribution of this stage is an honest, mathematically sound analysis of conformal prediction in time-series settings:

### 5.1 Why Demand Intervals Exhibited Undercoverage on Test
On the calibration set (May 28 to Sep 14, 2017), the calibrator achieved exact empirical coverage matching nominal levels:
- 80.09% on calibration for $\alpha = 0.20$
- 90.03% on calibration for $\alpha = 0.10$

However, on the held-out test set (Sep 16 to Dec 31, 2017), empirical coverage dropped to 66.03% and 85.47%. This demonstrates the theoretical principle documented in Safeguard #8:
1. **Seasonal Distribution Shift**: The calibration period occurs during peak summer in Florida, where HVAC cooling demand is high and relatively stationary across hot days. The test set encompasses autumn and winter, with cooler temperatures, reduced base cooling, Thanksgiving holidays, and December university semester breaks.
2. **Shift in Error Variance**: As demonstrated by test set point evaluation (`EXP_DEMAND_DL_D2_TEST`: test MAE 2.62 kWh vs val MAE 2.00 kWh), forecast residual variance expanded during the winter period. Because split conformal prediction uses a single static quantile $q$ calibrated in summer, it underestimated the spread required during winter.
3. **Violation of Exchangeability**: Ordinary split conformal guarantees require calibration and test nonconformity scores to be exchangeable (i.i.d. or invariant under permutation). In time series, strong autocorrelation and non-stationary seasonality violate exchangeability.

### 5.2 Why Wind Intervals Maintained Valid Coverage
For wind speed in Gujarat, the calibration period (mid-September to early November) encompassed the late southwest monsoon transition, characterized by gusty wind variations. The winter test period (mid-November to late December) experienced slightly lower average wind variance. Consequently, the calibration nonconformity scores were conservative relative to test fluctuations, preserving coverage above nominal levels.

---

## 6. Generated Visualizations

The uncertainty pipeline generated the following diagnostic and trajectory plots in `reports/figures/`:

1. `wind_conformal_calibration_curve.png`:
   Nominal vs empirical coverage curve across confidence levels (50% to 99%) on both calibration and test sets, comparing against the ideal 1:1 line.
2. `wind_conformal_intervals_test.png`:
   Chronological 240-hour trajectory comparing actual wind speed (m/s), `WindLSTM` point forecasts, and 80% and 90% conformal bands.
3. `wind_power_conformal_intervals_test.png`:
   Chronological trajectory of estimated turbine electrical power proxy (kW) with 80% and 90% prediction intervals.
4. `wind_conformal_residuals.png`:
   Histogram of calibration nonconformity scores with vertical markers at $q_{80}$ (1.044 m/s) and $q_{90}$ (1.532 m/s).
5. `demand_conformal_calibration_curve.png`:
   Calibration diagnostic curve showing perfect calibration on validation, but visible distribution shift degradation on test.
6. `demand_conformal_intervals_test.png`:
   Chronological 240-hour trajectory of actual building demand (kWh), `DemandLSTM` point forecasts, and 80% and 90% conformal bands.
7. `demand_conformal_residuals.png`:
   Histogram of calibration demand nonconformity scores with $q_{80}$ (2.478 kWh) and $q_{90}$ (4.763 kWh).

---

## 7. Experiment Registry Tracking

Four formal uncertainty evaluation records were logged in `experiments/experiment_registry.csv` without modifying or overwriting any prior entries:

| Experiment ID | Dataset | Model / Method | Partition | Nominal | Empirical Cov. | MPIW | Winkler Score | Metric Unit | Status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP_WIND_CP_80` | NREL India Wind | SplitConformal_WindLSTM | test | 80% | 84.22% | 2.0877 | 3.0501 | m/s | COMPLETED |
| `EXP_WIND_CP_90` | NREL India Wind | SplitConformal_WindLSTM | test | 90% | 92.83% | 3.0602 | 3.9157 | m/s | COMPLETED |
| `EXP_DEMAND_CP_80` | BDG2 Demand | SplitConformal_DemandLSTM | test | 80% | 66.03% | 4.9561 | 15.4839 | kWh | COMPLETED |
| `EXP_DEMAND_CP_90` | BDG2 Demand | SplitConformal_DemandLSTM | test | 90% | 85.47% | 9.5254 | 20.6045 | kWh | COMPLETED |

---

## 8. Unit Test Suite Status

The uncertainty estimation module and all dependent pipeline components were validated using pytest:
```
python -m pytest tests/ -q
======================= 162 passed, 14 warnings in 7.65s =======================
```
- Total tests: **162 passed, 0 failed, 0 skipped (100% passing)**.
- New unit tests in `tests/test_uncertainty.py` (15 tests) cover:
  1. Finite-sample quantile correction formula and index bounds ($k = \lceil (n+1)(1-\alpha) \rceil$).
  2. Quantile monotonicity across confidence levels ($\alpha$).
  3. Extreme small-sample clamping.
  4. Input error handling: invalid $\alpha$, empty arrays, non-finite values (NaN, Inf), 2D shapes, unfitted access.
  5. PyTorch tensor interoperability.
  6. Strict zero-leakage test isolation.
  7. Physical clipping width reduction and coverage preservation.
  8. Inverted bound detection.
  9. Winkler score penalty behavior under coverage and miscoverage.

---

## 9. Conclusion & Readiness for Stage 5

Stage 4 establishes a leakage-safe, empirically audited uncertainty estimation framework for sequence forecasting models in the DL MiniProject. The project now possesses:
1. Four trained and benchmarked forecasting models (Solar PV Nowcasting, NASA Irradiance, Gujarat Wind, Florida Demand).
2. Leakage-safe conformal prediction intervals with quantified empirical coverage, MPIWs, and Winkler scores.
3. Documented mathematical understanding of time-series exchangeability limitations under seasonal distribution shift.

The repository is fully primed for **Stage 5: Uncertainty-Aware Decision-Support PoC (Model E)**, where wind power proxies and demand forecasting intervals will be combined in location-specific feasibility analysis.
