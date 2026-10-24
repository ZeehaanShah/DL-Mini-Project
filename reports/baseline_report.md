# Baseline Model Evaluation Report

**Last Updated:** 2026-10-21 19:54:31  
**Auditor:** Automated Baseline Benchmark Framework (`scripts/evaluate_baselines.py`, `scripts/evaluate_demand_baselines.py`)

This report documents empirical validation baseline benchmarks for:
1. **Model A**: SKIPP'D Solar PV Nowcasting ($I_t \to P_t$ in kW)
2. **Model B**: NASA POWER Irradiance Forecasting ($X_{[t-23, t]} \to y_{t+1}$ in Wh/m²)
3. **Model D**: BDG2 Building Electricity Demand Forecasting ($X_{[t-23, t]} \to y_{t+1}$ in kWh)

> [!IMPORTANT]
> All baseline parameters were fitted strictly on **training partitions**.
> Model selection and baseline comparison are conducted strictly on **validation partitions**.
> The official test sets remain held out for final evaluation.

---

## 1. Summary of Baseline Results (Validation Set)

| Experiment ID | Dataset | Model Name | Partition | N Samples | MAE | RMSE | R² | Unit |
|---|---|---|---|---|---|---|---|---|
| `EXP_SOLAR_BASE_MEAN` | SKIPP'D | GlobalMeanBaseline | val | 12,818 | 7.0984 | 7.9990 | -0.0088 | kW |
| `EXP_SOLAR_BASE_MEDIAN` | SKIPP'D | GlobalMedianBaseline | val | 12,818 | 7.0644 | 8.2171 | -0.0646 | kW |
| `EXP_SOLAR_BASE_CLIM` | SKIPP'D | HourOfDayClimatologyBaseline | val | 12,818 | 1.8795 | 2.7114 | 0.8841 | kW |
| `EXP_SOLAR_BASE_PERSIST_PAIRS` | SKIPP'D | ConditionalPersistenceBaseline | val_pairs_le_120s | 12,783 | 0.1394 | 0.4400 | 0.9969 | kW |
| `EXP_SOLAR_BASE_PERSIST_FALLBACK` | SKIPP'D | ConditionalPersistenceWithClimatologyFallback | val | 12,818 | 0.1435 | 0.4566 | 0.9967 | kW |
| `EXP_NASA_BASE_PERSIST_1H` | NASA_POWER | NasaOneHourPersistenceBaseline | val | 1,290 | 61.1625 | 96.5181 | 0.8707 | Wh/m^2 |
| `EXP_NASA_BASE_SEASONAL_NAIVE` | NASA_POWER | NasaSeasonalNaiveBaseline | val | 1,290 | 23.0155 | 53.7911 | 0.9598 | Wh/m^2 |
| `EXP_WIND_BASE_PERSIST_VAL` | NREL_WIND_TOOLKIT | WindPersistenceBaseline | val | 1,290 | 0.7644 | 1.2845 | 0.6665 | m/s |
| `EXP_WIND_BASE_SEASONAL_VAL` | NREL_WIND_TOOLKIT | WindSeasonalNaiveBaseline | val | 1,290 | 1.4190 | 1.9749 | 0.2117 | m/s |
| `EXP_DEMAND_BASE_PERSIST_VAL` | BDG2 | DemandPersistenceBaseline | val | 2,607 | 1.8052 | 3.6647 | 0.8107 | kWh |
| `EXP_DEMAND_BASE_SEASONAL_24H_VAL` | BDG2 | DemandSeasonalNaiveBaseline | val | 2,607 | 5.6797 | 9.1857 | -0.1892 | kWh |
| `EXP_DEMAND_BASE_WEEKLY_168H_VAL` | BDG2 | DemandWeeklySeasonalNaiveBaseline | val | 2,463 | 6.5699 | 9.7644 | -0.3793 | kWh |

### Confirmatory Held-Out Test Set Results (Pre-Fixed Protocol)

| Experiment ID | Dataset | Model Name | Partition | N Samples | MAE | RMSE | R² | Unit |
|---|---|---|---|---|---|---|---|---|
| `EXP_WIND_BASE_PERSIST_TEST` | NREL_WIND_TOOLKIT | WindPersistenceBaseline | test | 1,242 | 0.6002 | 0.8679 | 0.7295 | m/s |
| `EXP_WIND_BASE_SEASONAL_TEST` | NREL_WIND_TOOLKIT | WindSeasonalNaiveBaseline | test | 1,242 | 1.2661 | 1.6807 | -0.0142 | m/s |
| `EXP_DEMAND_BASE_PERSIST_TEST` | BDG2 | DemandPersistenceBaseline | test | 2,561 | 2.5336 | 4.0823 | 0.9228 | kWh |
| `EXP_DEMAND_BASE_SEASONAL_24H_TEST` | BDG2 | DemandSeasonalNaiveBaseline | test | 2,561 | 6.7775 | 9.9977 | 0.5369 | kWh |
| `EXP_DEMAND_BASE_WEEKLY_168H_TEST` | BDG2 | DemandWeeklySeasonalNaiveBaseline | test | 2,417 | 11.7533 | 17.8784 | -1.3791 | kWh |

---

## 2. Model A (SKIPP'D) Baseline Taxonomy & Findings

> [!NOTE]
> **Benchmark Category Distinction**:
> - **Primary Autonomous Image Benchmark (`HourOfDayClimatologyBaseline`)**: Uses only the timestamp's diurnal local hour to estimate solar geometry. Suitable for autonomous sky cameras without power telemetry. Achieves **MAE = 1.88 kW** (system rated capacity: 30 kW).
> - **Active Telemetry Heuristic (`ConditionalPersistenceBaseline`)**: Assumes an active physical power meter streams $P_{t-1}$ within $\le 2$ minutes. Achieves **MAE = 0.14 kW** on the 12,783 eligible consecutive pairs (99.7% of validation set). It cannot be used by a pure camera model.
> - **Unconditioned Reference Bounds**: Global Mean (MAE = 7.10 kW) and Global Median (MAE = 7.06 kW).

---

## 3. Model B (NASA POWER) Baseline Taxonomy & Findings

> [!NOTE]
> **Causal Lag Derivation**:
> - **1-Hour Persistence (`NasaOneHourPersistenceBaseline`)**: Uses the final observation in the 24h input window (observation at forecast origin $t$) to predict $t+1$. Achieves **MAE = 61.16 Wh/m²** ($R^2 = 0.8707$).
> - **Same-Hour-Yesterday Seasonal Naive (`NasaSeasonalNaiveBaseline`)**: Predicts the observation exactly 24 hours prior to the target: $(t + 1) - 24 = t - 23\text{ hours}$. For an input window covering $[t-23, \dots, t]$, this corresponds strictly to index $0$ of the window. Achieves **MAE = 23.02 Wh/m²** ($R^2 = 0.9598$).

---

## 4. Model D (BDG2 Demand) Baseline Taxonomy & Findings

> [!NOTE]
> **Empirical Demand Dynamics for Panther_education_Rosalie**:
> - **1-Hour Persistence (`DemandPersistenceBaseline`)**: Predicts $\hat{y}_{t+1} = y_t$ (the electricity consumption at forecast origin $t$).
>   - **Validation**: **MAE = 1.81 kWh**, **RMSE = 3.66 kWh**, **R² = 0.8107** ($N = 2,607$).
>   - **Test**: **MAE = 2.53 kWh**, **RMSE = 4.08 kWh**, **R² = 0.9228** ($N = 2,561$).
>   - **Physical Insight**: High operational thermal inertia and steady occupancy schedules make the most recent hourly consumption by far the strongest baseline predictor.
> - **24-Hour Diurnal Seasonal Naive (`DemandSeasonalNaiveBaseline`)**: Predicts $\hat{y}_{t+1} = y_{t-23}$ (same hour yesterday).
>   - **Validation**: **MAE = 5.68 kWh**, **RMSE = 9.19 kWh**, **R² = -0.1892** ($N = 2,607$).
>   - **Test**: **MAE = 6.78 kWh**, **RMSE = 10.00 kWh**, **R² = 0.5369** ($N = 2,561$).
>   - **Physical Insight**: Daily temperature fluctuations (cooling load in Florida) and weekday vs weekend transitions degrade simple 24-hour lag performance compared to 1-hour persistence.
> - **168-Hour Weekly Seasonal Naive (`DemandWeeklySeasonalNaiveBaseline`)**: Predicts $\hat{y}_{t+1} = y_{t-167}$ (same hour last week).
>   - **Validation**: **MAE = 6.57 kWh**, **RMSE = 9.76 kWh**, **R² = -0.3793** ($N = 2,463$).
>   - **Test**: **MAE = 11.75 kWh**, **RMSE = 17.88 kWh**, **R² = -1.3791** ($N = 2,417$).
>   - **Physical Insight**: Weekly seasonality suffers from seasonal drift (summer break transition into fall semester at UCF) and calendar anomalies (labor day, exam periods).
> 
> **Controlled Comparison on Identical 168h Eligible Validation Samples ($N = 2,463$)**:
> - 1-Hour Persistence: MAE **1.84 kWh** | RMSE **3.71 kWh** | $R^2 = \mathbf{0.8004}$
> - 24-Hour Seasonal Naive: MAE **5.51 kWh** | RMSE **8.90 kWh** | $R^2 = -0.1467$
> - 168-Hour Weekly Seasonal: MAE **6.57 kWh** | RMSE **9.76 kWh** | $R^2 = -0.3793$

---

## 5. Model C (NREL India Wind) Baseline Taxonomy & Findings

> [!NOTE]
> **Empirical Wind Resource Dynamics for Gujarat Coastal Site (100m AGL)**:
> - **1-Hour Persistence (`WindPersistenceBaseline`)**: Predicts $\hat{v}_{t+1} = v_t$ (the physical wind speed at forecast origin $t$).
>   - **Validation Speed**: **MAE = 0.7644 m/s**, **RMSE = 1.2845 m/s**, **R² = 0.6665** ($N = 1,290$).
>   - **Validation Power Proxy**: **MAE = 86.06 kW**, **RMSE = 176.80 kW**, **R² = 0.5403** (Generic 2.0 MW turbine proxy).
>   - **Test Speed**: **MAE = 0.6002 m/s**, **RMSE = 0.8679 m/s**, **R² = 0.7295** ($N = 1,242$).
>   - **Test Power Proxy**: **MAE = 50.22 kW**, **RMSE = 80.23 kW**, **R² = 0.7010**.
>   - **Physical Insight**: Near-surface and boundary-layer atmospheric inertia ensures high short-term correlation, making 1-hour persistence a strong benchmark.
> - **24-Hour Diurnal Seasonal Naive (`WindSeasonalNaiveBaseline`)**: Predicts $\hat{v}_{t+1} = v_{t-23}$ (same hour yesterday).
>   - **Validation Speed**: **MAE = 1.4190 m/s**, **RMSE = 1.9749 m/s**, **R² = 0.2117** ($N = 1,290$).
>   - **Validation Power Proxy**: **MAE = 152.72 kW**, **RMSE = 248.15 kW**, **R² = 0.0944**.
>   - **Test Speed**: **MAE = 1.2661 m/s**, **RMSE = 1.6807 m/s**, **R² = -0.0142** ($N = 1,242$).
>   - **Test Power Proxy**: **MAE = 105.35 kW**, **RMSE = 151.85 kW**, **R² = -0.0709**.
>   - **Physical Insight**: Synoptic atmospheric pressure systems and frontal weather movements evolve across multi-day cycles rather than repeating strictly every 24 hours, causing 24h seasonal naive to perform poorly ($R^2 \approx 0.21$ val, $-0.01$ test).

---

## 6. Benchmark Metric Formulas

- **MAE** (Mean Absolute Error): $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
- **RMSE** (Root Mean Squared Error): $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **R²** (Coefficient of Determination): $1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$