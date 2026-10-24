# Data Audit Report

**Project**: DL MiniProject — Renewable Energy Forecasting  
**Date**: 2026-10-20  
**Phase**: 1 — Pre-Training Data Audit  
**Auditor**: Automated audit scripts + manual verification

---

## Executive Summary

Four dataset directories were inspected. All four datasets contain validated data (SKIPP'D solar, NASA POWER irradiance, NREL India wind, and BDG2 building electricity demand).

| Dataset | Status | Rows | Quality | Ready for Training? |
|---------|--------|------|---------|---------------------|
| SKIPP'D Solar | PARTIAL (1/5 shards) | 69,875 train + 14,003 test | Good (no NaN/negatives) | Yes (with caveats) |
| NASA POWER | COMPLETE | 8,760 | Excellent (no gaps) | Yes |
| NREL India Wind | COMPLETE | 8,760 | Excellent (100% continuous, 100m AGL) | Yes (Pilot verified) |
| BDG2 Demand | COMPLETE | 17,544 | Validated (100% continuous, 20 causal-filled) | Yes (Pilot verified) |

---

## 1. SKIPP'D Solar Dataset

### 1.1 Source and Provenance

| Property | Value |
|---|---|
| Dataset name | SKIPP'D (SKy Images and Photovoltaic Power generation Dataset) |
| Original source | Stanford SLAC National Accelerator Laboratory |
| Reference | Nie et al., Stanford Solar Forecasting Dataset |
| Location | Stanford University campus, California, USA |
| Installation | Rooftop PV system at SLAC |
| HuggingFace source | `torchgeo/skippd` |
| Download format | Parquet (sharded) |
| License | To be verified from source |

### 1.2 Files Present

| File | Size | Status |
|---|---|---|
| `train-00000-of-00005.parquet` | 427.24 MB | Shard 1 of 5 |
| `test-00000-of-00001.parquet` | 86.12 MB | Complete (1 of 1) |
| `times_trainval.npy` | 10.33 MB | Complete (349,372 entries) |
| `times_test.npy` | 0.41 MB | Complete (14,003 entries) |

### 1.3 Schema

```
Schema (3 logical fields):
  [0] image: struct<bytes: binary, path: string>
  [1] time:  timestamp[ms, tz=US/Pacific]
  [2] pv:    float32
```

- **image**: 64×64 RGB PNG images of the sky, encoded as raw bytes within a struct. `path` is always None.
- **time**: Timestamp with US/Pacific timezone (`timestamp[ms, tz=US/Pacific]`).
- **pv**: Photovoltaic power output in **kW** (kilowatts), verified from Nie et al. (Stanford rooftop array, 30 kW rated capacity). Float32, strictly non-negative.

### 1.4 Quantitative Summary

| Metric | Train (shard 1) | Test Parquet |
|---|---|---|
| Rows | 69,875 | 14,003 |
| NaN values | 0 | 0 |
| Null values | 0 | 0 |
| PV min | 0.000162 kW | 0.000095 kW |
| PV max | 29.558525 kW | 29.570940 kW |
| PV mean | 13.736201 kW | 12.588559 kW |
| PV std | 8.513349 kW | 7.643997 kW |
| PV median | 15.114739 kW | 12.967851 kW |
| Negative PV | 0 | 0 |
| Zero PV | 0 | 0 |
| Image size | 64×64 | 64×64 |
| Image mode | RGB | RGB |
| Image format | PNG | PNG |

### 1.5 Temporal Analysis & Interleaved Split Verification

| Property | Train (shard 1) | Test Parquet | trainval (npy) | test (npy) |
|---|---|---|---|---|
| Start | 2017-03-09 06:47:00-08:00 | 2017-06-24 07:38:40-07:00 | 2017-03-09 06:47:00 | 2017-06-24 06:38:40 |
| End | 2017-09-04 10:14:00-07:00 | 2019-10-19 19:15:30-07:00 | 2019-10-26 17:59:20 | 2019-10-19 18:15:30 |
| Calendar days | 92 days | 20 days | ~960 days | ~20 days |
| Resolution | Mostly 1-min (mode: 60s) | Mostly 1-min | Irregular | Irregular |
| Sorted | Yes (monotonic) | Yes (monotonic) | Yes (monotonic) | No (non-monotonic) |
| Duplicates | 0 | 0 | 0 | 0 |
| Shared days (train shard ↔ test) | **0 days** (empirically verified) | | | |
| Shared timestamps (train shard ↔ test) | **0 timestamps** (empirically verified) | | | |

#### Clarification on Bounding-Box Range vs. Calendar Day Disjointness
A common point of confusion is how the train shard date range (March 9, 2017 – September 4, 2017) overlaps with the test set date range (June 24, 2017 – October 19, 2019) while having **zero shared calendar days**:
- The SKIPP'D benchmark uses an **interleaved calendar-day partition**: 20 complete, discrete calendar days across the 3-year period (2017–2019) were designated exclusively as test days.
  - The 20 test dates are: `2017-06-24`, `2017-09-15`, `2017-09-20`, `2017-10-06`, `2017-10-11`, `2017-10-22`, `2018-01-25`, `2018-02-16`, `2018-03-09`, `2018-06-12`, `2018-06-23`, `2018-10-04`, `2019-01-25`, `2019-05-27`, `2019-06-23`, `2019-06-28`, `2019-07-14`, `2019-08-10`, `2019-10-14`, `2019-10-19`.
- Within the temporal span of train shard 1 (`2017-03-09` to `2017-09-04`), **exactly one test day occurs**: `2017-06-24`.
- Direct row-level inspection of train shard 1 reveals that train data includes records for `2017-06-23` and `2017-06-25`, but **completely skips `2017-06-24` (0 records present)**.
- Furthermore, checking the full 349,372 entries of `times_trainval.npy` against all 20 test dates yields **0 overlapping dates** (`set()`).
- Thus, while the broad date ranges overlap chronologically, the data is strictly partitioned by calendar day.

### 1.6 Schema Binding and Authoritative Timestamps

#### Parquet Columnar Tuple Binding Guarantee
In Apache Parquet files, column chunks within each row group have identical row counts (100 rows per group in both train and test files). The Arrow schema enforces:
1. `image`: `struct<bytes: binary, path: string>`
2. `time`: `timestamp[ms, tz=US/Pacific]`
3. `pv`: `float`

By the Parquet file format specification, row $i$ in `image`, row $i$ in `time`, and row $i$ in `pv` are guaranteed to belong to the exact same physical observation record. The timestamp is embedded natively with timezone `US/Pacific` co-located with the image bytes and PV reading.

#### Why NumPy Arrays (.npy) Must NOT Be Used for Mapping
- `times_test.npy` is an external, decoupled array with no schema-level link to `test-00000-of-00001.parquet`.
- `times_test.npy` is **non-monotonic** (unsorted).
- 13,464 out of 14,003 timestamps in `times_test.npy` are offset by exactly 1 hour relative to Parquet timestamps (DST conversion differences during dataset repackaging).
- Therefore, the Parquet embedded `time` column is strictly authoritative. Never assign `.npy` timestamps to Parquet rows by positional index.

### 1.7 Critical Issues & Mitigations

#### ISSUE 1: Partial Training Data (SEVERITY: HIGH)
We have only shard 1 of 5 (69,875 / ~349,372 = ~20% of training data). This is sufficient for a proof-of-concept but:
- Results CANNOT be compared to official SKIPP'D benchmarks.
- Model performance will likely be lower than reported in the original paper.
- Must document this limitation clearly in all reports.

#### ISSUE 2: NPY-Parquet Test Misalignment (STATUS: RESOLVED / ENFORCED)
Resolved by declaring Parquet embedded timestamps as the single source of truth and rejecting positional array mapping.

#### ISSUE 3: Interleaved Train/Test Split (STATUS: MITIGATED / VERIFIED)
Verified: 0 shared days and 0 shared timestamps between train shard 1 and test parquet. Internal validation split is performed using `src/data/splits.py:split_skippd_by_day` (calendar-day split), guaranteeing zero same-day leakage.

#### ISSUE 4: PV Units Verified (STATUS: RESOLVED)
Confirmed via Nie et al. (Stanford University EAO Group):
- System: Rooftop PV array with 30 kW rated capacity located on Stanford University campus.
- Target unit: **kW** (kilowatts).
- Sensor: Ground-based fisheye camera located ~125m from the PV system.
- Max recorded values (~29.57 kW) align with the 30 kW rated array capacity.

### 1.8 Recommendations

1. Use only the parquet `time` column for all temporal indexing and alignment.
2. Partition the 69,875 samples in shard 1 using `split_skippd_by_day` (calendar-day split).
3. Evaluate on the 14,003-sample test parquet as the held-out test set.
4. Report PV metrics in **kW**, noting the 30 kW system scale and shard-1 training subset.

---

## 2. NASA POWER Dataset

### 2.1 Source and Provenance

| Property | Value |
|---|---|
| Source | NASA POWER (Prediction Of Worldwide Energy Resources) |
| API | POWER Point Hourly Data |
| Location | Latitude 19.076°N, Longitude 72.878°E (Mumbai, India) |
| Elevation | 97.15m (MERRA-2 grid average) |
| Time zone | LST (Local Standard Time, IST = UTC+5:30) |
| Date range | 2023-01-01 through 2023-12-31 |
| Data source | CERES SYN1deg (radiation), MERRA-2 (temperature, humidity, wind) |

### 2.2 File Details

| Property | Value |
|---|---|
| File | `POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv` |
| Size | 296.2 KB |
| Total lines | 8,773 (12 header + 1 column header + 8,760 data rows) |
| Sentinel value | -999 (documented; none found in data) |

### 2.3 Schema and Parameters

| Column | Description | Unit | Source |
|---|---|---|---|
| YEAR | Year | integer | Calendar |
| MO | Month | integer | Calendar |
| DY | Day | integer | Calendar |
| HR | Hour (0-23) | integer | Calendar |
| ALLSKY_SFC_SW_DWN | All Sky Surface Shortwave Downward Irradiance | Wh/m² | CERES SYN1deg |
| T2M | Temperature at 2 Meters | °C | MERRA-2 |
| RH2M | Relative Humidity at 2 Meters | % | MERRA-2 |
| WS10M | Wind Speed at 10 Meters | m/s | MERRA-2 |

### 2.4 Quantitative Summary

| Parameter | Count | Min | Max | Mean | Std | 25% | 50% | 75% |
|---|---|---|---|---|---|---|---|---|
| ALLSKY_SFC_SW_DWN | 8,760 | 0.00 | 1,008.53 | 211.63 | 290.57 | 0.00 | 6.81 | 404.19 |
| T2M | 8,760 | 12.82 | 42.76 | 27.00 | 5.01 | 24.20 | 26.69 | 29.61 |
| RH2M | 8,760 | 7.95 | 100.00 | 66.91 | 24.08 | 48.23 | 73.41 | 87.11 |
| WS10M | 8,760 | 0.07 | 11.28 | 3.61 | 1.79 | 2.31 | 3.21 | 4.71 |

### 2.5 Data Quality

| Check | Result |
|---|---|
| Missing values (NaN) | 0 across all columns |
| Sentinel values (-999) | 0 found |
| Duplicate rows | 0 |
| Expected hours (365×24) | 8,760 ✓ |
| Temporal completeness | Complete (no gaps) |
| Negative irradiance | 0 (physically consistent) |
| Temperature range | Plausible for Mumbai |
| Humidity range | Plausible (7.95–100%) |
| Wind speed range | Plausible for 10m height |

### 2.6 Important Notes

1. **ALLSKY_SFC_SW_DWN is Wh/m²**: This is hourly accumulated energy, not instantaneous power (W/m²). For a 1-hour interval, the numerical value equals the average irradiance in W/m² over that hour.
2. **Satellite-derived reanalysis**: This is modeled data from CERES and MERRA-2, not ground-station measurements. It has inherent smoothing and may not capture rapid local variations.
3. **WS10M at 10 meters**: Wind turbine hub heights are typically 80–100m. Any wind power estimation would require height extrapolation.
4. **Location incompatibility**: Mumbai (India) has no physical relationship with the Stanford SKIPP'D installation (California, USA). These datasets MUST remain separate experiments.

### 2.7 Recommendations

1. Suitable for a next-hour solar radiation prediction experiment.
2. Chronological split is appropriate (full year with seasonal variation).
3. Persistence baseline: use the current hour's value as the next-hour forecast.
4. The radiation prediction is a weather/radiation forecast, NOT a PV power forecast.

---

## 3. NREL India Wind Dataset (Model C)

### 3.1 Source and Provenance

| Property | Value |
|---|---|
| Dataset name | NREL India Wind Toolkit / India Renewable Integration Study (IRIS) |
| Source agency | National Renewable Energy Laboratory (NREL) / National Laboratory of the Rockies (NLR) |
| Study partnership | NREL, USAID, Ministry of New and Renewable Energy (MNRE), POSOCO |
| Data nature | Modeled meso-scale atmospheric numerical simulation (WRF 3km grid reanalysis), **NOT physical anemometer tower measurements** |
| Endpoint used | `https://developer.nlr.gov/api/wind-toolkit/v2/wind/india-wind-download.csv` |
| Acquisition date | 2026-10-21 (Stage 2A Controlled Pilot) |
| Requested location | Gujarat coastal wind corridor (23.25°N, 69.67°E) |
| Actual WRF grid cell | SiteID: 42395, Latitude: 23.24346924°N, Longitude: 69.6620636°E |
| Measurement height | 100m Above Ground Level (AGL) |
| Temporal coverage | Full calendar year 2014 (365 days, non-leap year) |
| Temporal resolution | 1-hour interval (`interval=60`), timestamps centered at :30 UTC |

### 3.2 Files Present

| File | Size | Role | Status |
|---|---|---|---|
| `data/raw/wind/nrel_india_wind_2014_23.25N_69.67E.csv` | 389.95 KB | Original raw API response | Immutable, Complete |
| `data/raw/wind/nrel_india_wind_2014_23.25N_69.67E_metadata.json` | 1.15 KB | Machine-readable provenance sidecar | Complete |
| `data/processed/wind/wind_hourly_pilot.csv` | 260.25 KB | Standardized, validated hourly series | Complete (8,760 rows) |

### 3.3 Quantitative Summary & Data Quality

| Metric | wind_speed (m/s) | wind_direction (deg) | temperature (°C) | pressure (hPa) |
|---|---|---|---|---|
| Rows | 8,760 | 8,760 | 8,760 | 8,760 |
| Missing (NaN/null) | 0 | 0 | 0 | 0 |
| Infinite values | 0 | 0 | 0 | 0 |
| Sentinels (<= -900) | 0 | 0 | 0 | 0 |
| Minimum | 0.08 | 0.00 | 10.26 | 970.70 |
| Mean | 6.93 | 208.70 | 25.82 | 983.49 |
| Maximum | 18.64 | 360.00 | 40.25 | 995.10 |
| Physical check | Pass (0.0 to 75.0 m/s) | Pass (0.0 to 360.0°) | Pass (plausible Gujarat) | Pass (plausible barometric) |

### 3.4 Temporal Continuity & Resolution

- **Start timestamp**: `2014-01-01 00:30:00`
- **End timestamp**: `2014-12-31 23:30:00`
- **Duplicate timestamps**: 0
- **Temporal discontinuities**: 0 (all 8,759 consecutive differences are strictly `Timedelta(hours=1)`)
- **Actual vs Expected row count**: Exactly 8,760 rows (matches 365 days × 24 hours).

### 3.5 Processing Transformations Documented

1. **Header metadata extraction**: Extracted SiteID (`42395`), WRF Grid Latitude (`23.24346924`), and Longitude (`69.6620636`) from row 1.
2. **Datetime reconstruction**: Combined discrete columns `Year, Month, Day, Hour, Minute` into a single ISO-compatible `datetime` column.
3. **Column name standardization**:
   - `wind speed at 100m (m/s)` $\rightarrow$ `wind_speed`
   - `wind direction at 100m (deg)` $\rightarrow$ `wind_direction`
   - `temperature at 100m (C)` $\rightarrow$ `temperature`
4. **Unit normalization**: Air pressure was provided in Pascals (`Pa`) by NREL; converted to hectopascals (`hPa`) by dividing by 100.0: `pressure = pressure_Pa / 100.0`.
5. **Schema validation**: Passed `validate_wind_dataframe` with zero exceptions.

### 3.6 Target Definition & Turbine Power Disclaimer

- The machine learning model forecasts **meteorological wind speed** in **m/s**.
- Any electrical power estimation ($P_{\text{est}}$ in kW) generated by `TurbinePowerCurve` represents an **illustrative engineering approximation** based on a generic 2.0 MW cubic power proxy.
- **Strict Constraint**: Estimated turbine power must NEVER be characterized or described as measured wind-farm generation or physical turbine SCADA output.


---

## 4. BDG2 Electricity Demand Dataset (Model D)

### 4.1 Source and Provenance

| Property | Value |
|---|---|
| Dataset name | Building Data Genome Project 2 (BDG2) |
| Source repository | `https://github.com/buds-lab/building-data-genome-project-2` |
| Primary publication | Miller et al., *The Building Data Genome Project 2*, Scientific Data 7, 368 (2020). DOI: `10.1038/s41597-020-00712-x` |
| Permanent archive | Zenodo Record `10.5281/zenodo.3887306` |
| License | Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0) |
| Target building ID | `Panther_education_Rosalie` (verified at column index 53 in electricity matrix) |
| Site ID & location | `Panther` — University of Central Florida (UCF), Orlando, Florida, USA |
| Coordinates | Latitude: 28.5177°N, Longitude: -81.3790°W |
| Primary space usage | Education (Research facility) |
| Gross floor area | 690.5 m² (7,432 sq ft) |
| Year built | 2008 |
| Temporal coverage | Full 2 calendar years: 2016-01-01 00:00:00 to 2017-12-31 23:00:00 |
| Total observations | Exactly 17,544 hourly observations (8,784 in leap year 2016 + 8,760 in non-leap year 2017) |
| Measurement units | Reported in **kWh** (`kWhsum`, hourly accumulated energy consumption) |
| Equivalent power | Over a 1-hour interval ($\Delta t = 1\text{ h}$), numerical kWh equals average active power in **kW** |

### 4.2 Files Present

| File | Size | Role | Status |
|---|---|---|---|
| `data/raw/bdg2/bdg2_Panther_education_Rosalie_raw.csv` | 497.54 KB | Extracted raw meter series (2 columns: `timestamp`, `Panther_education_Rosalie`) | Immutable Raw Extract |
| `data/raw/bdg2/metadata_Panther_education_Rosalie.csv` | 276 B | Extracted official building metadata row | Complete |
| `data/raw/bdg2/bdg2_Panther_education_Rosalie_cleaned_official.csv` | 468.91 KB | Official author-cleaned meter extract from Miller et al. | Reference Benchmark |
| `data/raw/bdg2/imputation_audit_Panther_education_Rosalie.json` | 7.45 KB | Machine-readable record of all 20 causal-filled timestamps | Complete |
| `data/raw/bdg2/bdg2_Panther_education_Rosalie_raw_metadata.json` | 1.45 KB | Machine-readable provenance and statistics sidecar | Complete |
| `data/processed/demand/demand_hourly_pilot.csv` | 428.32 KB | Validated standardized hourly series (`datetime`, `demand_kwh`) | Complete (17,544 rows) |

### 4.3 Timezone Semantics & Daylight-Saving Transition Verification

- **Timezone**: `US/Eastern` Local Standard Time (EST / UTC-5).
- **Index Encoding**: Timestamps are formatted as naive ISO strings `YYYY-MM-DD HH:MM:SS`. BDG2 constructs an unbroken uniform sequence of exactly 24 rows per calendar day ($24 \times 366 = 8,784$ hours in 2016 and $24 \times 365 = 8,760$ hours in 2017, totaling exactly 17,544 rows). The table index itself does not drop or duplicate rows during DST transitions.
- **DST Physical Phenomenon Discovered**: On March 12, 2017 at 02:00:00 (the exact hour when local US/Eastern clocks "spring forward" to 03:00:00 EDT), the row `2017-03-12 02:00:00` is physically present in the table (row index 10466), but its meter reading is `NaN`. This missing value is a **real missing sensor observation** within a uniform calendar table: the local building data logger was tied to local wall-clock time (which jumped directly from 01:59:59 EST to 03:00:00 EDT), producing no physical reading for 02:00:00. It is not a timezone-conversion drop of the timestamp row or a table formatting artifact.
- **Handling**: Preserved the uniform 1-hour interval frequency; the 1-hour sensor dropout at row 10466 was causally filled from the immediately preceding hour ($64.8125\text{ kWh}$) with full provenance documentation and zero lookahead bias.

### 4.4 Quantitative Summary & Data Quality

| Metric | Raw Series (`Panther_education_Rosalie`) | Active Period (May 20, 2016 – Dec 31, 2017) | Processed Series (`demand_kwh`) |
|---|---|---|---|
| Total rows | 17,544 | 14,171 | 17,544 |
| Missing values (NaN) | 20 (0.11%) | 20 (0.14%) | 0 (100% complete) |
| Infinite / Non-numeric | 0 | 0 | 0 |
| Sentinel values ($\le -900$) | 0 | 0 | 0 |
| Negative values ($< 0.0$) | 0 | 0 | 0 |
| Zero readings ($== 0.0$) | 3,373 (early 2016 unmetered period) | 0 (strictly positive active load) | 3,373 |
| Minimum | 0.00 kWh | 11.00 kWh | 0.00 kWh |
| 25th Percentile | 49.41 kWh | 55.41 kWh | 49.41 kWh |
| Median (50th) | 60.61 kWh | 64.61 kWh | 60.61 kWh |
| Mean | 52.47 kWh | 64.98 kWh | 52.47 kWh |
| 75th Percentile | 71.01 kWh | 73.01 kWh | 71.01 kWh |
| Maximum | 131.30 kWh | 131.30 kWh | 131.30 kWh |
| Standard Deviation | 28.25 kWh | 13.26 kWh | 28.25 kWh |

### 4.5 Empirical Comparison: Raw vs. Official Cleaned Series

Direct comparison between `electricity.csv` (raw) and `electricity_cleaned.csv` (official Miller et al. cleaned matrix) revealed key domain insights:
1. **Early 2016 Commissioning/Unmetered Period**: From `2016-01-01 00:00:00` to `2016-05-20 17:00:00` (the first 3,373 hours of 2016), the raw meter recorded continuous blocks of `0.0` kWh. In `electricity_cleaned.csv`, Miller et al. flagged these 3,373 inactive readings as unmetered/commissioning artifacts and converted them to `NaN`.
2. **Active Operational Period**: From `2016-05-20 18:00:00` onward, the meter began recording continuous active building operations (mean: 64.98 kWh, min: 11.00 kWh, max: 131.30 kWh, zero zero-readings).
3. **Four Isolated Transmission Gaps in Active Period**:
   - Gap 1: `2017-02-14 19:00:00` to `2017-02-14 23:00:00` (5 hours, evening load: $52.61\text{ kWh}$)
   - Gap 2: `2017-03-12 02:00:00` to `2017-03-12 02:00:00` (1 hour, DST clock transition: $64.81\text{ kWh}$)
   - Gap 3: `2017-06-19 19:00:00` to `2017-06-20 00:00:00` (6 hours, evening load: $77.41\text{ kWh}$)
   - Gap 4: `2017-11-20 21:00:00` to `2017-11-21 04:00:00` (8 hours, overnight load: $32.21\text{ kWh}$)

### 4.6 Defensible Cleaning Policy & Imputation Audit

- **Cleaning Policy**: Causal forward-fill bounded by `max_causal_gap_hours = 8`.
- **Zero Lookahead Bias**: No future observations were utilized to impute earlier values ($\hat{y}_t = y_{t_{\text{last\_valid}}}$).
- **Audit Artifact**: All 20 imputed rows are machine-readably tracked in `data/raw/bdg2/imputation_audit_Panther_education_Rosalie.json` with exact row index, timestamp, original value (`null`), replacement value, method (`bounded_causal_forward_fill_limit_8h`), and justification.
- **Physical Bounds**: Validated using `validate_demand_dataframe`: non-negative ($0.0 \le y$), plausibility upper bound ($y \le 1000.0\text{ kWh}$), zero temporal discontinuities.

### 4.7 Chronological Splits & Zero Leakage

Under the project-wide 70% Train / 15% Validation / 15% Test chronological split with 24-hour embargo buffers:
- **Train partition**: 12,280 hours (`2016-01-01 00:00:00` to `2017-05-26 15:00:00`). Contains the early 2016 unmetered zeros and 6 NaNs (Feb 14 and Mar 12, 2017).
- **Embargo 1**: 24 hours purged (`2017-05-26 16:00:00` to `2017-05-27 15:00:00`). Zero sample overlap.
- **Validation partition**: 2,631 hours (`2017-05-27 16:00:00` to `2017-09-14 06:00:00`). Contains **0 zero values** (100% active operational consumption) and 6 NaNs (June 19-20, 2017).
- **Embargo 2**: 24 hours purged (`2017-09-14 07:00:00` to `2017-09-15 06:00:00`). Zero sample overlap.
- **Test partition**: 2,585 hours (`2017-09-15 07:00:00` to `2017-12-31 23:00:00`). Contains **0 zero values** (100% active operational consumption) and 8 NaNs (Nov 20-21, 2017).
- **Scaler Isolation**: `StandardScaler` parameters are fitted strictly on the Train partition; zero test or validation observations are accessed during scaling.

---

## 5. Cross-Dataset Compatibility

| Pair | Compatible? | Reason |
|---|---|---|
| SKIPP'D ↔ NASA POWER | **NO** | Different locations (Stanford, CA vs Mumbai, India), different years (2017-2019 vs 2023) |
| SKIPP'D ↔ Wind | **NO** | Stanford, CA vs Gujarat coastal corridor (2014) |
| SKIPP'D ↔ BDG2 | **NO** | Stanford, CA vs Orlando, FL (2016-2017) |
| NASA POWER ↔ Wind | **NO** | Mumbai (2023) vs Gujarat (2014); distinct microclimates and years |
| NASA POWER ↔ BDG2 | **NO** | Mumbai (India) vs Orlando (Florida, USA) |
| Wind ↔ BDG2 | **NO** | Gujarat (India, 2014) vs Orlando (Florida, 2016-2017) |

> [!CRITICAL]
> **Defensible Energy-Mix Constraint**: All four real datasets originate from physically independent geographies and disparate calendar years. The final decision-support module (Model E) must NEVER claim that these datasets represent co-located physical assets. Any multi-energy feasibility demonstration must be explicitly labeled as an architectural proof-of-concept integrating heterogeneous regional timeseries.

---

## 6. Audit Scripts and Automated Test Suites

All audit scripts and test suites are stored in `scripts/` and `tests/` and can be executed via:

```bash
# Data audit inspection scripts
python scripts/check_env.py              # Environment audit
python scripts/audit_data.py             # Full data audit
python scripts/audit_skippd_deep.py      # SKIPP'D deep inspection
python scripts/verify_skippd_alignment.py # Timestamp alignment
python scripts/download_bdg2_meter.py --skip-download # Re-run BDG2 demand validation audit
python scripts/evaluate_demand_baselines.py # Evaluate real demand baselines

# Complete Test Suite (140 tests, 100% passing, 0 skipped, 0 failed)
python -m pytest tests/ -q
```

---

## 7. Conclusion & Next Safe Steps

**Audit Phase Status**: **COMPLETE (ALL 4 DATASETS ACQUIRED & VALIDATED)**

1. **Model A (SKIPP'D Solar PV Nowcasting)**:
   - Data status: APPROVED (69,875 train + 14,003 test samples).
   - Baselines: Climatology (MAE 1.88 kW), Cond. Persistence (MAE 0.14 kW).
   - Deep Learning: S1–S4 completed; held-out test MAE 2.30 kW ($R^2 = 0.8052$).

2. **Model B (NASA POWER Weather/Irradiance)**:
   - Data status: APPROVED (8,760 hourly records, Mumbai 2023).
   - Baselines: 24h Seasonal Naive (MAE 23.02 Wh/m²), 1h Persistence (MAE 61.16 Wh/m²).
   - Deep Learning: W1–W4 completed; held-out test MAE 16.26 Wh/m² ($R^2 = 0.9844$).

3. **Model C (NREL India Wind 100m Speed & Power Proxy)**:
   - Data status: **ACQUIRED & VALIDATED** — 8,760 hourly records (2014, Gujarat coastal grid cell).
   - Baselines: Persistence and Seasonal Naive implemented and verified.
   - Status: Ready for baseline evaluation and neural network training (`WindLSTM`).

4. **Model D (BDG2 Electricity Demand)**:
   - Data status: **ACQUIRED & VALIDATED** — 17,544 hourly records (2016–2017, `Panther_education_Rosalie`, Orlando, FL).
   - Imputation: 20 short-gap hours causally imputed without lookahead; machine-readable audit verified.
   - Baselines: Evaluated on real validation split:
     - **1-Hour Persistence**: MAE 1.8052 kWh, RMSE 3.6647 kWh, $R^2 = 0.8107$ (Dominant baseline)
     - **24-Hour Seasonal Naive**: MAE 5.6797 kWh, RMSE 9.1857 kWh, $R^2 = -0.1892$
     - **168-Hour Weekly Seasonal Naive**: MAE 6.5699 kWh, RMSE 9.7644 kWh, $R^2 = -0.3793$
   - Status: Ready for neural network sequence modeling (`DemandLSTM`).

**Next Recommended Steps**:
1. Implement and evaluate baselines on real NREL wind dataset (Model C).
2. Train sequence neural network models (`WindLSTM` for wind speed, `DemandLSTM` for demand).
3. Implement uncertainty estimation (Monte Carlo Dropout / Deep Ensembles / conformal intervals).
4. Build decision-support feasibility module (Model E).
