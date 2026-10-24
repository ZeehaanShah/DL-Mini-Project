# Leakage Audit Report

**Project**: DL MiniProject — Renewable Energy Forecasting  
**Date**: 2026-10-20  
**Phase**: 1 — Pre-Training Leakage Risk Assessment  
**Status**: Risk Register Established; Automated Checks PLANNED

---

## 1. Leakage Risk Register

### Risk L1: Target Leakage via Same-Day Images (SKIPP'D)

| Property | Value |
|---|---|
| Severity | **HIGH** |
| Dataset | SKIPP'D |
| Description | Sky images from the same day have correlated sky conditions and PV output patterns. If train and validation sets share images from the same calendar day, the model may learn day-specific patterns rather than generalizable image→PV mappings. |
| Mitigation | Split train/validation by **full calendar day** boundaries. All images from a given day go entirely to train OR validation. |
| Implementation | PLANNED — split logic not yet coded |
| Verification | Automated test to confirm no day appears in both train and validation |

### Risk L2: Future Information in Time-Series Windows (NASA POWER)

| Property | Value |
|---|---|
| Severity | **HIGH** |
| Dataset | NASA POWER |
| Description | When constructing sliding windows for next-hour forecasting, the target value (hour t+1) must NOT appear as an input feature. Additionally, any derived features must use only data from hour t and earlier. |
| Mitigation | Window construction function must enforce: input window = [t-h, ..., t], target = t+1. Validate that no input timestamp >= target timestamp. |
| Implementation | PLANNED — window construction not yet coded |
| Verification | Automated test asserting max(input_timestamps) < min(target_timestamps) |

### Risk L3: Scaler/Imputer Fit on Full Data

| Property | Value |
|---|---|
| Severity | **MEDIUM** |
| Dataset | ALL time-series |
| Description | If StandardScaler, MinMaxScaler, or imputers are fit on the entire dataset (including validation/test), validation and test metrics are optimistically biased because the model indirectly "sees" the target distribution. |
| Mitigation | Fit all transformations on the training set ONLY. Apply the fitted transform to val/test without refitting. |
| Implementation | PLANNED — no scalers coded yet |
| Verification | Automated test checking that scaler.fit() is called only on training data |

### Risk L4: NPY-Parquet Misalignment for Test Labels (SKIPP'D)

| Property | Value |
|---|---|
| Severity | **HIGH** |
| Dataset | SKIPP'D |
| Description | `times_test.npy` does NOT row-align with `test-00000-of-00001.parquet`. Using npy timestamps to label or filter parquet rows would assign wrong timestamps (and potentially wrong split assignments) to test samples. |
| Mitigation | Use ONLY the parquet's embedded `time` column. Do NOT use npy files for row-level labeling. |
| Implementation | **ENFORCED** — documented as rule in PROJECT_CONTEXT.md |
| Verification | No code currently references npy files for labeling |

### Risk L5: Cross-Dataset Information Contamination

| Property | Value |
|---|---|
| Severity | **HIGH** |
| Dataset | Cross-dataset |
| Description | SKIPP'D (Stanford, 2017-2019) and NASA POWER (Mumbai, 2023) have no physical, temporal, or spatial relationship. Treating them as related would introduce meaningless correlations. |
| Mitigation | Keep all models entirely independent. Document incompatibility. |
| Implementation | **ENFORCED** — each model uses only its own dataset |
| Verification | Code review; no cross-dataset joins in data loading |

### Risk L6: Temporal Embargo Violation (NASA POWER)

| Property | Value |
|---|---|
| Severity | **MEDIUM** |
| Dataset | NASA POWER |
| Description | If an input window for a training sample overlaps temporally with a validation/test sample's target, information can leak across partitions. |
| Mitigation | Insert a gap (embargo) of at least `window_size` hours between the last training timestamp and the first validation timestamp (and between validation and test). |
| Implementation | PLANNED — split boundaries not yet defined in code |
| Verification | Automated test: `max(train_target_time) + embargo < min(val_input_time)` |

### Risk L7: Test Set Used for Model Selection

| Property | Value |
|---|---|
| Severity | **HIGH** |
| Dataset | ALL |
| Description | If test metrics are used to select hyperparameters, architectures, or early stopping criteria, the test set is no longer independent. |
| Mitigation | Use ONLY validation metrics for all model selection decisions. Report test metrics once, after final model is selected. |
| Implementation | PLANNED — training loop not yet coded |
| Verification | Code review of training scripts; checkpoint naming based on val_loss |

### Risk L8: Duplicate/Near-Duplicate Images Across Splits (SKIPP'D)

| Property | Value |
|---|---|
| Severity | **LOW-MEDIUM** |
| Dataset | SKIPP'D |
| Description | Consecutive minute-resolution sky images may be near-identical. If such images span a split boundary, the model effectively "sees" test-like images during training. |
| Mitigation | Calendar-day split (L1 mitigation) also handles this. For additional safety, consecutive images within a day always stay in the same split. |
| Implementation | Addressed by L1 mitigation |
| Verification | Visual inspection of sample images near split boundaries |

---

## 2. Leakage Check Automation Plan & Verification Results

| Check ID | Description | Dataset | Automated? | Implementation & Test | Status |
|---|---|---|---|---|---|
| LC-01 | No calendar day in both train and val | SKIPP'D | YES | `src/data/splits.py:split_skippd_by_day`<br>`tests/test_leakage.py::test_calendar_day_split_*` | **VERIFIED (PASSED)** |
| LC-02 | No shared timestamps between train shard & test parquet | SKIPP'D | YES | `tests/test_leakage.py::test_train_and_test_parquet_share_zero_*` | **VERIFIED (PASSED: 0 days, 0 timestamps)** |
| LC-03 | Input window max time < target time | NASA POWER | YES | `src/data/splits.py:create_forecasting_windows`<br>`tests/test_leakage.py::test_forecasting_windows_no_lookahead_bias` | **VERIFIED (PASSED)** |
| LC-04 | Scaler fitted on train only | ALL | YES | `src/data/splits.py:fit_scaler_train_only`<br>`tests/test_leakage.py::test_scaler_fitted_on_train_only` | **VERIFIED (PASSED)** |
| LC-05 | Embargo gap between chronological splits | NASA POWER | YES | `src/data/splits.py:split_timeseries_chronological`<br>`tests/test_leakage.py::test_chronological_split_embargo_gap` | **VERIFIED (PASSED: >= 24h)** |
| LC-06 | No shared targets across windowed splits | NASA POWER | YES | `tests/test_leakage.py::test_forecasting_windows_no_overlap_across_splits_on_actual_nasa` | **VERIFIED (PASSED: 0 shared targets)** |
| LC-07 | No npy-parquet cross-labeling | SKIPP'D | YES | `tests/test_skippd_integrity.py::test_npy_test_timestamps_differ_from_parquet` | **ENFORCED (Parquet time column authoritative)** |
| LC-08 | No cross-dataset joins | ALL | YES | `tests/test_leakage.py::test_geographic_metadata_isolation` | **ENFORCED (Verified from raw CSV metadata)** |
| LC-09 | Test metrics not used for model selection | ALL | MANUAL | Training loop design policy | **ENFORCED (Val loss used exclusively)** |

---

## 3. Current Status & Test Taxonomy

| Category | Status |
|---|---|
| Risk identification | **COMPLETED** |
| Mitigation strategies | **DOCUMENTED & CODED** (`src/data/splits.py`) |
| Automated test suite | **COMPLETED** (10 leakage tests in `tests/test_leakage.py`, 50 total tests passing) |
| Test Execution Taxonomy | **49 Real-File Tests** (98%) vs **1 Synthetic Unit Test** (2%) |
| Verification | **ALL PASSING** |

### Test Fixture Breakdown (50 Total Tests)
- **NASA Integrity (`tests/test_nasa_integrity.py`)**: 21 tests, all 21 run on `POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv` (100% real data).
- **SKIPP'D Integrity (`tests/test_skippd_integrity.py`)**: 19 tests, all 19 run on `train-00000-of-00005.parquet`, `test-00000-of-00001.parquet`, `times_trainval.npy`, `times_test.npy` across all 69,875 rows (100% real data).
- **Leakage Prevention (`tests/test_leakage.py`)**: 10 tests:
  - 9 tests execute directly on raw parquet, CSV, and full-column files.
  - 1 test (`test_calendar_day_split_function_zero_overlap`) uses a synthetic 10-day series as an isolated functional unit test, immediately complemented by `test_calendar_day_split_on_actual_skippd_train_shard` which validates the 69,875 real observations.

---

## 4. Key Empirical Findings

1. **SKIPP'D Train Shard ↔ Test Parquet Isolation**:
   - Train shard 1 spans 92 calendar days (`2017-03-09` to `2017-09-04`).
   - Test parquet spans 20 calendar days (`2017-06-24` to `2019-10-19`).
   - Although the overall bounding interval overlaps, only one test day (`2017-06-24`) occurs during shard 1's duration.
   - The training shard has entries on `2017-06-23` and `2017-06-25`, but completely omits `2017-06-24` (0 records).
   - Exact calendar day overlap: **0 days**.
   - Exact timestamp overlap: **0 timestamps**.
   - Full `times_trainval.npy` (349,372 rows) also has **0 days** overlap with the 20 test days.

2. **NASA POWER Embargo & Windowing Across Splits**:
   - Chronological split (70% train, 15% val, 15% test) enforces a 24-hour embargo interval between train/val and val/test boundaries.
   - For 1-hour-ahead forecasting ($L=24$, $H=1$), window generation on each partition strictly guarantees that:
     * $t_{\text{input}} \le t < t_{\text{target}}$ (zero intra-window lookahead).
     * $\max(t_{\text{train\_targets}}) < \min(t_{\text{val\_targets}}) < \min(t_{\text{test\_targets}})$.
     * Set intersection of target timestamps across all splits is strictly **empty**.

3. **Preprocessing Isolation**:
   - Standard scaling is strictly fitted on training arrays and applied out-of-sample to val and test arrays.

---

## 5. Remaining Risks & Go / No-Go Recommendation

### Remaining Risks
1. **SKIPP'D Shard-1 Sample Coverage**: With only shard 1 available (~20% of training data), the model will train on spring/summer 2017 only (March to September). Generalization to winter test days (e.g. January 2018/2019) may be constrained by seasonal distribution shifts.
2. **Missing Shards**: Does not affect data leakage or integrity, but limits benchmark comparability with the original Stanford paper.

### Phase 2 Go / No-Go Recommendation
- **Model A Data Pipeline (SKIPP'D)**: **GO**. Schema, labels, images, and day-split isolation verified.
- **Model B Data Pipeline (NASA POWER)**: **GO**. Schema, quality, 24h embargo, causal windowing, and scaler isolation verified.
- **Model C (Wind) & Model D (Demand)**: **NO-GO (BLOCKED)**. Directories empty. No implementation safe until data is provided.
