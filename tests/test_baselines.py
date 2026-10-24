"""
Test suite for baseline models and evaluation metrics.

Validates:
1. Global mean and median baselines use training data only.
2. Hour-of-day climatology uses sample timestamp's local hour.
3. Missing climatology hours fall back to global training mean.
4. Conditional persistence never pairs samples separated by > 2 minutes.
5. NASA one-hour persistence uses final input irradiance value.
6. NASA seasonal naive selects correct t - 23h index for target at t + 1h.
7. Baseline predictions have expected lengths and finite values.
8. Evaluation metrics match manually calculated values on synthetic data.
9. Metrics reject NaN and infinite inputs.
10. Integration on real SKIPP'D and NASA validation partitions.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    calculate_regression_metrics,
)
from src.evaluation.registry import ExperimentRegistry, REGISTRY_COLUMNS
from src.models.solar.baselines import (
    GlobalMeanBaseline,
    GlobalMedianBaseline,
    HourOfDayClimatologyBaseline,
    ConditionalPersistenceBaseline,
)
from src.models.nasa_weather.baselines import (
    NasaOneHourPersistenceBaseline,
    NasaSeasonalNaiveBaseline,
)
from src.data.datasets import build_skippd_datasets, build_nasa_datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PARQUET = PROJECT_ROOT / "data" / "raw" / "skippd" / "train-00000-of-00005.parquet"
NASA_CSV = PROJECT_ROOT / "data" / "raw" / "nasa_power" / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"


# =====================================================================
# Synthetic Unit Tests
# =====================================================================

class TestMetricCalculations:
    """Validate metric mathematical definitions and edge case handling."""

    def test_evaluation_metrics_synthetic_manual_check(self):
        """Verify MAE, RMSE, and R2 match exact analytical calculation."""
        y_true = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        y_pred = np.array([1.0, 5.0, 9.0], dtype=np.float64)

        # MAE: (|2-1| + |4-5| + |6-9|) / 3 = 5/3 = 1.666667
        # RMSE: sqrt((1 + 1 + 9) / 3) = sqrt(11/3) = 1.914854
        # SS_tot: (2-4)^2 + (4-4)^2 + (6-4)^2 = 8.0
        # SS_res: 11.0
        # R2: 1 - (11/8) = -0.375
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        assert np.isclose(mae, 5.0 / 3.0, atol=1e-5)
        assert np.isclose(rmse, np.sqrt(11.0 / 3.0), atol=1e-5)
        assert np.isclose(r2, -0.375, atol=1e-5)

    def test_metrics_reject_nan_and_inf(self):
        """Verify metric functions raise assertion when NaN or Inf are passed."""
        y_good = np.array([1.0, 2.0, 3.0])
        y_nan = np.array([1.0, np.nan, 3.0])
        y_inf = np.array([1.0, np.inf, 3.0])

        with pytest.raises(AssertionError, match="non-finite"):
            mean_absolute_error(y_nan, y_good)

        with pytest.raises(AssertionError, match="non-finite"):
            root_mean_squared_error(y_good, y_inf)

        with pytest.raises(AssertionError, match="Shape mismatch"):
            mean_absolute_error(y_good, np.array([1.0, 2.0]))

    def test_r2_score_constant_target_returns_nan(self):
        """Verify R2 returns NaN when ground truth has zero variance (SS_tot == 0)."""
        y_const = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.1, 4.9, 5.0])
        score = r2_score(y_const, y_pred)
        assert np.isnan(score)

    def test_calculate_regression_metrics_prefix(self):
        """Verify prefix parameter correctly formats dictionary keys."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        metrics = calculate_regression_metrics(y_true, y_pred, prefix="val_")
        assert "val_mae" in metrics
        assert "val_rmse" in metrics
        assert "val_r2" in metrics
        assert "val_n_samples" in metrics
        assert metrics["val_n_samples"] == 3


class TestSolarBaselinesSynthetic:
    """Unit tests for SKIPP'D baseline models on synthetic fixtures."""

    def test_global_mean_and_median_train_only(self):
        """Verify mean and median baselines derive statistics strictly from training data."""
        y_train = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        y_val = np.array([500.0, 600.0, 700.0], dtype=np.float32)

        mean_model = GlobalMeanBaseline().fit(y_train)
        median_model = GlobalMedianBaseline().fit(y_train)

        assert mean_model.mean_ == 20.0
        assert median_model.median_ == 20.0

        preds_mean = mean_model.predict(len(y_val))
        preds_median = median_model.predict(len(y_val))

        np.testing.assert_array_equal(preds_mean, [20.0, 20.0, 20.0])
        np.testing.assert_array_equal(preds_median, [20.0, 20.0, 20.0])

    def test_hour_of_day_climatology_uses_local_hour(self):
        """Verify climatology model maps timestamps to their corresponding local hour."""
        timestamps = pd.to_datetime([
            "2023-05-01 10:00:00",
            "2023-05-01 10:30:00",
            "2023-05-01 14:00:00",
            "2023-05-02 10:00:00",
        ])
        y_train = np.array([10.0, 14.0, 25.0, 12.0], dtype=np.float32)

        model = HourOfDayClimatologyBaseline().fit(timestamps, y_train)

        # Hour 10 mean: (10 + 14 + 12) / 3 = 12.0
        # Hour 14 mean: 25.0
        assert np.isclose(model.hourly_means_[10], 12.0)
        assert np.isclose(model.hourly_means_[14], 25.0)

        val_ts = pd.to_datetime(["2023-06-01 10:15:00", "2023-06-01 14:45:00"])
        preds = model.predict(val_ts)
        np.testing.assert_allclose(preds, [12.0, 25.0], atol=1e-5)

    def test_climatology_missing_hour_fallback(self):
        """Verify missing hour falls back to global training mean."""
        timestamps = pd.to_datetime(["2023-05-01 12:00:00", "2023-05-01 12:30:00"])
        y_train = np.array([20.0, 22.0], dtype=np.float32)

        model = HourOfDayClimatologyBaseline().fit(timestamps, y_train)
        assert model.global_mean_ == 21.0

        # Predict for hour 8 (not in training data)
        val_ts = pd.to_datetime(["2023-05-02 08:00:00"])
        preds = model.predict(val_ts)
        assert preds[0] == 21.0

    def test_conditional_persistence_max_gap_enforcement(self):
        """Verify conditional persistence strictly enforces delta_t <= 120s."""
        timestamps = pd.to_datetime([
            "2023-05-01 10:00:00",
            "2023-05-01 10:01:00",  # delta_t = 60s  (VALID)
            "2023-05-01 10:02:30",  # delta_t = 90s  (VALID)
            "2023-05-01 10:10:00",  # delta_t = 450s (INVALID, gap > 120s)
            "2023-05-01 10:11:00",  # delta_t = 60s  (VALID)
        ])
        pv_values = np.array([5.0, 6.0, 7.0, 20.0, 21.0], dtype=np.float32)

        model = ConditionalPersistenceBaseline(max_gap_seconds=120.0)
        valid_idx, y_true, y_pred = model.evaluate_valid_pairs(timestamps, pv_values)

        # Expected valid pairs: indices 1, 2, and 4
        # Index 0 has no predecessor; Index 3 has 450s gap > 120s
        np.testing.assert_array_equal(valid_idx, [1, 2, 4])
        np.testing.assert_array_equal(y_true, [6.0, 7.0, 21.0])
        np.testing.assert_array_equal(y_pred, [5.0, 6.0, 20.0])

        # Test fallback behavior
        fallback = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        full_preds = model.predict_with_fallback(timestamps, pv_values, fallback)
        np.testing.assert_array_equal(full_preds, [100.0, 5.0, 6.0, 100.0, 20.0])

    def test_conditional_persistence_rejects_across_day_pairs(self):
        """Verify conditional persistence never pairs across calendar days, even if gap <= 120s."""
        timestamps = pd.to_datetime([
            "2023-05-01 23:59:30",
            "2023-05-02 00:00:30",  # delta_t = 60s, but DIFFERENT calendar day
        ])
        pv_values = np.array([15.0, 14.0], dtype=np.float32)

        model = ConditionalPersistenceBaseline(max_gap_seconds=120.0)
        valid_idx, y_true, y_pred = model.evaluate_valid_pairs(timestamps, pv_values)

        # Must be empty: no pairing across overnight/calendar day boundary
        assert len(valid_idx) == 0
        assert len(y_true) == 0
        assert len(y_pred) == 0


class TestNasaBaselinesSynthetic:
    """Unit tests for NASA POWER baseline models on synthetic fixtures."""

    def test_nasa_one_hour_persistence_uses_final_input_value(self):
        """Verify 1h persistence takes the observation at sequence_length - 1 (forecast origin)."""
        # Shape: (3 samples, 24 timesteps, 4 features)
        X = np.zeros((3, 24, 4), dtype=np.float32)
        X[0, -1, 0] = 123.0
        X[1, -1, 0] = 456.0
        X[2, -1, 0] = 789.0

        model = NasaOneHourPersistenceBaseline(irradiance_feature_idx=0)
        preds = model.predict(X)

        np.testing.assert_array_equal(preds, [123.0, 456.0, 789.0])

    def test_nasa_seasonal_naive_uses_t_minus_23h(self):
        """
        Verify seasonal naive selects index 0 for sequence_length=24 and horizon=1.
        (Target is at t + 1; 24h prior is t - 23, which is index 0 of window [t-23, ..., t]).
        """
        model = NasaSeasonalNaiveBaseline(
            sequence_length=24,
            forecast_horizon=1,
            seasonal_period=24,
            irradiance_feature_idx=0,
        )
        assert model.seasonal_idx == 0

        X = np.zeros((2, 24, 4), dtype=np.float32)
        X[0, 0, 0] = 314.0
        X[1, 0, 0] = 271.0

        preds = model.predict(X)
        np.testing.assert_array_equal(preds, [314.0, 271.0])

    def test_nasa_baselines_with_feature_scaler(self):
        """Verify NASA baselines correctly invert standardization to output physical units."""
        raw_values = np.array([[100.0, 10.0, 50.0, 2.0], [300.0, 20.0, 60.0, 4.0]], dtype=np.float32)
        scaler = StandardScaler().fit(raw_values)

        # Create standardized input tensor
        scaled_values = scaler.transform(raw_values)
        X = np.zeros((2, 24, 4), dtype=np.float32)
        X[0, -1, :] = scaled_values[0]
        X[1, -1, :] = scaled_values[1]
        X[0, 0, :] = scaled_values[1]
        X[1, 0, :] = scaled_values[0]

        # 1. 1h persistence with scaler
        persist_model = NasaOneHourPersistenceBaseline(irradiance_feature_idx=0, feature_scaler=scaler)
        preds_1h = persist_model.predict(X)
        np.testing.assert_allclose(preds_1h, [100.0, 300.0], atol=1e-4)

        # 2. Seasonal naive with scaler
        seasonal_model = NasaSeasonalNaiveBaseline(
            sequence_length=24,
            forecast_horizon=1,
            seasonal_period=24,
            irradiance_feature_idx=0,
            feature_scaler=scaler,
        )
        preds_seas = seasonal_model.predict(X)
        np.testing.assert_allclose(preds_seas, [300.0, 100.0], atol=1e-4)


class TestExperimentRegistry:
    """Unit tests for experiment tracking registry."""

    def test_experiment_registry_create_log_and_idempotent(self, tmp_path):
        """Verify registry file creation, 19-column schema validation, and idempotent overwrites."""
        reg_file = tmp_path / "test_registry.csv"
        registry = ExperimentRegistry(reg_file)

        assert reg_file.exists()
        df_init = registry.get_records()
        assert list(df_init.columns) == REGISTRY_COLUMNS
        assert len(df_init) == 0

        # Log first experiment
        rec1 = registry.log_experiment(
            experiment_id="TEST_EXP_01",
            dataset="TEST_DATA",
            task="testing",
            model_name="TestModel",
            partition="val",
            features="test_feat",
            target="test_targ",
            n_samples=100,
            mae=1.2345,
            rmse=2.3456,
            r2=0.8765,
            metric_unit="units",
        )
        df_after1 = registry.get_records()
        assert len(df_after1) == 1
        assert df_after1.iloc[0]["experiment_id"] == "TEST_EXP_01"
        assert df_after1.iloc[0]["mae"] == 1.2345

        # Re-log same experiment_id with updated metric (idempotent update)
        rec2 = registry.log_experiment(
            experiment_id="TEST_EXP_01",
            dataset="TEST_DATA",
            task="testing",
            model_name="TestModel",
            partition="val",
            features="test_feat",
            target="test_targ",
            n_samples=100,
            mae=0.9999,
            rmse=1.9999,
            r2=0.9500,
            metric_unit="units",
        )
        df_after2 = registry.get_records()
        assert len(df_after2) == 1, "Idempotent re-run should overwrite existing experiment_id row, not duplicate"
        assert df_after2.iloc[0]["mae"] == 0.9999


# =====================================================================
# Real-File Integration Tests
# =====================================================================

class TestBaselinesRealDataIntegration:
    """Integration test verifying baseline execution and registry logging on actual files."""

    @pytest.mark.skipif(not TRAIN_PARQUET.exists(), reason="SKIPP'D parquet missing")
    def test_skippd_baselines_validation_run(self):
        """Verify SKIPP'D baselines produce valid, finite predictions on actual validation set."""
        train_ds, val_ds, _ = build_skippd_datasets(
            train_parquet_path=TRAIN_PARQUET,
            train_ratio=0.8,
            chronological=True,
        )

        y_train = train_ds.get_pv_values()
        t_train = train_ds.get_timestamps()
        y_val = val_ds.get_pv_values()
        t_val = val_ds.get_timestamps()

        # 1. Climatology
        clim = HourOfDayClimatologyBaseline().fit(t_train, y_train)
        preds_clim = clim.predict(t_val)
        assert len(preds_clim) == len(y_val)
        assert np.isfinite(preds_clim).all()
        metrics = calculate_regression_metrics(y_val, preds_clim)
        assert metrics["mae"] < 5.0  # Climatology MAE should be ~1.88 kW
        assert metrics["r2"] > 0.8   # R2 should be ~0.88

        # 2. Conditional Persistence
        persist = ConditionalPersistenceBaseline(max_gap_seconds=120.0)
        valid_idx, y_sub, p_sub = persist.evaluate_valid_pairs(t_val, y_val)
        assert len(valid_idx) > 12000
        p_metrics = calculate_regression_metrics(y_sub, p_sub)
        assert p_metrics["mae"] < 0.5  # Persistence MAE should be ~0.14 kW
        assert p_metrics["r2"] > 0.95

    @pytest.mark.skipif(not NASA_CSV.exists(), reason="NASA CSV missing")
    def test_nasa_baselines_validation_run(self):
        """Verify NASA POWER baselines produce valid predictions on actual validation set."""
        _, val_ds, _, _, _ = build_nasa_datasets(
            csv_path=NASA_CSV,
            sequence_length=24,
            forecast_horizon=1,
            scale=False,
        )

        X_val = val_ds.features.numpy()
        y_val = val_ds.targets.numpy().ravel()

        # 1. 1h Persistence
        persist_1h = NasaOneHourPersistenceBaseline(irradiance_feature_idx=0)
        p_1h = persist_1h.predict(X_val)
        assert len(p_1h) == len(y_val)
        m_1h = calculate_regression_metrics(y_val, p_1h)
        assert m_1h["mae"] < 70.0  # ~61 Wh/m²
        assert m_1h["r2"] > 0.85

        # 2. Seasonal Naive
        seasonal = NasaSeasonalNaiveBaseline(
            sequence_length=24,
            forecast_horizon=1,
            seasonal_period=24,
            irradiance_feature_idx=0,
        )
        p_seas = seasonal.predict(X_val)
        assert len(p_seas) == len(y_val)
        m_seas = calculate_regression_metrics(y_val, p_seas)
        assert m_seas["mae"] < 30.0  # ~23 Wh/m²
        assert m_seas["r2"] > 0.95
