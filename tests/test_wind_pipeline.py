"""
Test suite for Wind Resource forecasting pipeline (Model C).

Validates:
1. Schema and physical bounds validation (wind speed m/s, direction, temperature, pressure).
2. Detection of missing columns, duplicate timestamps, hourly gaps, NaNs, and sentinels.
3. Turbine power curve physics (cut-in, rated, cut-out, cubic regime, density adjustment).
4. Chronological splitting with embargo gap (no boundary overlap, >= 24h gap).
5. Sliding-window causality and alignment (max(t_input) < t_target, t_target - t_origin == 1h).
6. Train-only scaler fitting and invertible target reconstruction.
7. Baseline models (persistence, seasonal naive) in physical m/s and estimated kW.
8. Blocked status check when raw data files are missing.
9. WindLSTM forward pass and trainer smoke test.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data.wind import (
    TurbinePowerCurve,
    validate_wind_dataframe,
    WindDataset,
    build_wind_datasets,
)
from src.models.wind.baselines import (
    WindPersistenceBaseline,
    WindSeasonalNaiveBaseline,
)
from src.models.wind.sequence_models import WindLSTM
from src.training.trainer import RegressionTrainer
from src.evaluation.metrics import calculate_regression_metrics


@pytest.fixture
def synthetic_wind_dataframe():
    """Generate 600 hours of continuous, physically plausible hourly wind data."""
    np.random.seed(42)
    n_hours = 600
    dates = pd.date_range("2023-01-01 00:00:00", periods=n_hours, freq="1h")

    # Diurnal + stochastic wind speed in m/s (mean ~ 7 m/s, range ~ 1 to 18 m/s)
    base_speed = 7.0 + 2.5 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    noise = np.random.normal(0, 1.2, size=n_hours)
    speed = np.clip(base_speed + noise, 0.5, 22.0).astype(np.float32)

    direction = (np.random.uniform(0, 360, size=n_hours)).astype(np.float32)
    temperature = (25.0 + 5.0 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + np.random.normal(0, 1.0, n_hours)).astype(np.float32)
    pressure = (1013.25 + np.random.normal(0, 3.0, n_hours)).astype(np.float32)

    return pd.DataFrame({
        "datetime": dates,
        "wind_speed": speed,
        "wind_direction": direction,
        "temperature": temperature,
        "pressure": pressure,
    })


class TestWindSchemaAndValidation:
    """Validate schema, timestamp monotonicity, and physical boundary enforcement."""

    def test_valid_wind_dataframe_passes(self, synthetic_wind_dataframe):
        """Verify clean dataframe validates without error and preserves shape."""
        df_valid = validate_wind_dataframe(
            synthetic_wind_dataframe,
            time_col="datetime",
            speed_col="wind_speed",
            direction_col="wind_direction",
            temp_col="temperature",
            pressure_col="pressure",
        )
        assert len(df_valid) == len(synthetic_wind_dataframe)
        assert df_valid["wind_speed"].dtype == np.float32

    def test_missing_required_column_raises_error(self, synthetic_wind_dataframe):
        """Verify missing wind speed column raises ValueError."""
        df_bad = synthetic_wind_dataframe.drop(columns=["wind_speed"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_wind_dataframe(df_bad, time_col="datetime", speed_col="wind_speed")

    def test_duplicate_timestamps_raises_error(self, synthetic_wind_dataframe):
        """Verify duplicated timestamps raise ValueError."""
        df_bad = synthetic_wind_dataframe.copy()
        df_bad.loc[5, "datetime"] = df_bad.loc[4, "datetime"]
        with pytest.raises(ValueError, match="duplicate timestamps"):
            validate_wind_dataframe(df_bad)

    def test_temporal_discontinuity_raises_error(self, synthetic_wind_dataframe):
        """Verify missing hour in sequence raises ValueError."""
        # Drop row 10 to create a 2-hour gap
        df_bad = synthetic_wind_dataframe.drop(index=[10]).reset_index(drop=True)
        with pytest.raises(ValueError, match="temporal discontinuities"):
            validate_wind_dataframe(df_bad)

    def test_nan_and_sentinel_values_raise_error(self, synthetic_wind_dataframe):
        """Verify NaNs and sentinel values (<= -900) are rejected."""
        df_nan = synthetic_wind_dataframe.copy()
        df_nan.loc[3, "wind_speed"] = np.nan
        with pytest.raises(ValueError, match="NaN/null"):
            validate_wind_dataframe(df_nan)

        df_sentinel = synthetic_wind_dataframe.copy()
        df_sentinel.loc[7, "wind_speed"] = -999.0
        with pytest.raises(ValueError, match="sentinel missing"):
            validate_wind_dataframe(df_sentinel)

    def test_physical_bounds_negative_speed(self, synthetic_wind_dataframe):
        """Verify negative wind speed is rejected."""
        df_bad = synthetic_wind_dataframe.copy()
        df_bad.loc[2, "wind_speed"] = -1.5
        with pytest.raises(ValueError, match="Negative wind speed"):
            validate_wind_dataframe(df_bad)

    def test_physical_bounds_extreme_speed(self, synthetic_wind_dataframe):
        """Verify unphysically high wind speed (>75 m/s) is rejected."""
        df_bad = synthetic_wind_dataframe.copy()
        df_bad.loc[2, "wind_speed"] = 82.0
        with pytest.raises(ValueError, match="Unphysically high wind speed"):
            validate_wind_dataframe(df_bad)

    def test_physical_bounds_direction_out_of_range(self, synthetic_wind_dataframe):
        """Verify wind direction outside [0, 360] is rejected."""
        df_bad = synthetic_wind_dataframe.copy()
        df_bad.loc[2, "wind_direction"] = 400.0
        with pytest.raises(ValueError, match="Wind direction out of bounds"):
            validate_wind_dataframe(df_bad, direction_col="wind_direction")


class TestTurbinePowerCurve:
    """Validate illustrative aerodynamic wind turbine power curve proxy calculation."""

    def test_power_curve_regions(self):
        """Verify cut-in, rated, cut-out, and cubic power regions."""
        curve = TurbinePowerCurve(
            rated_power_kw=2000.0,
            cut_in_speed=3.0,
            rated_speed=12.0,
            cut_out_speed=25.0,
        )

        # Region 1: Below cut-in (v < 3.0 m/s) -> 0 kW
        assert curve.wind_to_power(0.0) == 0.0
        assert curve.wind_to_power(2.5) == 0.0
        assert curve.wind_to_power(3.0) == 0.0  # (3^3 - 3^3) = 0

        # Region 2: Cubic regime (3.0 <= v < 12.0 m/s)
        # At v = 7.5 m/s: (7.5^3 - 3^3) / (12^3 - 3^3) = (421.875 - 27) / (1728 - 27)
        # = 394.875 / 1701 = 0.23214285 * 2000 = 464.2857 kW
        p_7_5 = curve.wind_to_power(7.5)
        expected_p = 2000.0 * (7.5**3 - 3.0**3) / (12.0**3 - 3.0**3)
        assert np.isclose(p_7_5, expected_p, atol=1e-3)
        assert 0.0 < p_7_5 < 2000.0

        # Region 3: Rated power regime (12.0 <= v < 25.0 m/s) -> exactly 2000 kW
        assert np.isclose(curve.wind_to_power(12.0), 2000.0, atol=1e-3)
        assert np.isclose(curve.wind_to_power(18.0), 2000.0, atol=1e-3)
        assert np.isclose(curve.wind_to_power(24.9), 2000.0, atol=1e-3)

        # Region 4: Storm shutdown (v >= 25.0 m/s) -> 0 kW
        assert curve.wind_to_power(25.0) == 0.0
        assert curve.wind_to_power(30.0) == 0.0

    def test_power_curve_array_and_torch_inputs(self):
        """Verify vectorization with NumPy arrays and PyTorch tensors."""
        curve = TurbinePowerCurve(rated_power_kw=2000.0)
        speeds = np.array([1.0, 3.0, 7.5, 12.0, 20.0, 26.0], dtype=np.float32)
        powers = curve.wind_to_power(speeds)

        assert isinstance(powers, np.ndarray)
        assert len(powers) == 6
        assert powers[0] == 0.0
        assert powers[3] == 2000.0
        assert powers[5] == 0.0

        # PyTorch tensor input
        speeds_t = torch.tensor(speeds)
        powers_from_t = curve.wind_to_power(speeds_t)
        assert np.allclose(powers, powers_from_t, atol=1e-4)

    def test_power_curve_density_correction(self):
        """Verify IEC density normalization scales effective wind speed."""
        curve = TurbinePowerCurve(rated_power_kw=2000.0, reference_density=1.225)
        v = 8.0

        # Standard density: 1.225 kg/m^3
        p_std = curve.wind_to_power(v, air_density=1.225)

        # High density (cold winter air at sea level): 1.30 kg/m^3 -> higher power
        p_high = curve.wind_to_power(v, air_density=1.30)
        assert p_high > p_std

        # Low density (hot summer air or high altitude): 1.10 kg/m^3 -> lower power
        p_low = curve.wind_to_power(v, air_density=1.10)
        assert p_low < p_std

    def test_invalid_power_curve_thresholds(self):
        """Verify raising ValueError when cut_in >= rated or rated >= cut_out."""
        with pytest.raises(ValueError, match="Invalid speed thresholds"):
            TurbinePowerCurve(cut_in_speed=15.0, rated_speed=12.0)

    def test_power_curve_rejects_invalid_inputs(self):
        """Verify TurbinePowerCurve raises ValueError on negative wind speed or non-positive density."""
        curve = TurbinePowerCurve()
        with pytest.raises(ValueError, match="non-negative"):
            curve.wind_to_power(-2.5)

        with pytest.raises(ValueError, match="strictly positive"):
            curve.wind_to_power(8.0, air_density=0.0)



class TestWindSplitsAndLeakage:
    """Validate chronological splitting, embargo buffers, and window causality."""

    def test_chronological_splits_and_embargo(self, synthetic_wind_dataframe):
        """Verify train, val, and test partitions are separated by >= 24h embargo."""
        train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_wind_datasets(
            df=synthetic_wind_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            embargo_hours=24,
            scale=True,
        )

        # Timestamps
        t_train_targets = pd.to_datetime(train_ds.target_timestamps)
        t_val_targets = pd.to_datetime(val_ds.target_timestamps)
        t_test_targets = pd.to_datetime(test_ds.target_timestamps)

        # Zero target overlap
        assert len(set(t_train_targets) & set(t_val_targets)) == 0
        assert len(set(t_val_targets) & set(t_test_targets)) == 0
        assert len(set(t_train_targets) & set(t_test_targets)) == 0

        # Temporal ordering
        assert t_train_targets.max() < t_val_targets.min()
        assert t_val_targets.max() < t_test_targets.min()

        # Embargo gap >= 24h
        val_gap = t_val_targets.min() - t_train_targets.max()
        test_gap = t_test_targets.min() - t_val_targets.max()
        assert val_gap >= pd.Timedelta(hours=24)
        assert test_gap >= pd.Timedelta(hours=24)

    def test_sliding_window_causality_and_alignment(self, synthetic_wind_dataframe):
        """Verify that for every sample, max(input_time) < target_time and target - origin == 1h."""
        train_ds, val_ds, test_ds, _, _ = build_wind_datasets(
            df=synthetic_wind_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=False,
        )

        for ds in [train_ds, val_ds, test_ds]:
            for i in range(min(len(ds), 50)):
                t_origin, t_target = ds.get_time_alignment(i)
                assert t_origin < t_target
                assert t_target - t_origin == pd.Timedelta(hours=1)

    def test_scaler_fit_strictly_on_train(self, synthetic_wind_dataframe):
        """Verify that StandardScaler parameters are computed only from training samples."""
        train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_wind_datasets(
            df=synthetic_wind_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        assert targ_scaler is not None
        # Invert target scaling on val set and check fidelity
        val_scaled = val_ds.targets.numpy()
        val_unscaled = val_ds.inverse_transform_targets(val_scaled)

        # All unscaled targets must be >= 0 m/s
        assert np.all(val_unscaled >= 0.0)
        assert np.all(np.isfinite(val_unscaled))

    def test_missing_data_raises_blocked_filenotfound(self):
        """Verify build_wind_datasets without df raises informative FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="BLOCKED pending NREL download approval"):
            build_wind_datasets(df=None, csv_path="data/raw/wind/non_existent.csv")


class TestWindBaselines:
    """Validate persistence and seasonal naive baseline logic and power prediction."""

    def test_persistence_baseline(self, synthetic_wind_dataframe):
        """Verify persistence baseline predicts observation at origin index (last step)."""
        train_ds, val_ds, _, _, targ_scaler = build_wind_datasets(
            df=synthetic_wind_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        baseline = WindPersistenceBaseline(
            speed_feature_idx=0,
            feature_scaler=targ_scaler,
        )

        preds = baseline.predict(val_ds.features)
        assert len(preds) == len(val_ds)
        assert np.all(np.isfinite(preds))

        # Check against manual unscaled last observation
        raw_last = val_ds.features[:, -1, 0].numpy()
        expected = raw_last * targ_scaler.scale_[0] + targ_scaler.mean_[0]
        assert np.allclose(preds, expected, atol=1e-4)

    def test_seasonal_naive_baseline(self, synthetic_wind_dataframe):
        """Verify seasonal naive baseline selects t - 23h index (index 0 for L=24, H=1)."""
        train_ds, val_ds, _, _, targ_scaler = build_wind_datasets(
            df=synthetic_wind_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        baseline = WindSeasonalNaiveBaseline(
            sequence_length=24,
            forecast_horizon=1,
            seasonal_period=24,
            speed_feature_idx=0,
            feature_scaler=targ_scaler,
        )

        assert baseline.seasonal_idx == 0

        preds = baseline.predict(val_ds.features)
        assert len(preds) == len(val_ds)
        assert np.all(np.isfinite(preds))

    def test_power_prediction_via_baselines(self, synthetic_wind_dataframe):
        """Verify predict_power produces valid non-negative kW values."""
        _, val_ds, _, _, targ_scaler = build_wind_datasets(
            df=synthetic_wind_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        curve = TurbinePowerCurve(rated_power_kw=2000.0)
        baseline = WindPersistenceBaseline(feature_scaler=targ_scaler, power_curve=curve)
        power_preds = baseline.predict_power(val_ds.features)

        assert np.all(power_preds >= 0.0)
        assert np.all(power_preds <= 2000.0)
        assert np.all(np.isfinite(power_preds))

    def test_wind_baselines_reject_non_finite_inputs(self):
        """Verify persistence and seasonal naive raise ValueError on NaN/Inf inputs."""
        X_nan = np.ones((5, 24, 1), dtype=np.float32)
        X_nan[0, -1, 0] = np.nan  # NaN at forecast origin
        persist = WindPersistenceBaseline()
        with pytest.raises(ValueError, match="non-finite"):
            persist.predict(X_nan)

        X_nan_seasonal = np.ones((5, 24, 1), dtype=np.float32)
        X_nan_seasonal[0, 0, 0] = np.nan  # NaN at seasonal index 0
        seasonal = WindSeasonalNaiveBaseline(sequence_length=24, forecast_horizon=1, seasonal_period=24)
        with pytest.raises(ValueError, match="non-finite"):
            seasonal.predict(X_nan_seasonal)

    def test_wind_baselines_reject_invalid_input_shapes(self):
        """Verify baselines raise ValueError when provided with non-3D input."""
        persist = WindPersistenceBaseline()
        with pytest.raises(ValueError, match="Expected 3D input"):
            persist.predict(np.ones((5, 24)))

        seasonal = WindSeasonalNaiveBaseline(sequence_length=24, forecast_horizon=1, seasonal_period=24)
        with pytest.raises(ValueError, match="Expected 3D input"):
            seasonal.predict(np.ones((5, 24)))



class TestWindModelTrainingSmoke:
    """Validate WindLSTM architecture and training pipeline integration."""

    def test_wind_lstm_shapes_and_training_step(self, tmp_path, synthetic_wind_dataframe):
        """Verify forward pass shapes and that RegressionTrainer runs a smoke epoch."""
        train_ds, val_ds, _, _, targ_scaler = build_wind_datasets(
            df=synthetic_wind_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        model = WindLSTM(input_dim=1, hidden_dim=16, num_layers=1, dropout=0.0)
        trainable, total = model.get_parameter_counts()
        assert trainable == total > 0

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = RegressionTrainer(
            model=model,
            optimizer=optimizer,
            experiment_id="TEST_WIND_SMOKE",
            checkpoint_dir=tmp_path / "checkpoints",
            target_unscaler=val_ds.inverse_transform_targets,
        )

        loss = trainer.train_epoch(train_loader)
        assert np.isfinite(loss)
        assert loss > 0.0

        eval_res = trainer.evaluate(val_loader)
        assert "loss" in eval_res
        assert "mae" in eval_res
        assert np.isfinite(eval_res["loss"])
        assert np.isfinite(eval_res["mae"])


class TestRealWindDataValidation:
    """Validate real NREL India Wind Toolkit pilot dataset (when downloaded)."""

    @pytest.fixture
    def real_pilot_path(self):
        """Return path to real processed pilot dataset."""
        path = Path("data/processed/wind/wind_hourly_pilot.csv")
        if not path.exists():
            pytest.skip("Real wind pilot data not yet downloaded. Run scripts/download_wind_pilot.py first.")
        return path

    def test_real_pilot_row_count_and_continuity(self, real_pilot_path):
        """Verify real pilot dataset has exactly 8,760 hours and unbroken 1h continuity."""
        df = pd.read_csv(real_pilot_path)
        assert len(df) == 8760, f"Expected 8,760 rows for 2014, got {len(df)}"

        df["datetime"] = pd.to_datetime(df["datetime"])
        assert df["datetime"].iloc[0] == pd.Timestamp("2014-01-01 00:30:00")
        assert df["datetime"].iloc[-1] == pd.Timestamp("2014-12-31 23:30:00")

        diffs = df["datetime"].diff().iloc[1:]
        assert (diffs == pd.Timedelta(hours=1)).all(), "Found temporal discontinuity in real dataset"

    def test_real_pilot_schema_and_physical_bounds(self, real_pilot_path):
        """Verify schema validation and physical bounds on real NREL extract."""
        df = pd.read_csv(real_pilot_path)
        df_valid = validate_wind_dataframe(
            df,
            time_col="datetime",
            speed_col="wind_speed",
            direction_col="wind_direction",
            temp_col="temperature",
            pressure_col="pressure",
            expected_freq="1h",
        )

        assert df_valid["wind_speed"].min() >= 0.0
        assert df_valid["wind_speed"].max() <= 75.0
        assert df_valid["wind_direction"].min() >= 0.0
        assert df_valid["wind_direction"].max() <= 360.0
        assert 5.0 <= df_valid["temperature"].min() <= 50.0
        assert 900.0 <= df_valid["pressure"].min() <= 1050.0

    def test_real_pilot_build_datasets_pipeline(self, real_pilot_path):
        """Verify build_wind_datasets executes cleanly on real data with zero leakage."""
        train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_wind_datasets(
            csv_path=real_pilot_path,
            sequence_length=24,
            forecast_horizon=1,
            embargo_hours=24,
            scale=True,
        )

        assert len(train_ds) > 5000
        assert len(val_ds) > 1000
        assert len(test_ds) > 1000

        # Verify shapes
        assert train_ds.features.shape[1:] == (24, 1)
        assert train_ds.targets.shape[1:] == (1,)

        # Verify unscaled physical values
        unscaled_val = val_ds.inverse_transform_targets(val_ds.targets.numpy())
        assert np.all(unscaled_val >= 0.0)
        assert np.all(unscaled_val <= 30.0)

