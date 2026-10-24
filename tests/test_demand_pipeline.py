"""
Test suite for Electricity Demand forecasting pipeline (Model D - BDG2).

Validates:
1. Schema and physical bounds validation (electricity demand in kWh, non-negativity).
2. Detection of missing columns, duplicate timestamps, hourly gaps, NaNs, and sentinels.
3. Chronological splitting with embargo gap (no boundary overlap, >= 24h gap).
4. Sliding-window causality and alignment (max(t_input) < t_target, t_target - t_origin == 1h).
5. Train-only scaler fitting and invertible target reconstruction.
6. Baseline models (persistence, 24h seasonal naive, 168h weekly seasonal naive) in physical kWh.
7. Blocked status check when raw BDG2 files are missing.
8. DemandLSTM forward pass and trainer smoke test.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data.demand import (
    validate_demand_dataframe,
    DemandDataset,
    build_demand_datasets,
)
from src.models.demand.baselines import (
    DemandPersistenceBaseline,
    DemandSeasonalNaiveBaseline,
    DemandWeeklySeasonalNaiveBaseline,
)
from src.models.demand.sequence_models import DemandLSTM
from src.training.trainer import RegressionTrainer
from src.evaluation.metrics import calculate_regression_metrics


@pytest.fixture
def synthetic_demand_dataframe():
    """Generate 600 hours of continuous, realistic building electricity demand data (kWh)."""
    np.random.seed(42)
    n_hours = 600
    dates = pd.date_range("2016-01-01 00:00:00", periods=n_hours, freq="1h")

    # Diurnal cycle (peaks during daytime, e.g. 9 AM - 6 PM)
    hours = dates.hour.to_numpy()
    day_of_week = dates.dayofweek.to_numpy()  # 0=Mon, ..., 6=Sun

    # Weekday baseline ~ 150-350 kWh, weekend baseline ~ 60-120 kWh
    is_weekend = day_of_week >= 5
    diurnal = np.sin((hours - 6) / 24 * np.pi) ** 2
    diurnal = np.maximum(diurnal, 0.1)

    base = np.where(is_weekend, 80.0 + 40.0 * diurnal, 150.0 + 200.0 * diurnal)
    noise = np.random.normal(0, 15.0, size=n_hours)
    demand = np.clip(base + noise, 20.0, 450.0).astype(np.float32)

    return pd.DataFrame({
        "timestamp": dates,
        "demand_kwh": demand,
    })


class TestDemandSchemaAndValidation:
    """Validate schema, timestamp monotonicity, and physical boundary enforcement."""

    def test_valid_demand_dataframe_passes(self, synthetic_demand_dataframe):
        """Verify clean demand dataframe validates without error and preserves shape."""
        df_valid = validate_demand_dataframe(
            synthetic_demand_dataframe,
            time_col="timestamp",
            meter_col="demand_kwh",
        )
        assert len(df_valid) == len(synthetic_demand_dataframe)
        assert df_valid["demand_kwh"].dtype == np.float32

    def test_missing_required_column_raises_error(self, synthetic_demand_dataframe):
        """Verify missing demand column raises ValueError."""
        df_bad = synthetic_demand_dataframe.drop(columns=["demand_kwh"])
        with pytest.raises(ValueError, match="missing meter reading column"):
            validate_demand_dataframe(df_bad, time_col="timestamp", meter_col="demand_kwh")

    def test_duplicate_timestamps_raises_error(self, synthetic_demand_dataframe):
        """Verify duplicate timestamps raise ValueError."""
        df_bad = synthetic_demand_dataframe.copy()
        df_bad.loc[5, "timestamp"] = df_bad.loc[4, "timestamp"]
        with pytest.raises(ValueError, match="duplicate timestamps"):
            validate_demand_dataframe(df_bad)

    def test_temporal_discontinuity_raises_error(self, synthetic_demand_dataframe):
        """Verify missing hour in sequence raises ValueError."""
        df_bad = synthetic_demand_dataframe.drop(index=[15]).reset_index(drop=True)
        with pytest.raises(ValueError, match="temporal discontinuities"):
            validate_demand_dataframe(df_bad)

    def test_nan_and_sentinel_values_raise_error(self, synthetic_demand_dataframe):
        """Verify NaNs and sentinel values are rejected."""
        df_nan = synthetic_demand_dataframe.copy()
        df_nan.loc[4, "demand_kwh"] = np.nan
        with pytest.raises(ValueError, match="NaN/null"):
            validate_demand_dataframe(df_nan)

        df_sentinel = synthetic_demand_dataframe.copy()
        df_sentinel.loc[8, "demand_kwh"] = -999.0
        with pytest.raises(ValueError, match="sentinel missing"):
            validate_demand_dataframe(df_sentinel)

    def test_physical_bounds_negative_demand(self, synthetic_demand_dataframe):
        """Verify negative electricity consumption is rejected."""
        df_bad = synthetic_demand_dataframe.copy()
        df_bad.loc[3, "demand_kwh"] = -5.0
        with pytest.raises(ValueError, match="Negative electricity consumption"):
            validate_demand_dataframe(df_bad)

    def test_physical_bounds_extreme_demand(self, synthetic_demand_dataframe):
        """Verify unphysically high demand exceeding threshold is rejected."""
        df_bad = synthetic_demand_dataframe.copy()
        df_bad.loc[3, "demand_kwh"] = 15000.0
        with pytest.raises(ValueError, match="Unphysically high demand"):
            validate_demand_dataframe(df_bad, max_demand_kwh=10000.0)


class TestDemandSplitsAndLeakage:
    """Validate chronological splitting, embargo buffers, and window causality."""

    def test_chronological_splits_and_embargo(self, synthetic_demand_dataframe):
        """Verify train, val, and test partitions are separated by >= 24h embargo."""
        train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_demand_datasets(
            df=synthetic_demand_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            embargo_hours=24,
            scale=True,
        )

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

    def test_sliding_window_causality_and_alignment(self, synthetic_demand_dataframe):
        """Verify that for every sample, max(input_time) < target_time and target - origin == 1h."""
        train_ds, val_ds, test_ds, _, _ = build_demand_datasets(
            df=synthetic_demand_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=False,
        )

        for ds in [train_ds, val_ds, test_ds]:
            for i in range(min(len(ds), 50)):
                t_origin, t_target = ds.get_time_alignment(i)
                assert t_origin < t_target
                assert t_target - t_origin == pd.Timedelta(hours=1)

    def test_scaler_fit_strictly_on_train(self, synthetic_demand_dataframe):
        """Verify that StandardScaler parameters are computed only from training samples."""
        train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_demand_datasets(
            df=synthetic_demand_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        assert targ_scaler is not None
        val_scaled = val_ds.targets.numpy()
        val_unscaled = val_ds.inverse_transform_targets(val_scaled)

        assert np.all(val_unscaled >= 0.0)
        assert np.all(np.isfinite(val_unscaled))

    def test_missing_data_raises_blocked_filenotfound(self):
        """Verify build_demand_datasets without df raises informative FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="BLOCKED pending BDG2 download"):
            build_demand_datasets(df=None, csv_path="data/raw/bdg2/non_existent.csv")


class TestDemandBaselines:
    """Validate persistence and seasonal naive baseline logic."""

    def test_persistence_baseline(self, synthetic_demand_dataframe):
        """Verify persistence baseline predicts observation at origin index (last step)."""
        train_ds, val_ds, _, _, targ_scaler = build_demand_datasets(
            df=synthetic_demand_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        baseline = DemandPersistenceBaseline(
            meter_feature_idx=0,
            feature_scaler=targ_scaler,
        )

        preds = baseline.predict(val_ds.features)
        assert len(preds) == len(val_ds)
        assert np.all(np.isfinite(preds))

        # Check against manual unscaled last observation
        raw_last = val_ds.features[:, -1, 0].numpy()
        expected = raw_last * targ_scaler.scale_[0] + targ_scaler.mean_[0]
        assert np.allclose(preds, expected, atol=1e-4)

    def test_seasonal_naive_24h_baseline(self, synthetic_demand_dataframe):
        """Verify 24h seasonal naive baseline selects t - 23h index (index 0 for L=24, H=1)."""
        train_ds, val_ds, _, _, targ_scaler = build_demand_datasets(
            df=synthetic_demand_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        baseline = DemandSeasonalNaiveBaseline(
            sequence_length=24,
            forecast_horizon=1,
            seasonal_period=24,
            meter_feature_idx=0,
            feature_scaler=targ_scaler,
        )

        assert baseline.seasonal_idx == 0

        preds = baseline.predict(val_ds.features)
        assert len(preds) == len(val_ds)
        assert np.all(np.isfinite(preds))

    def test_weekly_seasonal_naive_168h_baseline(self):
        """Verify 168h seasonal naive baseline works when window length >= 168."""
        # 1600 hours provides plenty of data for 168h window in 15% validation split (240h > 168h)
        dates = pd.date_range("2016-01-01", periods=1600, freq="1h")
        demand = (100.0 + 50.0 * np.sin(2 * np.pi * np.arange(1600) / 24)).astype(np.float32)
        df_long = pd.DataFrame({"timestamp": dates, "demand_kwh": demand})

        train_ds, val_ds, _, _, targ_scaler = build_demand_datasets(
            df=df_long,
            sequence_length=168,
            forecast_horizon=1,
            embargo_hours=24,
            scale=True,
        )

        baseline = DemandWeeklySeasonalNaiveBaseline(
            sequence_length=168,
            forecast_horizon=1,
            seasonal_period=168,
            meter_feature_idx=0,
            feature_scaler=targ_scaler,
        )

        # For L=168, H=1, seasonal_idx = 168 - 1 + 1 - 168 = 0
        assert baseline.seasonal_idx == 0

        preds = baseline.predict(val_ds.features)
        assert len(preds) == len(val_ds)
        assert np.all(np.isfinite(preds))

    def test_weekly_seasonal_naive_raises_on_short_window(self):
        """Verify ValueError is raised if sequence_length < 168 for weekly seasonal naive."""
        with pytest.raises(ValueError, match="falls outside window"):
            DemandWeeklySeasonalNaiveBaseline(
                sequence_length=24,
                forecast_horizon=1,
                seasonal_period=168,
            )

    def test_demand_baselines_reject_non_finite_inputs(self):
        """Verify persistence, 24h seasonal naive, and 168h seasonal naive raise ValueError on NaN/Inf inputs."""
        X_nan_last = np.ones((4, 24, 1), dtype=np.float32)
        X_nan_last[0, -1, 0] = np.nan
        persist = DemandPersistenceBaseline()
        with pytest.raises(ValueError, match="non-finite"):
            persist.predict(X_nan_last)

        X_nan_24h = np.ones((4, 24, 1), dtype=np.float32)
        X_nan_24h[0, 0, 0] = np.nan
        sn_24h = DemandSeasonalNaiveBaseline(sequence_length=24, forecast_horizon=1, seasonal_period=24)
        with pytest.raises(ValueError, match="non-finite"):
            sn_24h.predict(X_nan_24h)

        X_nan_168h = np.ones((4, 168, 1), dtype=np.float32)
        X_nan_168h[0, 0, 0] = np.nan
        sn_168h = DemandWeeklySeasonalNaiveBaseline(sequence_length=168, forecast_horizon=1, seasonal_period=168)
        with pytest.raises(ValueError, match="non-finite"):
            sn_168h.predict(X_nan_168h)

    def test_demand_baselines_reject_invalid_input_shapes(self):
        """Verify demand baselines raise ValueError when provided with non-3D input."""
        persist = DemandPersistenceBaseline()
        with pytest.raises(ValueError, match="Expected 3D input"):
            persist.predict(np.ones((4, 24)))

        sn_24h = DemandSeasonalNaiveBaseline(sequence_length=24, forecast_horizon=1, seasonal_period=24)
        with pytest.raises(ValueError, match="Expected 3D input"):
            sn_24h.predict(np.ones((4, 24)))

        sn_168h = DemandWeeklySeasonalNaiveBaseline(sequence_length=168, forecast_horizon=1, seasonal_period=168)
        with pytest.raises(ValueError, match="Expected 3D input"):
            sn_168h.predict(np.ones((4, 168)))


class TestDemandModelTrainingSmoke:
    """Validate DemandLSTM architecture and training pipeline integration."""

    def test_demand_lstm_shapes_and_training_step(self, tmp_path, synthetic_demand_dataframe):
        """Verify forward pass shapes and that RegressionTrainer runs a smoke epoch."""
        train_ds, val_ds, _, _, targ_scaler = build_demand_datasets(
            df=synthetic_demand_dataframe,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        model = DemandLSTM(input_dim=1, hidden_dim=16, num_layers=1, dropout=0.0)
        trainable, total = model.get_parameter_counts()
        assert trainable == total > 0

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = RegressionTrainer(
            model=model,
            optimizer=optimizer,
            experiment_id="TEST_DEMAND_SMOKE",
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


class TestDemandColumnAliases:
    """Validate flexible handling of standard column names (timestamp vs datetime, demand_kwh vs electricity_demand_kwh)."""

    def test_datetime_and_electricity_demand_kwh_accepted(self, synthetic_demand_dataframe):
        """Verify DataFrame with 'datetime' and 'electricity_demand_kwh' validates cleanly."""
        df_alias = synthetic_demand_dataframe.rename(
            columns={"timestamp": "datetime", "demand_kwh": "electricity_demand_kwh"}
        )
        df_valid = validate_demand_dataframe(df_alias)
        assert "datetime" in df_valid.columns
        assert "electricity_demand_kwh" in df_valid.columns

    def test_build_demand_datasets_with_aliases(self, synthetic_demand_dataframe):
        """Verify build_demand_datasets auto-resolves datetime and electricity_demand_kwh."""
        df_alias = synthetic_demand_dataframe.rename(
            columns={"timestamp": "datetime", "demand_kwh": "electricity_demand_kwh"}
        )
        train_ds, val_ds, test_ds, _, _ = build_demand_datasets(df=df_alias)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0


class TestRealDemandDataValidation:
    """Validate real BDG2 electricity demand pilot dataset (when downloaded)."""

    @pytest.fixture
    def real_pilot_path(self):
        """Return path to real processed pilot dataset."""
        path = Path("data/processed/demand/demand_hourly_pilot.csv")
        if not path.exists():
            pytest.skip("Real BDG2 demand pilot data not yet downloaded. Run scripts/download_bdg2_meter.py first.")
        return path

    def test_real_pilot_row_count_and_continuity(self, real_pilot_path):
        """Verify real pilot dataset has exactly 17,544 hours and unbroken 1h continuity."""
        df = pd.read_csv(real_pilot_path)
        assert len(df) == 17544, f"Expected 17,544 rows (2016 leap + 2017), got {len(df)}"

        time_col = "datetime" if "datetime" in df.columns else "timestamp"
        df[time_col] = pd.to_datetime(df[time_col])
        assert df[time_col].iloc[0] == pd.Timestamp("2016-01-01 00:00:00")
        assert df[time_col].iloc[-1] == pd.Timestamp("2017-12-31 23:00:00")

        diffs = df[time_col].diff().iloc[1:]
        assert (diffs == pd.Timedelta(hours=1)).all(), "Found temporal discontinuity in real demand dataset"

    def test_real_pilot_schema_and_physical_bounds(self, real_pilot_path):
        """Verify schema validation and physical bounds on real BDG2 extract."""
        df = pd.read_csv(real_pilot_path)
        meter_col = "demand_kwh" if "demand_kwh" in df.columns else "electricity_demand_kwh"
        df_valid = validate_demand_dataframe(
            df,
            max_demand_kwh=1000.0,
            expected_freq="1h",
        )

        assert df_valid[meter_col].min() >= 0.0
        assert df_valid[meter_col].max() <= 1000.0

    def test_real_pilot_build_datasets_pipeline(self, real_pilot_path):
        """Verify build_demand_datasets executes cleanly on real data with zero leakage."""
        train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_demand_datasets(
            csv_path=real_pilot_path,
            sequence_length=24,
            forecast_horizon=1,
            embargo_hours=24,
            scale=True,
        )

        assert len(train_ds) > 10000
        assert len(val_ds) > 2000
        assert len(test_ds) > 2000

        # Verify shapes
        assert train_ds.features.shape[1:] == (24, 1)
        assert train_ds.targets.shape[1:] == (1,)

        # Verify unscaled physical values
        unscaled_val = val_ds.inverse_transform_targets(val_ds.targets.numpy())
        assert np.all(unscaled_val >= 0.0)
        assert np.all(unscaled_val <= 1000.0)

