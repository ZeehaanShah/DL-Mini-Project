"""
Comprehensive test suite for data leakage prevention across all datasets.
Validates:
- LC-01: No calendar day overlap between train and validation splits (SKIPP'D)
- LC-02: Zero shared timestamps between train shard and test parquet (SKIPP'D)
- LC-03: No lookahead bias in time-series sliding windows (NASA POWER)
- LC-04: Scaler fit strictly on training partition only
- LC-05: Strict temporal ordering and embargo gap between chronological splits
- Cross-dataset geographic and temporal isolation
"""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.splits import (
    split_skippd_by_day,
    split_timeseries_chronological,
    create_forecasting_windows,
    fit_scaler_train_only,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKIPPD_DIR = PROJECT_ROOT / "data" / "raw" / "skippd"
NASA_DIR = PROJECT_ROOT / "data" / "raw" / "nasa_power"

TRAIN_PARQUET = SKIPPD_DIR / "train-00000-of-00005.parquet"
TEST_PARQUET = SKIPPD_DIR / "test-00000-of-00001.parquet"
NASA_CSV = NASA_DIR / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"


class TestSkippdLeakage:
    """Validate zero leakage between SKIPP'D partitions."""

    @pytest.mark.skipif(not TRAIN_PARQUET.exists() or not TEST_PARQUET.exists(), reason="SKIPP'D files missing")
    def test_train_and_test_parquet_share_zero_calendar_days(self):
        """LC-01 / LC-02: Ensure train shard 1 and test parquet have 0 shared calendar days."""
        tr_table = pq.read_table(TRAIN_PARQUET, columns=["time"])
        te_table = pq.read_table(TEST_PARQUET, columns=["time"])

        tr_times = tr_table.to_pandas()["time"]
        te_times = te_table.to_pandas()["time"]

        tr_days = set(tr_times.dt.date)
        te_days = set(te_times.dt.date)

        shared_days = tr_days & te_days
        assert len(shared_days) == 0, f"Leakage detected: {len(shared_days)} shared calendar days: {shared_days}"

    @pytest.mark.skipif(not TRAIN_PARQUET.exists() or not TEST_PARQUET.exists(), reason="SKIPP'D files missing")
    def test_train_and_test_parquet_share_zero_timestamps(self):
        """LC-02: Ensure train shard 1 and test parquet have 0 identical timestamps."""
        tr_table = pq.read_table(TRAIN_PARQUET, columns=["time"])
        te_table = pq.read_table(TEST_PARQUET, columns=["time"])

        tr_times = set(tr_table.to_pandas()["time"])
        te_times = set(te_table.to_pandas()["time"])

        shared = tr_times & te_times
        assert len(shared) == 0, f"Leakage detected: {len(shared)} shared exact timestamps"

    def test_calendar_day_split_function_zero_overlap(self):
        """LC-01: Verify split_skippd_by_day produces zero shared days and indices."""
        # Synthesize a multi-day minute-level timestamp series
        dr = pd.date_range("2020-01-01 06:00", "2020-01-10 18:00", freq="1min")
        train_idx, val_idx = split_skippd_by_day(dr, train_ratio=0.8, chronological=True)

        assert len(set(train_idx) & set(val_idx)) == 0, "Indices overlap between train and val"
        assert len(train_idx) + len(val_idx) == len(dr), "Index count mismatch"

        train_days = set(dr[train_idx].date)
        val_days = set(dr[val_idx].date)
        assert len(train_days & val_days) == 0, "Shared calendar days in split_skippd_by_day"

    @pytest.mark.skipif(not TRAIN_PARQUET.exists(), reason="SKIPP'D train file missing")
    def test_calendar_day_split_on_actual_skippd_train_shard(self):
        """LC-01: Verify calendar day split on actual 69,875 train shard timestamps."""
        tr_table = pq.read_table(TRAIN_PARQUET, columns=["time"])
        tr_times = tr_table.to_pandas()["time"]

        train_idx, val_idx = split_skippd_by_day(tr_times, train_ratio=0.8, chronological=True)

        train_days = set(tr_times.iloc[train_idx].dt.date)
        val_days = set(tr_times.iloc[val_idx].dt.date)

        assert len(train_days & val_days) == 0, "Shared calendar days in actual train shard split"
        # Since chronological is True, max train date must be before min val date
        assert max(train_days) < min(val_days), "Chronological ordering violated in day split"


class TestNasaLeakage:
    """Validate zero temporal leakage in NASA POWER sequence forecasting."""

    @pytest.fixture(scope="class")
    def nasa_data(self):
        if not NASA_CSV.exists():
            pytest.skip("NASA CSV not found")
        df = pd.read_csv(NASA_CSV, skiprows=12)
        df["datetime"] = pd.to_datetime(
            df[["YEAR", "MO", "DY"]].rename(
                columns={"YEAR": "year", "MO": "month", "DY": "day"}
            ).assign(hour=df["HR"])
        )
        return df

    def test_chronological_split_strict_ordering(self, nasa_data):
        """LC-05: Ensure train times < val times < test times with embargo."""
        train_df, val_df, test_df = split_timeseries_chronological(
            nasa_data,
            time_col="datetime",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_steps=24,
        )

        assert train_df["datetime"].max() < val_df["datetime"].min()
        assert val_df["datetime"].max() < test_df["datetime"].min()

    def test_chronological_split_embargo_gap(self, nasa_data):
        """LC-05: Check that embargo gap is at least 24 hours."""
        embargo = 24
        train_df, val_df, test_df = split_timeseries_chronological(
            nasa_data,
            time_col="datetime",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_steps=embargo,
        )

        train_to_val_gap = val_df["datetime"].min() - train_df["datetime"].max()
        val_to_test_gap = test_df["datetime"].min() - val_df["datetime"].max()

        # At hourly resolution, embargo of 24 steps creates a gap of 25 hours between max and min
        assert train_to_val_gap >= pd.Timedelta(hours=embargo)
        assert val_to_test_gap >= pd.Timedelta(hours=embargo)

    def test_forecasting_windows_no_lookahead_bias(self, nasa_data):
        """LC-03: Ensure window input timestamps are strictly prior to target timestamp."""
        seq_len = 24
        horizon = 1
        features = nasa_data[["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]].to_numpy()
        targets = nasa_data["ALLSKY_SFC_SW_DWN"].to_numpy()
        timestamps = nasa_data["datetime"].to_numpy()

        X, y = create_forecasting_windows(
            features=features,
            targets=targets,
            sequence_length=seq_len,
            forecast_horizon=horizon,
        )

        # Build windowed timestamps to strictly verify ordering
        time_windows, target_times = create_forecasting_windows(
            features=timestamps.reshape(-1, 1),
            targets=timestamps,
            sequence_length=seq_len,
            forecast_horizon=horizon,
        )

        for i in [0, 100, 500, len(X) - 1]:
            max_input_time = time_windows[i, :, 0].max()
            target_time = target_times[i]
            assert max_input_time < target_time, \
                f"Lookahead violation at sample {i}: input {max_input_time} >= target {target_time}"

    def test_scaler_fitted_on_train_only(self, nasa_data):
        """LC-04: Ensure StandardScaler fit exclusively uses training split parameters."""
        train_df, val_df, test_df = split_timeseries_chronological(
            nasa_data,
            time_col="datetime",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_steps=24,
        )

        train_vals = train_df[["ALLSKY_SFC_SW_DWN", "T2M"]].to_numpy()
        val_vals = val_df[["ALLSKY_SFC_SW_DWN", "T2M"]].to_numpy()
        test_vals = test_df[["ALLSKY_SFC_SW_DWN", "T2M"]].to_numpy()

        scaler, tr_scaled, v_scaled, te_scaled = fit_scaler_train_only(
            train_vals, val_vals, test_vals
        )

        # Scaler mean must exactly match training set mean
        np.testing.assert_allclose(scaler.mean_, train_vals.mean(axis=0))
        # Scaler scale must exactly match training set std
        np.testing.assert_allclose(scaler.scale_, train_vals.std(axis=0))

        # Check that val scaled mean is NOT 0.0 (because it wasn't centered on val)
        assert not np.allclose(v_scaled.mean(axis=0), 0.0, atol=1e-2), \
            "Validation data appears to have been used to fit the scaler"


    def test_forecasting_windows_no_overlap_across_splits_on_actual_nasa(self, nasa_data):
        """LC-03 / LC-05: Ensure windows generated on splits share zero target timestamps and zero boundary overlap."""
        seq_len = 24
        horizon = 1
        embargo = 24

        train_df, val_df, test_df = split_timeseries_chronological(
            nasa_data,
            time_col="datetime",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_steps=embargo,
        )

        # Build windowed timestamps for train, val, and test
        _, train_target_times = create_forecasting_windows(
            features=train_df["datetime"].to_numpy().reshape(-1, 1),
            targets=train_df["datetime"].to_numpy(),
            sequence_length=seq_len,
            forecast_horizon=horizon,
        )
        _, val_target_times = create_forecasting_windows(
            features=val_df["datetime"].to_numpy().reshape(-1, 1),
            targets=val_df["datetime"].to_numpy(),
            sequence_length=seq_len,
            forecast_horizon=horizon,
        )
        _, test_target_times = create_forecasting_windows(
            features=test_df["datetime"].to_numpy().reshape(-1, 1),
            targets=test_df["datetime"].to_numpy(),
            sequence_length=seq_len,
            forecast_horizon=horizon,
        )

        # 1. Zero target timestamp overlap between splits
        assert len(set(train_target_times) & set(val_target_times)) == 0, "Train and Val share target timestamps"
        assert len(set(val_target_times) & set(test_target_times)) == 0, "Val and Test share target timestamps"
        assert len(set(train_target_times) & set(test_target_times)) == 0, "Train and Test share target timestamps"

        # 2. Strict chronological order across split targets
        assert train_target_times.max() < val_target_times.min(), "Train target exceeds Val target timestamp"
        assert val_target_times.max() < test_target_times.min(), "Val target exceeds Test target timestamp"


class TestCrossDatasetIsolation:
    """Verify that independent datasets are never conflated or unified inappropriately."""

    def test_geographic_metadata_isolation(self):
        """Confirm datasets belong to completely different geographic coordinates using actual NASA header."""
        if not NASA_CSV.exists():
            pytest.skip("NASA CSV not found")

        # Parse real latitude and longitude from line 3 of NASA POWER CSV
        nasa_lat, nasa_lon = None, None
        with open(NASA_CSV, "r") as f:
            for _ in range(12):
                line = f.readline()
                if "Location: Latitude" in line:
                    parts = line.split()
                    nasa_lat = float(parts[2])
                    nasa_lon = float(parts[4])
                    break

        assert nasa_lat is not None and nasa_lon is not None, "Failed to parse NASA coordinates from header"

        # SKIPP'D: Stanford rooftop array (~37.42N, -122.17W)
        skippd_lat, skippd_lon = 37.42, -122.17

        # Distance check: they are in completely different continents/hemispheres
        lat_diff = abs(skippd_lat - nasa_lat)
        lon_diff = abs(skippd_lon - nasa_lon)

        assert lat_diff > 15.0, "Locations too close to be independent"
        assert lon_diff > 100.0, "Locations too close to be independent"
