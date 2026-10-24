"""
Test suite for PyTorch Dataset classes and data pipeline builders.
Validates:
- Output tensor shapes and types
- Image decoding and transform pipelines
- Target alignment and metadata integrity
- Deterministic splitting
- Prevention of future information leakage
- Lightweight DataLoader smoke tests
"""
from pathlib import Path
import io
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    SkippdDataset,
    NasaWeatherDataset,
    build_skippd_datasets,
    build_nasa_datasets,
    get_default_skippd_transform,
)
from src.data.splits import split_skippd_by_day

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PARQUET = PROJECT_ROOT / "data" / "raw" / "skippd" / "train-00000-of-00005.parquet"
TEST_PARQUET = PROJECT_ROOT / "data" / "raw" / "skippd" / "test-00000-of-00001.parquet"
NASA_CSV = PROJECT_ROOT / "data" / "raw" / "nasa_power" / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"


# =====================================================================
# Synthetic Unit Tests
# =====================================================================

class TestSyntheticDatasets:
    """Isolated unit tests using synthetic fixtures."""

    @pytest.fixture
    def synthetic_parquet_table(self):
        """Create an in-memory PyArrow table mimicking SKIPP'D schema."""
        # Generate a small 64x64 solid color image as PNG bytes
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        n_rows = 15
        image_structs = [{"bytes": png_bytes, "path": None} for _ in range(n_rows)]
        timestamps = pd.date_range("2020-01-01 10:00", periods=n_rows, freq="1min", tz="US/Pacific")
        pv_values = [float(5.0 + i * 0.5) for i in range(n_rows)]

        table = pa.Table.from_arrays(
            [
                pa.array(image_structs),
                pa.array(timestamps),
                pa.array(pv_values, type=pa.float32()),
            ],
            names=["image", "time", "pv"],
        )
        return table

    def test_skippd_dataset_output_shapes_and_types(self, synthetic_parquet_table):
        """Verify tensor shapes, types, and value ranges from synthetic SkippdDataset."""
        ds = SkippdDataset(table_or_path=synthetic_parquet_table)
        assert len(ds) == 15

        img_tensor, pv_tensor = ds[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert isinstance(pv_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 64, 64)
        assert img_tensor.dtype == torch.float32
        assert pv_tensor.dtype == torch.float32
        assert torch.isclose(pv_tensor, torch.tensor(5.0, dtype=torch.float32))

    def test_skippd_dataset_with_subset_indices(self, synthetic_parquet_table):
        """Verify subset indexing works correctly."""
        subset = np.array([2, 5, 8])
        ds = SkippdDataset(table_or_path=synthetic_parquet_table, indices=subset)
        assert len(ds) == 3

        _, pv0 = ds[0]
        # Index 2 has pv = 5.0 + 2 * 0.5 = 6.0
        assert torch.isclose(pv0, torch.tensor(6.0, dtype=torch.float32))

    def test_skippd_dataset_metadata_return(self, synthetic_parquet_table):
        """Verify return_metadata mode contains required tracking fields."""
        ds = SkippdDataset(table_or_path=synthetic_parquet_table, return_metadata=True)
        item = ds[0]
        assert isinstance(item, dict)
        assert "image" in item and "pv" in item and "timestamp" in item and "index" in item
        assert item["index"] == 0

    def test_nasa_dataset_synthetic(self):
        """Verify NasaWeatherDataset output dimensions."""
        X = np.random.randn(20, 24, 4).astype(np.float32)
        y = np.random.randn(20).astype(np.float32)
        ds = NasaWeatherDataset(features=X, targets=y)
        assert len(ds) == 20

        x_item, y_item = ds[0]
        assert x_item.shape == (24, 4)
        assert y_item.shape == (1,)
        assert x_item.dtype == torch.float32
        assert y_item.dtype == torch.float32

    def test_deterministic_splitting(self):
        """Verify split_skippd_by_day produces identical splits on repeated calls."""
        dates = pd.date_range("2021-01-01 06:00", "2021-01-20 18:00", freq="10min")
        tr1, val1 = split_skippd_by_day(dates, train_ratio=0.75, chronological=True)
        tr2, val2 = split_skippd_by_day(dates, train_ratio=0.75, chronological=True)

        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(val1, val2)

    def test_nasa_dataframe_validation_missing_columns(self, tmp_path):
        """Verify validate_and_load_nasa_dataframe catches missing required columns."""
        from src.data.datasets import validate_and_load_nasa_dataframe
        csv_file = tmp_path / "bad_cols.csv"
        # Missing 'HR' and 'WS10M'
        content = "\n" * 12 + "YEAR,MO,DY,ALLSKY_SFC_SW_DWN,T2M,RH2M\n2023,1,1,0.0,20.0,60.0\n"
        csv_file.write_text(content)
        with pytest.raises(ValueError, match="missing required"):
            validate_and_load_nasa_dataframe(csv_file)

    def test_nasa_dataframe_validation_hourly_gap(self, tmp_path):
        """Verify validate_and_load_nasa_dataframe detects non-continuous hourly intervals."""
        from src.data.datasets import validate_and_load_nasa_dataframe
        csv_file = tmp_path / "gap.csv"
        # Hour skips from 0 to 2 (missing hour 1)
        header = "\n" * 12 + "YEAR,MO,DY,HR,ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M\n"
        rows = [
            "2023,1,1,0,0.0,20.0,60.0,2.0",
            "2023,1,1,2,0.0,19.5,62.0,2.1",
        ]
        csv_file.write_text(header + "\n".join(rows) + "\n")
        with pytest.raises(ValueError, match="not continuous hourly"):
            validate_and_load_nasa_dataframe(csv_file)

    def test_nasa_dataframe_validation_duplicate_timestamps(self, tmp_path):
        """Verify validate_and_load_nasa_dataframe rejects duplicate hourly timestamps."""
        from src.data.datasets import validate_and_load_nasa_dataframe
        csv_file = tmp_path / "dup.csv"
        header = "\n" * 12 + "YEAR,MO,DY,HR,ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M\n"
        rows = [
            "2023,1,1,0,0.0,20.0,60.0,2.0",
            "2023,1,1,0,0.0,20.0,60.0,2.0",
        ]
        csv_file.write_text(header + "\n".join(rows) + "\n")
        with pytest.raises(ValueError, match="duplicate timestamps"):
            validate_and_load_nasa_dataframe(csv_file)

    def test_nasa_dataframe_validation_nan_or_sentinel(self, tmp_path):
        """Verify validate_and_load_nasa_dataframe detects NaNs and sentinel values (<= -900)."""
        from src.data.datasets import validate_and_load_nasa_dataframe
        csv_file = tmp_path / "sentinel.csv"
        header = "\n" * 12 + "YEAR,MO,DY,HR,ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M\n"
        rows = [
            "2023,1,1,0,-999.0,20.0,60.0,2.0",
            "2023,1,1,1,0.0,20.0,60.0,2.0",
        ]
        csv_file.write_text(header + "\n".join(rows) + "\n")
        with pytest.raises(ValueError, match="sentinel missing values"):
            validate_and_load_nasa_dataframe(csv_file)

    def test_nasa_weather_dataset_inverse_transform(self):
        """Verify NasaWeatherDataset inverse_transform_targets recovers original values."""
        from sklearn.preprocessing import StandardScaler
        raw_targets = np.array([0.0, 250.5, 680.2, 940.0, 110.3], dtype=np.float32)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw_targets.reshape(-1, 1)).flatten()

        X = np.zeros((len(raw_targets), 24, 4), dtype=np.float32)
        ds = NasaWeatherDataset(features=X, targets=scaled, target_scaler=scaler)

        recovered = ds.inverse_transform_targets(ds.targets)
        np.testing.assert_allclose(recovered.flatten(), raw_targets, atol=1e-4)


# =====================================================================
# Real-File Integration Tests
# =====================================================================

class TestRealDataIntegration:
    """Integration tests on actual physical files in data/raw/."""

    @pytest.mark.skipif(not TRAIN_PARQUET.exists(), reason="SKIPP'D train parquet missing")
    def test_skippd_real_sample_decoding(self):
        """Verify reading and decoding actual 64x64 sky images from Parquet."""
        ds = SkippdDataset(
            table_or_path=TRAIN_PARQUET,
            indices=np.arange(10),
            transform=get_default_skippd_transform(normalize=True),
            return_metadata=True,
        )
        assert len(ds) == 10

        item0 = ds[0]
        img = item0["image"]
        pv = item0["pv"]
        ts = item0["timestamp"]

        assert img.shape == (3, 64, 64)
        assert torch.isfinite(img).all()
        # PV output should be in plausible range [0, 30] kW
        assert 0.0 <= pv.item() <= 30.0
        assert "2017-03-09" in ts

    @pytest.mark.skipif(not TRAIN_PARQUET.exists(), reason="SKIPP'D train parquet missing")
    def test_skippd_memory_safety(self):
        """Verify in-memory footprint of SKIPP'D table is within safe limits for 16GB RAM."""
        import pyarrow.parquet as pq
        table = pq.read_table(TRAIN_PARQUET)
        mem_mb = table.nbytes / (1024 * 1024)
        # Table should be ~427 MB, well below 1 GB safety ceiling
        assert mem_mb < 600.0, f"SKIPP'D in-memory footprint unexpectedly large: {mem_mb:.1f} MB"

    @pytest.mark.skipif(not TRAIN_PARQUET.exists() or not TEST_PARQUET.exists(), reason="SKIPP'D files missing")
    def test_build_skippd_datasets_calendar_day_isolation(self):
        """Verify build_skippd_datasets ensures zero day overlap between train and val."""
        train_ds, val_ds, test_ds = build_skippd_datasets(
            train_parquet_path=TRAIN_PARQUET,
            test_parquet_path=TEST_PARQUET,
            train_ratio=0.8,
            chronological=True,
        )

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) == 14003

        # Check calendar dates of train vs val
        train_times = train_ds.get_timestamps()
        val_times = val_ds.get_timestamps()

        train_days = set(train_times.dt.date)
        val_days = set(val_times.dt.date)

        assert len(train_days & val_days) == 0, "Train and Val share calendar days!"
        assert max(train_days) < min(val_days), "Chronological ordering violated"

    @pytest.mark.skipif(not NASA_CSV.exists(), reason="NASA POWER CSV missing")
    def test_nasa_real_timestamp_hourly_continuity(self):
        """Verify real NASA POWER CSV satisfies strict hourly continuity for all 8,760 hours."""
        from src.data.datasets import validate_and_load_nasa_dataframe
        df = validate_and_load_nasa_dataframe(NASA_CSV)
        assert len(df) == 8760
        diffs = df["datetime"].diff().iloc[1:]
        assert (diffs == pd.Timedelta(hours=1)).all()

    @pytest.mark.skipif(not NASA_CSV.exists(), reason="NASA POWER CSV missing")
    def test_build_nasa_datasets_window_integrity(self):
        """Verify build_nasa_datasets window shapes and target causality on real data."""
        train_ds, val_ds, test_ds, f_scaler, t_scaler = build_nasa_datasets(
            csv_path=NASA_CSV,
            sequence_length=24,
            forecast_horizon=1,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_hours=24,
            scale=True,
        )

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert f_scaler is not None
        assert t_scaler is not None

        # Sample check
        x, y = train_ds[0]
        assert x.shape == (24, 4)
        assert y.shape == (1,)
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()

    @pytest.mark.skipif(not NASA_CSV.exists(), reason="NASA POWER CSV missing")
    def test_nasa_window_precise_origin_and_target_alignment(self):
        """Verify exact 1-hour-ahead alignment: target timestamp is exactly origin timestamp + 1h."""
        train_ds, val_ds, test_ds, _, _ = build_nasa_datasets(
            csv_path=NASA_CSV,
            sequence_length=24,
            forecast_horizon=1,
            embargo_hours=24,
            scale=True,
        )

        for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            origins = pd.to_datetime(ds.origin_timestamps)
            targets = pd.to_datetime(ds.target_timestamps)
            deltas = targets - origins

            # Check that every single target is exactly 1 hour ahead of its origin
            assert (deltas == pd.Timedelta(hours=1)).all(), f"{name} split has non-1h forecast horizon deltas"

            # Check specific alignment on first sample
            t_orig0, t_targ0 = ds.get_time_alignment(0)
            assert t_targ0 - t_orig0 == pd.Timedelta(hours=1)

    @pytest.mark.skipif(not NASA_CSV.exists(), reason="NASA POWER CSV missing")
    def test_nasa_split_boundaries_and_embargo_isolation(self):
        """Verify disjoint target timestamps and >= 24h embargo gap between all split boundaries."""
        train_ds, val_ds, test_ds, _, _ = build_nasa_datasets(
            csv_path=NASA_CSV,
            sequence_length=24,
            forecast_horizon=1,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_hours=24,
            scale=True,
        )

        train_targets = pd.to_datetime(train_ds.target_timestamps)
        val_targets = pd.to_datetime(val_ds.target_timestamps)
        test_targets = pd.to_datetime(test_ds.target_timestamps)

        # 1. Zero target overlap
        assert len(set(train_targets) & set(val_targets)) == 0, "Train and Val share target timestamps"
        assert len(set(val_targets) & set(test_targets)) == 0, "Val and Test share target timestamps"
        assert len(set(train_targets) & set(test_targets)) == 0, "Train and Test share target timestamps"

        # 2. Strict chronological ordering with embargo gap
        gap_train_val = val_targets.min() - train_targets.max()
        gap_val_test = test_targets.min() - val_targets.max()

        assert gap_train_val >= pd.Timedelta(hours=24), f"Train-Val embargo gap {gap_train_val} is < 24h"
        assert gap_val_test >= pd.Timedelta(hours=24), f"Val-Test embargo gap {gap_val_test} is < 24h"

    @pytest.mark.skipif(not NASA_CSV.exists(), reason="NASA POWER CSV missing")
    def test_nasa_real_target_unscaling_fidelity(self):
        """Verify real NASA targets unscale back to physical Wh/m² with high fidelity."""
        # Unscaled build
        train_raw, _, _, _, _ = build_nasa_datasets(
            csv_path=NASA_CSV,
            sequence_length=24,
            forecast_horizon=1,
            scale=False,
        )
        # Scaled build
        train_scaled, _, _, _, _ = build_nasa_datasets(
            csv_path=NASA_CSV,
            sequence_length=24,
            forecast_horizon=1,
            scale=True,
        )

        unscaled_recovered = train_scaled.inverse_transform_targets(train_scaled.targets)
        np.testing.assert_allclose(
            unscaled_recovered,
            train_raw.targets.numpy(),
            atol=1e-3,
            err_msg="Inverse transformed targets do not match raw physical targets",
        )

    @pytest.mark.skipif(not TRAIN_PARQUET.exists() or not NASA_CSV.exists(), reason="Datasets missing")
    def test_dataloader_lightweight_smoke_test(self):
        """Smoke test verifying PyTorch DataLoader batches without errors or deadlocks."""
        # SKIPP'D mini-loader
        skippd_ds = SkippdDataset(
            table_or_path=TRAIN_PARQUET,
            indices=np.arange(16),
            transform=get_default_skippd_transform(normalize=False),
        )
        skippd_loader = DataLoader(skippd_ds, batch_size=4, shuffle=False)
        batch_imgs, batch_pvs = next(iter(skippd_loader))

        assert batch_imgs.shape == (4, 3, 64, 64)
        assert batch_pvs.shape == (4,)

        # NASA mini-loader
        _, val_ds, _, _, _ = build_nasa_datasets(
            csv_path=NASA_CSV,
            sequence_length=24,
            forecast_horizon=1,
            embargo_hours=24,
            scale=True,
        )
        nasa_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        batch_x, batch_y = next(iter(nasa_loader))

        assert batch_x.shape == (8, 24, 4)
        assert batch_y.shape == (8, 1)
