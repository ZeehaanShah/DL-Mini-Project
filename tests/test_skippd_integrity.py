"""
Test suite for SKIPP'D dataset integrity.
Tests schema, data quality, temporal properties, and image validity.
Uses bounded reads to avoid loading the full dataset into memory.
"""
import pytest
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "skippd"

# Skip all tests if data files are not present
TRAIN_PARQUET = DATA_RAW / "train-00000-of-00005.parquet"
TEST_PARQUET = DATA_RAW / "test-00000-of-00001.parquet"
TIMES_TRAINVAL = DATA_RAW / "times_trainval.npy"
TIMES_TEST = DATA_RAW / "times_test.npy"

pytestmark = pytest.mark.skipif(
    not TRAIN_PARQUET.exists(),
    reason="SKIPP'D data files not found"
)


class TestSkippdSchema:
    """Test that parquet files have the expected schema."""

    def test_train_parquet_exists(self):
        assert TRAIN_PARQUET.exists(), "Train parquet file not found"

    def test_test_parquet_exists(self):
        assert TEST_PARQUET.exists(), "Test parquet file not found"

    def test_train_schema_fields(self):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(TRAIN_PARQUET)
        schema = pf.schema_arrow
        field_names = [schema.field(i).name for i in range(len(schema))]
        assert "image" in field_names, "Missing 'image' field"
        assert "time" in field_names, "Missing 'time' field"
        assert "pv" in field_names, "Missing 'pv' field"

    def test_test_schema_fields(self):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(TEST_PARQUET)
        schema = pf.schema_arrow
        field_names = [schema.field(i).name for i in range(len(schema))]
        assert "image" in field_names, "Missing 'image' field"
        assert "time" in field_names, "Missing 'time' field"
        assert "pv" in field_names, "Missing 'pv' field"

    def test_train_schema_types(self):
        import pyarrow.parquet as pq
        import pyarrow as pa
        pf = pq.ParquetFile(TRAIN_PARQUET)
        schema = pf.schema_arrow
        # pv should be float
        pv_idx = [i for i in range(len(schema)) if schema.field(i).name == "pv"][0]
        pv_type = schema.field(pv_idx).type
        assert pv_type == pa.float32() or pv_type == pa.float64(), f"pv type is {pv_type}, expected float"

    def test_train_row_count(self):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(TRAIN_PARQUET)
        assert pf.metadata.num_rows == 69875, f"Expected 69875 rows, got {pf.metadata.num_rows}"

    def test_test_row_count(self):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(TEST_PARQUET)
        assert pf.metadata.num_rows == 14003, f"Expected 14003 rows, got {pf.metadata.num_rows}"


class TestSkippdDataQuality:
    """Test data quality properties."""

    def test_pv_no_nan(self):
        """PV target should have no NaN values across all 69,875 rows in train shard."""
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(TRAIN_PARQUET)
        table = pf.read(columns=["pv"])
        pv = table.column("pv").to_numpy()
        assert (pv != pv).sum() == 0, "Found NaN values in PV"

    def test_pv_non_negative(self):
        """PV power output should never be negative across all rows."""
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(TRAIN_PARQUET)
        table = pf.read(columns=["pv"])
        pv = table.column("pv").to_numpy()
        assert (pv >= 0).all(), "Found negative PV values"

    def test_pv_finite(self):
        """PV values should all be finite across all rows."""
        import pyarrow.parquet as pq
        import numpy as np
        pf = pq.ParquetFile(TRAIN_PARQUET)
        table = pf.read(columns=["pv"])
        pv = table.column("pv").to_numpy()
        assert np.isfinite(pv).all(), "Found non-finite PV values"

    def test_timestamps_not_null(self):
        """Timestamps should not be null across all rows."""
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(TRAIN_PARQUET)
        table = pf.read(columns=["time"])
        assert table.column("time").null_count == 0, "Found null timestamps"

    def test_image_decodable(self):
        """First image should be decodable as a valid PNG."""
        import pyarrow.parquet as pq
        from PIL import Image
        import io

        pf = pq.ParquetFile(TRAIN_PARQUET)
        batch = next(pf.iter_batches(batch_size=1))
        df = batch.to_pandas()
        img_data = df["image"].iloc[0]

        # Extract bytes from struct
        if isinstance(img_data, dict):
            img_bytes = img_data["bytes"]
        else:
            img_bytes = img_data

        img = Image.open(io.BytesIO(img_bytes))
        assert img.size == (64, 64), f"Expected 64x64, got {img.size}"
        assert img.mode == "RGB", f"Expected RGB, got {img.mode}"

    def test_image_batch_decodable(self):
        """First 10 images should all be decodable."""
        import pyarrow.parquet as pq
        from PIL import Image
        import io

        pf = pq.ParquetFile(TRAIN_PARQUET)
        batch = next(pf.iter_batches(batch_size=10))
        df = batch.to_pandas()

        for i in range(len(df)):
            img_data = df["image"].iloc[i]
            img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
            img = Image.open(io.BytesIO(img_bytes))
            assert img.size == (64, 64), f"Image {i}: wrong size {img.size}"
            assert img.mode == "RGB", f"Image {i}: wrong mode {img.mode}"


class TestSkippdTemporal:
    """Test temporal properties of the dataset."""

    def test_train_timestamps_monotonic(self):
        """Train shard timestamps should be monotonically increasing."""
        import pyarrow.parquet as pq
        import pandas as pd

        pf = pq.ParquetFile(TRAIN_PARQUET)
        times = []
        for batch in pf.iter_batches(batch_size=5000, columns=["time"]):
            times.extend(batch.to_pandas()["time"].tolist())

        ts = pd.Series(times)
        assert ts.is_monotonic_increasing, "Train timestamps are not monotonically increasing"

    def test_npy_trainval_no_duplicates(self):
        """times_trainval.npy should have no duplicate timestamps."""
        if not TIMES_TRAINVAL.exists():
            pytest.skip("times_trainval.npy not found")

        import numpy as np
        import pandas as pd
        arr = np.load(TIMES_TRAINVAL, allow_pickle=True)
        ts = pd.to_datetime(arr)
        assert ts.duplicated().sum() == 0, "Duplicate timestamps in trainval"

    def test_npy_test_no_duplicates(self):
        """times_test.npy should have no duplicate timestamps."""
        if not TIMES_TEST.exists():
            pytest.skip("times_test.npy not found")

        import numpy as np
        import pandas as pd
        arr = np.load(TIMES_TEST, allow_pickle=True)
        ts = pd.to_datetime(arr)
        assert ts.duplicated().sum() == 0, "Duplicate timestamps in test"

    def test_npy_no_timestamp_overlap(self):
        """trainval and test npy arrays should have zero overlapping timestamps."""
        if not (TIMES_TRAINVAL.exists() and TIMES_TEST.exists()):
            pytest.skip("npy files not found")

        import numpy as np
        import pandas as pd
        tv = np.load(TIMES_TRAINVAL, allow_pickle=True)
        te = np.load(TIMES_TEST, allow_pickle=True)
        overlap = set(tv) & set(te)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping timestamps"

    def test_npy_test_count_matches_parquet(self):
        """times_test.npy length should match test parquet row count."""
        if not TIMES_TEST.exists():
            pytest.skip("times_test.npy not found")

        import numpy as np
        import pyarrow.parquet as pq

        arr = np.load(TIMES_TEST, allow_pickle=True)
        pf = pq.ParquetFile(TEST_PARQUET)
        assert len(arr) == pf.metadata.num_rows, \
            f"npy has {len(arr)} entries, parquet has {pf.metadata.num_rows} rows"

    def test_npy_test_timestamps_differ_from_parquet(self):
        """
        IMPORTANT: Verify that npy test timestamps do NOT align with parquet.
        This is a KNOWN issue — the test documents the mismatch.
        """
        if not TIMES_TEST.exists():
            pytest.skip("times_test.npy not found")

        import numpy as np
        import pyarrow.parquet as pq
        import pandas as pd

        arr = np.load(TIMES_TEST, allow_pickle=True)
        npy_first = pd.Timestamp(arr[0])

        pf = pq.ParquetFile(TEST_PARQUET)
        batch = next(pf.iter_batches(batch_size=1, columns=["time"]))
        parquet_first = batch.to_pandas()["time"].iloc[0].tz_localize(None)

        # This test DOCUMENTS the known mismatch
        # If they suddenly match, something changed and we need to investigate
        assert npy_first != parquet_first, \
            "NPY and parquet first timestamps match — unexpected! Investigate if data source changed."
