"""
SKIPP'D deep audit — fix schema iteration and npy pickle loading.
"""
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "skippd"

def audit_parquet(label, fpath):
    print(f"\n{'='*60}")
    print(f"  {label} Parquet: {fpath.name}")
    print(f"{'='*60}")
    
    if not fpath.exists():
        print(f"  FILE NOT FOUND")
        return
    
    print(f"  File size: {fpath.stat().st_size / (1024**2):.2f} MB")
    
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pandas as pd
    
    pf = pq.ParquetFile(fpath)
    meta = pf.metadata
    schema = pf.schema_arrow
    
    print(f"  Rows: {meta.num_rows}")
    print(f"  Columns: {meta.num_columns}")
    print(f"  Row groups: {meta.num_row_groups}")
    print(f"  Created by: {meta.created_by}")
    
    # Schema via iteration
    print(f"\n  Schema ({len(schema)} fields):")
    for i in range(len(schema)):
        field = schema.field(i)
        print(f"    [{i}] {field.name}: {field.type}")
    
    # Read just 5 rows from first row group
    print(f"\n  Reading first 5 rows from row group 0...")
    batch = next(pf.iter_batches(batch_size=5))
    df = batch.to_pandas()
    
    print(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        print(f"\n  Column: '{col}' (dtype={df[col].dtype})")
        for row_idx in range(min(3, len(df))):
            val = df[col].iloc[row_idx]
            if isinstance(val, dict):
                print(f"    row[{row_idx}]: dict with keys={list(val.keys())}")
                for k, v in val.items():
                    if isinstance(v, bytes):
                        print(f"      {k}: bytes, len={len(v)}")
                    else:
                        print(f"      {k}: {type(v).__name__} = {repr(v)[:120]}")
            elif isinstance(val, bytes):
                print(f"    row[{row_idx}]: bytes, len={len(val)}")
            elif isinstance(val, (float, int)):
                print(f"    row[{row_idx}]: {val}")
            else:
                print(f"    row[{row_idx}]: {type(val).__name__} = {repr(val)[:200]}")
    
    # Try image decode from first row
    for col in df.columns:
        val = df[col].iloc[0]
        img_bytes = None
        if isinstance(val, dict) and 'bytes' in val:
            img_bytes = val['bytes']
        elif isinstance(val, bytes):
            img_bytes = val
        
        if img_bytes:
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_bytes))
                print(f"\n  IMAGE DECODED from '{col}':")
                print(f"    Size: {img.size}")
                print(f"    Mode: {img.mode}")
                print(f"    Format: {img.format}")
                print(f"    Channels: {len(img.getbands())}, bands: {img.getbands()}")
            except Exception as e:
                print(f"  Image decode failed for '{col}': {e}")
    
    # Check numeric columns stats across full first row group
    print(f"\n  Full row group 0 stats:")
    table_rg0 = pf.read_row_group(0)
    df_rg0 = table_rg0.to_pandas()
    for col in df_rg0.columns:
        if df_rg0[col].dtype in ['float64', 'float32', 'int64', 'int32', 'float16']:
            s = df_rg0[col]
            print(f"    {col}: count={len(s)}, min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}, nan={s.isna().sum()}")
    
    null_total = df_rg0.isna().sum().sum()
    print(f"    Total nulls in RG0: {null_total}")
    
    del df_rg0, table_rg0, df, batch


def audit_npy(label, fpath):
    print(f"\n{'='*60}")
    print(f"  NumPy: {label} ({fpath.name})")
    print(f"{'='*60}")
    
    if not fpath.exists():
        print("  FILE NOT FOUND")
        return
    
    print(f"  File size: {fpath.stat().st_size / (1024**2):.2f} MB")
    
    import numpy as np
    
    # Try without pickle first, then with
    try:
        arr = np.load(fpath, allow_pickle=False)
    except ValueError:
        print("  Requires allow_pickle=True (contains object array)")
        arr = np.load(fpath, allow_pickle=True)
    
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  ndim: {arr.ndim}")
    
    if arr.dtype == object:
        print(f"  Object array - inspecting elements:")
        print(f"    First element type: {type(arr[0]).__name__}")
        print(f"    First 5 elements: {arr[:5]}")
        print(f"    Last 5 elements: {arr[-5:]}")
        
        # Check if elements are strings (timestamps)
        if isinstance(arr[0], str):
            print(f"    String format sample: '{arr[0]}'")
            print(f"    String lengths (first 5): {[len(str(x)) for x in arr[:5]]}")
            # Try parsing as datetime
            try:
                import pandas as pd
                ts = pd.to_datetime(arr[:5])
                print(f"    Parsed as datetime: {ts.tolist()}")
                ts_all = pd.to_datetime(arr)
                print(f"    Date range: {ts_all.min()} to {ts_all.max()}")
                print(f"    Total timestamps: {len(ts_all)}")
                print(f"    Unique timestamps: {ts_all.nunique()}")
                # Check for duplicates
                dups = ts_all.duplicated().sum()
                print(f"    Duplicate timestamps: {dups}")
                # Check ordering
                is_sorted = ts_all.is_monotonic_increasing
                print(f"    Monotonically increasing: {is_sorted}")
            except Exception as e:
                print(f"    Datetime parse failed: {e}")
        elif isinstance(arr[0], (int, float, np.integer, np.floating)):
            print(f"    Numeric range: {arr.min()} to {arr.max()}")
    else:
        print(f"  First 5: {arr[:5]}")
        print(f"  Last 5: {arr[-5:]}")
        if np.issubdtype(arr.dtype, np.number):
            print(f"  Min: {arr.min()}, Max: {arr.max()}")
            print(f"  Unique: {len(np.unique(arr))}")
    
    del arr


if __name__ == "__main__":
    print("SKIPP'D Deep Audit")
    print(f"Run: {__import__('datetime').datetime.now().isoformat()}")
    
    audit_parquet("Train", DATA_RAW / "train-00000-of-00005.parquet")
    audit_parquet("Test", DATA_RAW / "test-00000-of-00001.parquet")
    audit_npy("times_trainval", DATA_RAW / "times_trainval.npy")
    audit_npy("times_test", DATA_RAW / "times_test.npy")
    
    # Cross-reference counts
    print(f"\n{'='*60}")
    print("  CROSS-REFERENCE SUMMARY")
    print(f"{'='*60}")
    
    import pyarrow.parquet as pq
    import numpy as np
    
    train_pf = pq.ParquetFile(DATA_RAW / "train-00000-of-00005.parquet")
    test_pf = pq.ParquetFile(DATA_RAW / "test-00000-of-00001.parquet")
    
    train_rows = train_pf.metadata.num_rows
    test_rows = test_pf.metadata.num_rows
    
    tv_arr = np.load(DATA_RAW / "times_trainval.npy", allow_pickle=True)
    te_arr = np.load(DATA_RAW / "times_test.npy", allow_pickle=True)
    
    print(f"  Train parquet rows:       {train_rows}")
    print(f"  Test parquet rows:        {test_rows}")
    print(f"  Total parquet rows:       {train_rows + test_rows}")
    print(f"  times_trainval length:    {len(tv_arr)}")
    print(f"  times_test length:        {len(te_arr)}")
    print(f"  Total npy timestamps:     {len(tv_arr) + len(te_arr)}")
    print(f"  Train parquet == npy?:    {train_rows == len(tv_arr)}")
    print(f"  Test parquet == npy?:     {test_rows == len(te_arr)}")
    
    # Check overlap between trainval and test timestamps
    import pandas as pd
    tv_ts = pd.to_datetime(tv_arr)
    te_ts = pd.to_datetime(te_arr)
    overlap = set(tv_ts) & set(te_ts)
    print(f"  Timestamp overlap (trainval ∩ test): {len(overlap)}")
    if overlap:
        print(f"    WARNING: Overlapping timestamps detected!")
        sorted_overlap = sorted(overlap)
        print(f"    First 5 overlapping: {sorted_overlap[:5]}")
    
    del tv_arr, te_arr
