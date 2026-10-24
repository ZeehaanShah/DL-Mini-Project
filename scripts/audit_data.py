"""
Data audit script for DL MiniProject.
Inspects all datasets with bounded memory usage.
Outputs structured findings to stdout.
"""
import os
import sys
import json
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

# =====================================================================
#  1. SKIPP'D PARQUET FILES
# =====================================================================
def audit_skippd():
    section("SKIPP'D DATASET AUDIT")
    skippd_dir = DATA_RAW / "skippd"
    
    parquet_files = {
        "train": skippd_dir / "train-00000-of-00005.parquet",
        "test": skippd_dir / "test-00000-of-00001.parquet",
    }
    
    for label, fpath in parquet_files.items():
        print(f"--- {label} Parquet: {fpath.name} ---")
        if not fpath.exists():
            print(f"  FILE NOT FOUND: {fpath}")
            continue
        
        print(f"  File size: {fpath.stat().st_size / (1024**2):.2f} MB")
        
        try:
            import pyarrow.parquet as pq
            
            # Read metadata only first
            pf = pq.ParquetFile(fpath)
            meta = pf.metadata
            schema = pf.schema_arrow
            
            print(f"  Parquet metadata:")
            print(f"    Num rows: {meta.num_rows}")
            print(f"    Num columns: {meta.num_columns}")
            print(f"    Num row groups: {meta.num_row_groups}")
            print(f"    Created by: {meta.created_by}")
            print(f"    Format version: {meta.format_version}")
            
            print(f"  Schema ({len(schema)} fields):")
            for i in range(len(schema)):
                field = schema.field(i)
                print(f"    [{i}] {field.name}: {field.type}")
                if hasattr(field.type, 'value_type'):
                    print(f"         (nested value_type: {field.type.value_type})")
            
            # Row group sizes
            for rg_idx in range(min(meta.num_row_groups, 3)):
                rg = meta.row_group(rg_idx)
                print(f"  Row group {rg_idx}: {rg.num_rows} rows, {rg.total_byte_size / (1024**2):.1f} MB")
            
            # Read a small sample (first 5 rows)
            print(f"\n  Sampling first 5 rows...")
            import pandas as pd
            table = pf.read_row_group(0, columns=None)
            df_sample = table.to_pandas().head(5)
            
            print(f"  Columns: {list(df_sample.columns)}")
            print(f"  Dtypes:")
            for col in df_sample.columns:
                dtype = df_sample[col].dtype
                print(f"    {col}: {dtype}")
                # Check for nested/image data
                if dtype == object:
                    first_val = df_sample[col].iloc[0]
                    val_type = type(first_val).__name__
                    if isinstance(first_val, dict):
                        print(f"      Dict keys: {list(first_val.keys())}")
                        for k, v in first_val.items():
                            if isinstance(v, bytes):
                                print(f"        {k}: bytes, len={len(v)}")
                            elif isinstance(v, (list, dict)):
                                print(f"        {k}: {type(v).__name__}, len={len(v) if hasattr(v,'__len__') else 'N/A'}")
                            else:
                                print(f"        {k}: {type(v).__name__} = {repr(v)[:100]}")
                    elif isinstance(first_val, bytes):
                        print(f"      bytes, len={len(first_val)}")
                    elif isinstance(first_val, list):
                        print(f"      list, len={len(first_val)}, elem_type={type(first_val[0]).__name__ if first_val else 'empty'}")
                    else:
                        print(f"      type={val_type}, sample={repr(first_val)[:200]}")
                else:
                    # Numeric stats
                    try:
                        col_data = df_sample[col]
                        print(f"      sample values: {col_data.values[:5].tolist()}")
                        if col_data.dtype in ['float64', 'float32', 'int64', 'int32']:
                            print(f"      min={col_data.min()}, max={col_data.max()}")
                    except Exception:
                        pass
            
            # Check for NaN/null in full first row group
            df_rg0 = table.to_pandas()
            print(f"\n  Row group 0 ({len(df_rg0)} rows) null counts:")
            for col in df_rg0.columns:
                null_count = df_rg0[col].isna().sum()
                if null_count > 0:
                    print(f"    {col}: {null_count} nulls ({100*null_count/len(df_rg0):.1f}%)")
            if df_rg0.isna().sum().sum() == 0:
                print(f"    No nulls in row group 0")
                
            # Try to detect image data
            print(f"\n  Attempting image inspection...")
            for col in df_sample.columns:
                first_val = df_sample[col].iloc[0]
                if isinstance(first_val, dict) and 'bytes' in first_val:
                    try:
                        from PIL import Image
                        import io
                        img_bytes = first_val['bytes']
                        img = Image.open(io.BytesIO(img_bytes))
                        print(f"    {col}: Image decoded! size={img.size}, mode={img.mode}, format={img.format}")
                    except Exception as e:
                        print(f"    {col}: Image decode failed: {e}")
                elif isinstance(first_val, bytes):
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(first_val))
                        print(f"    {col}: Image decoded! size={img.size}, mode={img.mode}, format={img.format}")
                    except Exception as e:
                        print(f"    {col}: Bytes but not a decodable image: {e}")
            
            del df_rg0, table, df_sample
            
        except Exception as e:
            print(f"  ERROR reading parquet: {e}")
            traceback.print_exc()
        
        print()

    # Numpy timestamp files
    print("--- NumPy timestamp arrays ---")
    npy_files = {
        "times_trainval": skippd_dir / "times_trainval.npy",
        "times_test": skippd_dir / "times_test.npy",
    }
    
    for label, fpath in npy_files.items():
        print(f"\n  {label}: {fpath.name}")
        if not fpath.exists():
            print(f"    FILE NOT FOUND")
            continue
        
        print(f"    File size: {fpath.stat().st_size / (1024**2):.2f} MB")
        
        try:
            import numpy as np
            arr = np.load(fpath, allow_pickle=True)
            print(f"    Shape: {arr.shape}")
            print(f"    Dtype: {arr.dtype}")
            print(f"    First 5 values: {arr[:5]}")
            print(f"    Last 5 values: {arr[-5:]}")
            print(f"    Min: {arr.min()}")
            print(f"    Max: {arr.max()}")
            print(f"    Unique count: {len(np.unique(arr))}")
            print(f"    Has NaN: {np.isnan(arr).any() if np.issubdtype(arr.dtype, np.floating) else 'N/A (integer type)'}")
            
            # Attempt timestamp interpretation
            if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
                try:
                    import pandas as pd
                    # Try Unix timestamp (seconds)
                    ts_seconds = pd.to_datetime(arr[:3], unit='s', utc=True)
                    print(f"    As Unix seconds: {ts_seconds.tolist()}")
                except:
                    pass
                try:
                    # Try Unix timestamp (milliseconds)
                    ts_ms = pd.to_datetime(arr[:3], unit='ms', utc=True)
                    print(f"    As Unix ms: {ts_ms.tolist()}")
                except:
                    pass
            
            del arr
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()


# =====================================================================
#  2. NASA POWER CSV
# =====================================================================
def audit_nasa_power():
    section("NASA POWER DATASET AUDIT")
    nasa_dir = DATA_RAW / "nasa_power"
    csv_file = nasa_dir / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"
    
    if not csv_file.exists():
        print(f"  FILE NOT FOUND: {csv_file}")
        return
    
    print(f"  File: {csv_file.name}")
    print(f"  File size: {csv_file.stat().st_size / 1024:.1f} KB")
    
    # Read the first 30 lines to find header/metadata
    print(f"\n  First 30 lines (raw):")
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line.rstrip('\n'))
            if i < 30:
                print(f"    [{i:3d}] {line.rstrip()}")
    
    total_lines = len(lines)
    print(f"\n  Total lines in file: {total_lines}")
    
    # Find the header row (NASA POWER CSVs have metadata rows starting with -)
    header_row = None
    for i, line in enumerate(lines):
        # The header is typically the first row that contains YEAR,MO,DY or similar
        if 'YEAR' in line and ('MO' in line or 'DOY' in line):
            header_row = i
            break
    
    if header_row is not None:
        print(f"\n  Detected header at line {header_row}: {lines[header_row]}")
        print(f"  Metadata lines (before header): {header_row}")
        print(f"  Metadata content:")
        for i in range(header_row):
            print(f"    {lines[i]}")
    else:
        print(f"  WARNING: Could not auto-detect header row")
        header_row = 0  # fallback
    
    # Parse with pandas
    try:
        import pandas as pd
        df = pd.read_csv(csv_file, skiprows=header_row)
        
        print(f"\n  Parsed DataFrame:")
        print(f"    Rows: {len(df)}")
        print(f"    Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"    Dtypes:")
        for col in df.columns:
            print(f"      {col}: {df[col].dtype}")
        
        print(f"\n  First 3 rows:")
        print(df.head(3).to_string(index=False))
        
        print(f"\n  Last 3 rows:")
        print(df.tail(3).to_string(index=False))
        
        print(f"\n  Summary statistics:")
        print(df.describe().to_string())
        
        # Check for missing values
        print(f"\n  Missing/sentinel value check:")
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                print(f"    {col}: {null_count} NaN ({100*null_count/len(df):.1f}%)")
            # Check for common sentinel values like -999
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                neg999 = (df[col] == -999).sum()
                neg9999 = (df[col] == -999.0).sum()
                neg99 = (df[col] == -99).sum()
                if neg999 > 0:
                    print(f"    {col}: {neg999} values == -999 (possible sentinel)")
                if neg99 > 0 and neg99 != neg999:
                    print(f"    {col}: {neg99} values == -99 (possible sentinel)")
        
        total_na = df.isna().sum().sum()
        if total_na == 0:
            print(f"    No NaN values detected")
        
        # Check date range
        date_cols = [c for c in df.columns if c.upper() in ['YEAR', 'MO', 'DY', 'HR']]
        if date_cols:
            print(f"\n  Date-related columns: {date_cols}")
            for c in date_cols:
                print(f"    {c}: min={df[c].min()}, max={df[c].max()}, unique={df[c].nunique()}")
        
        # Check for expected parameters
        expected_params = ['ALLSKY_SFC_SW_DWN', 'T2M', 'RH2M', 'WS10M']
        print(f"\n  Expected parameters check:")
        for param in expected_params:
            if param in df.columns:
                col = df[param]
                valid = col[col != -999]  # exclude sentinel
                print(f"    {param}: PRESENT, valid_count={len(valid)}, min={valid.min():.2f}, max={valid.max():.2f}, mean={valid.mean():.2f}")
            else:
                # Check case-insensitive
                matches = [c for c in df.columns if param.lower() in c.lower()]
                if matches:
                    print(f"    {param}: Not exact match, but similar columns: {matches}")
                else:
                    print(f"    {param}: NOT FOUND in columns")
        
        # Duplicate check
        dup_count = df.duplicated().sum()
        print(f"\n  Duplicate rows: {dup_count}")
        
        del df
        
    except Exception as e:
        print(f"  ERROR parsing CSV: {e}")
        traceback.print_exc()


# =====================================================================
#  3. WIND DATASET
# =====================================================================
def audit_wind():
    section("WIND DATASET AUDIT")
    wind_dir = DATA_RAW / "wind"
    
    if not wind_dir.exists():
        print("  Directory does not exist")
        return
    
    contents = list(wind_dir.iterdir())
    if not contents:
        print("  STATUS: EMPTY DIRECTORY - NO DATA DOWNLOADED")
        print("  Action required: Wind dataset must be downloaded before Model C can proceed.")
        return
    
    print(f"  Contents ({len(contents)} items):")
    for item in contents:
        if item.is_file():
            print(f"    FILE: {item.name} ({item.stat().st_size / (1024**2):.2f} MB)")
        else:
            print(f"    DIR:  {item.name}/")


# =====================================================================
#  4. BDG2 DEMAND DATASET
# =====================================================================
def audit_bdg2():
    section("BDG2 DEMAND DATASET AUDIT")
    bdg2_dir = DATA_RAW / "bdg2"
    
    if not bdg2_dir.exists():
        print("  Directory does not exist")
        return
    
    contents = list(bdg2_dir.iterdir())
    if not contents:
        print("  STATUS: EMPTY DIRECTORY - NO DATA DOWNLOADED")
        print("  Action required: BDG2 dataset must be downloaded before Model D can proceed.")
        return
    
    print(f"  Contents ({len(contents)} items):")
    for item in contents:
        if item.is_file():
            print(f"    FILE: {item.name} ({item.stat().st_size / (1024**2):.2f} MB)")
        else:
            print(f"    DIR:  {item.name}/")


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  DL MiniProject — Comprehensive Data Audit")
    print(f"  Run at: {__import__('datetime').datetime.now().isoformat()}")
    print("=" * 70)
    
    audit_skippd()
    audit_nasa_power()
    audit_wind()
    audit_bdg2()
    
    section("AUDIT COMPLETE")
    print("Review all findings above before proceeding to model training.")
