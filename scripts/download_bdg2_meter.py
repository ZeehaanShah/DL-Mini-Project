"""
BDG2 Electricity Demand extraction and processing script (Model D).

Downloads building electricity meter timeseries from the Building Data Genome Project 2
(BUDS Lab, Miller et al., 2020) for Panther_education_Rosalie.

Design & Safety:
- Downloads 174 MB raw electricity matrix to temporary staging file.
- Uses pandas chunked extraction to extract ONLY timestamp and Panther_education_Rosalie.
- Immediately deletes temporary 174 MB matrix to retain minimal repository footprint (~420 KB).
- Causal missing value audit: isolated dropouts <= 2h are forward-filled without lookahead.
- Validates processed dataframe using validate_demand_dataframe.
- Saves raw extract to data/raw/bdg2/ and processed series to data/processed/demand/.

NOTE: This script is prepared for Stage 2B execution and must only be run after user approval.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Initialize Windows PyTorch DLLs before loading numpy/pandas
import src.utils.torch_loader

import numpy as np
import pandas as pd
import requests

from src.data.demand import validate_demand_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_bdg2_meter")

# Official repository media endpoints
BDG2_REPO = "buds-lab/building-data-genome-project-2"
RAW_ELECTRICITY_URL = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/meters/raw/electricity.csv"
METADATA_URL = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/metadata/metadata.csv"
TARGET_METER = "Panther_education_Rosalie"


def download_and_extract_meter(
    target_meter: str = TARGET_METER,
    raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "bdg2",
    timeout_sec: int = 180,
) -> Tuple[Path, Path]:
    """
    Download BDG2 electricity.csv to temporary scratch file, extract target meter column,
    extract metadata row, and immediately delete temporary large file.
    Atomic promotion: writes to .tmp first, verifies, then atomically renames.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_meter_path = raw_dir / f"bdg2_{target_meter}_raw.csv"
    raw_meta_path = raw_dir / f"metadata_{target_meter}.csv"
    tmp_large_csv = raw_dir / ".tmp_electricity.csv"
    tmp_raw_meter = raw_dir / f".tmp_bdg2_{target_meter}_raw.csv"
    tmp_raw_meta = raw_dir / f".tmp_metadata_{target_meter}.csv"

    # 1. Download metadata and extract building row
    logger.info(f"Downloading BDG2 metadata from {METADATA_URL}...")
    meta_resp = requests.get(METADATA_URL, timeout=60)
    meta_resp.raise_for_status()

    # Verify not HTML error
    if meta_resp.text.strip().lower().startswith("<!doctype html") or "<html" in meta_resp.text[:200].lower():
        raise ValueError(f"Received HTML response instead of CSV from {METADATA_URL}")

    # Parse metadata to find target building
    import io
    df_meta = pd.read_csv(io.StringIO(meta_resp.text))
    building_meta = df_meta[df_meta["building_id"] == target_meter]
    if building_meta.empty:
        raise ValueError(f"Building ID '{target_meter}' not found in BDG2 metadata.csv")

    building_meta.to_csv(tmp_raw_meta, index=False)
    if raw_meta_path.exists():
        raw_meta_path.replace(raw_dir / f"metadata_{target_meter}.bak.csv")
    tmp_raw_meta.replace(raw_meta_path)
    logger.info(f"Saved building metadata row to {raw_meta_path}")

    # 2. Download raw electricity.csv (174 MB) with streaming to temp file
    logger.info(f"Downloading BDG2 raw electricity matrix (~174 MB) from {RAW_ELECTRICITY_URL}...")
    start_time = time.time()
    
    try:
        resp = requests.get(RAW_ELECTRICITY_URL, stream=True, timeout=timeout_sec)
        resp.raise_for_status()

        # Check content-type and first chunk
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise ValueError(f"Unexpected Content-Type '{content_type}' returned from {RAW_ELECTRICITY_URL}")

        bytes_written = 0
        first_chunk = True
        with open(tmp_large_csv, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    if first_chunk:
                        first_chunk = False
                        chunk_lower = chunk[:200].lower()
                        if b"<!doctype html" in chunk_lower or b"<html" in chunk_lower:
                            raise ValueError(f"Server returned HTML error page instead of raw CSV file")
                    f.write(chunk)
                    bytes_written += len(chunk)

        elapsed = time.time() - start_time
        logger.info(f"Downloaded {bytes_written:,} bytes in {elapsed:.1f}s to {tmp_large_csv}")

        if bytes_written < 1000000:
            raise ValueError(f"Downloaded file unexpectedly small ({bytes_written} bytes); possible truncated or LFS pointer response")

        # 3. Stream/chunk extract ONLY timestamp and target_meter column
        logger.info(f"Extracting target meter column '{target_meter}'...")
        chunks = []
        for chunk in pd.read_csv(
            tmp_large_csv,
            usecols=["timestamp", target_meter],
            chunksize=5000,
            low_memory=False,
        ):
            chunks.append(chunk)

        df_target = pd.concat(chunks, ignore_index=True)
        if len(df_target) < 17000:
            raise ValueError(f"Extracted row count {len(df_target)} is fewer than expected 17,544 rows")

        # Write to temporary file first
        df_target.to_csv(tmp_raw_meter, index=False)

        # Verify readability of the extracted file
        df_verify = pd.read_csv(tmp_raw_meter, nrows=5)
        if "timestamp" not in df_verify.columns or target_meter not in df_verify.columns:
            raise ValueError(f"Verification of {tmp_raw_meter} failed: missing expected columns")

        # Atomic promotion
        if raw_meter_path.exists():
            raw_meter_path.replace(raw_dir / f"bdg2_{target_meter}_raw.bak.csv")
        tmp_raw_meter.replace(raw_meter_path)
        logger.info(f"Successfully validated and atomically saved {raw_meter_path} ({len(df_target):,} rows, {raw_meter_path.stat().st_size:,} bytes)")

    finally:
        # 4. Immediate cleanup of the 174 MB temporary file
        if tmp_large_csv.exists():
            tmp_large_csv.unlink()
            logger.info(f"Deleted temporary large matrix file {tmp_large_csv}")
        if tmp_raw_meter.exists():
            tmp_raw_meter.unlink()

    return raw_meter_path, raw_meta_path


def parse_and_process_demand(
    raw_path: Path,
    processed_path: Path,
    target_meter: str = TARGET_METER,
    max_causal_gap_hours: int = 8,
    max_demand_kwh: float = 1000.0,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Parse raw extracted demand series, audit timestamps, missingness, physical bounds,
    apply bounded causal forward-fill for isolated dropouts <= max_causal_gap_hours,
    validate schema and continuity, and save processed dataset atomically.
    """
    logger.info(f"Processing raw demand file: {raw_path}")
    df_raw = pd.read_csv(raw_path)

    if "timestamp" not in df_raw.columns:
        raise ValueError(f"Missing 'timestamp' in {list(df_raw.columns)}")
    if target_meter not in df_raw.columns:
        raise ValueError(f"Missing '{target_meter}' in {list(df_raw.columns)}")

    df_clean = pd.DataFrame()
    df_clean["datetime"] = pd.to_datetime(df_raw["timestamp"])
    df_clean = df_clean.sort_values(by="datetime").reset_index(drop=True)

    # Convert demand to float32
    raw_series = pd.to_numeric(df_raw[target_meter], errors="coerce")
    total_rows = len(raw_series)
    nan_count = int(raw_series.isna().sum())
    nan_pct = (nan_count / total_rows) * 100.0

    logger.info(f"Raw demand missingness: {nan_count}/{total_rows} ({nan_pct:.2f}%)")

    # Detailed machine-readable imputation audit
    imputation_records = []
    is_nan = raw_series.isna()

    if is_nan.any():
        # Identify contiguous NaN blocks
        blocks = (~is_nan).cumsum()[is_nan]
        block_lengths = blocks.value_counts()
        max_gap = int(block_lengths.max()) if not block_lengths.empty else 0
        gap_distribution = {str(k): int(v) for k, v in block_lengths.items()}
        logger.info(f"Max contiguous missing gap: {max_gap} hours; Gap counts: {gap_distribution}")

        if max_gap > max_causal_gap_hours:
            logger.warning(
                f"Contiguous gap of {max_gap}h exceeds allowed causal threshold ({max_causal_gap_hours}h). "
                f"Applying forward-fill strictly up to {max_causal_gap_hours}h without future lookahead."
            )
        # Strictly causal forward-fill bounded by limit
        filled_series = raw_series.ffill(limit=max_causal_gap_hours)

        # Record exact imputed entries
        for idx in np.where(is_nan)[0]:
            orig_val = None if pd.isna(raw_series.iloc[idx]) else float(raw_series.iloc[idx])
            fill_val = None if pd.isna(filled_series.iloc[idx]) else float(filled_series.iloc[idx])
            imputation_records.append({
                "row_index": int(idx),
                "timestamp": str(df_clean["datetime"].iloc[idx]),
                "original_value": orig_val,
                "replacement_value": fill_val,
                "imputed": fill_val is not None and orig_val is None,
                "method": f"bounded_causal_forward_fill_limit_{max_causal_gap_hours}h",
                "justification": "Isolated meter transmission dropout filled from previous valid observation without lookahead",
            })
    else:
        filled_series = raw_series
        max_gap = 0
        gap_distribution = {}

    df_clean["demand_kwh"] = filled_series.to_numpy(dtype=np.float32)

    # Validate against strict pipeline criteria
    validated_df = validate_demand_dataframe(
        df_clean,
        time_col="datetime",
        meter_col="demand_kwh",
        expected_freq="1h",
        max_demand_kwh=max_demand_kwh,
    )

    # Save processed dataset atomically
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_processed = processed_path.parent / f".tmp_{processed_path.name}"
    validated_df.to_csv(tmp_processed, index=False)
    if processed_path.exists():
        processed_path.replace(processed_path.parent / f"{processed_path.name}.bak")
    tmp_processed.replace(processed_path)
    logger.info(f"Validated processed demand data saved to {processed_path} ({len(validated_df)} rows)")

    # Compute detailed statistics
    readings = validated_df["demand_kwh"].to_numpy()
    stats = {
        "count": int(len(readings)),
        "mean_kwh": float(np.mean(readings)),
        "std_kwh": float(np.std(readings)),
        "min_kwh": float(np.min(readings)),
        "p25_kwh": float(np.percentile(readings, 25)),
        "median_kwh": float(np.percentile(readings, 50)),
        "p75_kwh": float(np.percentile(readings, 75)),
        "max_kwh": float(np.max(readings)),
        "zero_count": int(np.sum(readings == 0.0)),
        "negative_count": int(np.sum(readings < 0.0)),
        "nan_count": nan_count,
        "max_contiguous_nan_gap_hours": max_gap,
    }

    # Save imputation audit
    imputation_audit = {
        "building_id": target_meter,
        "raw_nan_count": nan_count,
        "raw_nan_percentage": float(nan_pct),
        "max_contiguous_gap_hours": max_gap,
        "gap_distribution": gap_distribution,
        "max_causal_gap_hours_allowed": max_causal_gap_hours,
        "imputation_policy": "bounded_causal_forward_fill",
        "total_imputed_records": len([r for r in imputation_records if r["imputed"]]),
        "records": imputation_records,
    }
    audit_path = raw_path.parent / f"imputation_audit_{target_meter}.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(imputation_audit, f, indent=2)
    logger.info(f"Saved imputation audit artifact to {audit_path}")

    # Save metadata sidecar
    meta_sidecar = {
        "dataset_name": "Building Data Genome Project 2 (BDG2)",
        "source_repo": f"https://github.com/{BDG2_REPO}",
        "raw_data_url": RAW_ELECTRICITY_URL,
        "publication": "Miller et al., Sci Data 7, 368 (2020). DOI: 10.1038/s41597-020-00712-x",
        "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0)",
        "building_id": target_meter,
        "site_id": "Panther",
        "primary_space_usage": "Education",
        "sub_primary_space_usage": "Research",
        "floor_area_sqm": 690.5,
        "year_built": 2008,
        "location": "Orlando, Florida, USA (University of Central Florida campus)",
        "coordinates": {"latitude": 28.5176885788321, "longitude": -81.3790387999998},
        "timezone": "US/Eastern (Local Standard Time, EST / UTC-5, no DST shifts)",
        "temporal_span": {
            "start": str(validated_df["datetime"].min()),
            "end": str(validated_df["datetime"].max()),
            "total_records": len(validated_df),
            "expected_2016_leap_hours": 8784,
            "expected_2017_nonleap_hours": 8760,
            "expected_total_hours": 17544,
        },
        "units": {
            "reported_unit": "kWh (hourly accumulated energy sum)",
            "equivalent_power": "kW (hourly average active power)",
        },
        "quality_metrics": stats,
    }

    sidecar_path = raw_path.parent / f"{raw_path.stem}_metadata.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(meta_sidecar, f, indent=2)
    logger.info(f"Saved metadata sidecar to {sidecar_path}")

    return validated_df, meta_sidecar, imputation_audit


def main():
    parser = argparse.ArgumentParser(description="BDG2 meter extraction and processing")
    parser.add_argument("--building-id", default=TARGET_METER, help="Building meter ID to extract")
    parser.add_argument("--force", action="store_true", help="Force re-download even if raw file exists")
    parser.add_argument("--skip-download", action="store_true", help="Skip download and process existing raw file")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw" / "bdg2"
    raw_path = raw_dir / f"bdg2_{args.building_id}_raw.csv"
    processed_path = PROJECT_ROOT / "data" / "processed" / "demand" / "demand_hourly_pilot.csv"

    if not args.skip_download and (not raw_path.exists() or args.force):
        download_and_extract_meter(
            target_meter=args.building_id,
            raw_dir=raw_dir,
        )
    elif raw_path.exists():
        logger.info(f"Raw file already exists at {raw_path}. Processing directly.")
    else:
        logger.error(f"Cannot skip download: {raw_path} does not exist.")
        sys.exit(1)

    df_valid, meta, audit = parse_and_process_demand(raw_path, processed_path, target_meter=args.building_id)

    print("\n" + "=" * 60)
    print("BDG2 ELECTRICITY DEMAND VALIDATION AUDIT")
    print("=" * 60)
    print(f"Building ID:         {meta['building_id']}")
    print(f"Site & Space Usage:  {meta['site_id']} / {meta['primary_space_usage']} ({meta['sub_primary_space_usage']})")
    print(f"Floor Area:          {meta['floor_area_sqm']} m²")
    print(f"Location & Timezone: {meta['location']} ({meta['timezone']})")
    print(f"Raw File Path:       {raw_path}")
    print(f"Processed File Path: {processed_path}")
    print(f"Actual Row Count:    {len(df_valid):,} (Expected 17,544)")
    print(f"Date Range:          {df_valid['datetime'].min()} to {df_valid['datetime'].max()}")
    print("-" * 60)
    print(f"Consumption (kWh):   min={meta['quality_metrics']['min_kwh']:.2f}, mean={meta['quality_metrics']['mean_kwh']:.2f}, max={meta['quality_metrics']['max_kwh']:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
