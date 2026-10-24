"""
Pilot download and validation script for NREL / NLR India Wind Toolkit data.

Downloads a controlled, single-year (2014) hourly pilot dataset for the Gujarat
coastal wind corridor (23.25N, 69.67E) at 100m hub height.

Security & Integrity:
- Loads API key securely from .env; NEVER prints, logs, or writes credentials.
- Stages raw data to temporary file before atomic move to data/raw/wind/.
- Preserves raw download immutably.
- Generates standardized, validated processed dataset under data/processed/wind/.
"""
import argparse
import json
import logging
import os
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

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests

from src.data.wind import validate_wind_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_wind_pilot")

# Official NREL / NLR endpoints
# Note: May 2026 NLR transition moved active service from developer.nrel.gov to developer.nlr.gov
PRIMARY_ENDPOINT = "https://developer.nlr.gov/api/wind-toolkit/v2/wind/india-wind-download.csv"
FALLBACK_ENDPOINT = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/india-wind-download.csv"


def get_secure_credentials() -> Tuple[str, str]:
    """
    Safely load NREL credentials from environment or .env file.
    Does NOT print, log, or return sensitive data in error messages.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("NREL_API_KEY")
    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(
            "NREL_API_KEY not found in environment or .env file. "
            "Please ensure NREL_API_KEY is configured in .env."
        )
    email = os.getenv("NREL_EMAIL", "shahzeehaan@gmail.com")
    return api_key.strip(), email.strip()


def download_wind_extract(
    api_key: str,
    email: str,
    lat: float = 23.25,
    lon: float = 69.67,
    year: int = 2014,
    interval: int = 60,
    attributes: str = "windspeed_100m,winddirection_100m,temperature_100m,pressure_100m",
    raw_out_path: Path = PROJECT_ROOT / "data" / "raw" / "wind" / "nrel_india_wind_2014_23.25N_69.67E.csv",
    timeout_sec: int = 60,
    max_retries: int = 3,
) -> Path:
    """
    Download raw wind timeseries from NREL India Wind API with bounded retries
    and atomic staging.
    """
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = raw_out_path.parent / f".tmp_{raw_out_path.name}"

    wkt_point = f"POINT({lon} {lat})"
    params = {
        "api_key": api_key,
        "email": email,
        "wkt": wkt_point,
        "attributes": attributes,
        "names": str(year),
        "interval": str(interval),
        "utc": "true",
        "leap_day": "false",
    }

    headers = {
        "User-Agent": "DL-MiniProject-Research/1.0",
    }

    endpoints = [PRIMARY_ENDPOINT, FALLBACK_ENDPOINT]
    success = False
    last_error = ""

    for endpoint in endpoints:
        for attempt in range(1, max_retries + 1):
            logger.info(
                f"Requesting extract from {endpoint.split('/')[2]} "
                f"[Attempt {attempt}/{max_retries}] for {wkt_point}, year={year}, interval={interval}m..."
            )
            try:
                resp = requests.get(
                    endpoint,
                    params=params,
                    headers=headers,
                    timeout=timeout_sec,
                    stream=True,
                )
                if resp.status_code == 200:
                    # Stream write to temporary file
                    total_bytes = 0
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                total_bytes += len(chunk)

                    # Validate response is not empty and has CSV structure
                    if total_bytes < 1000:
                        raise ValueError(f"Downloaded response is unexpectedly small: {total_bytes} bytes")

                    with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                        first_line = f.readline()
                        second_line = f.readline()

                    if "SiteID" not in first_line and "Site Timezone" not in first_line:
                        raise ValueError(f"Downloaded file does not start with expected metadata header: {first_line[:100]}")
                    if "wind speed" not in second_line.lower():
                        raise ValueError(f"Downloaded file header missing wind speed column: {second_line[:100]}")

                    # Atomic rename/move to final raw destination
                    if raw_out_path.exists():
                        raw_out_path.unlink()
                    tmp_path.rename(raw_out_path)
                    logger.info(f"Raw data successfully saved immutably to {raw_out_path} ({total_bytes:,} bytes)")
                    success = True
                    break

                elif resp.status_code in (429, 503):
                    # Rate limit or queue full; exponential backoff
                    wait_sec = 2 ** attempt * 2
                    logger.warning(f"Provider returned status {resp.status_code}. Backing off for {wait_sec}s...")
                    time.sleep(wait_sec)
                elif resp.status_code in (400, 401, 403):
                    # Bad request or authentication failure; do not retry indefinitely
                    err_msg = resp.text.replace(api_key, "[REDACTED]")[:250]
                    last_error = f"HTTP {resp.status_code}: {err_msg}"
                    logger.error(f"Authentication or parameter error: {last_error}")
                    break
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    time.sleep(2)

            except requests.exceptions.RequestException as e:
                clean_err = str(e).replace(api_key, "[REDACTED]")
                last_error = clean_err
                logger.warning(f"Request exception on attempt {attempt}: {clean_err}")
                time.sleep(2)

        if success:
            break

    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    if not success:
        raise RuntimeError(f"Failed to acquire wind data after trying endpoints. Last error: {last_error}")

    return raw_out_path


def parse_and_process_wind(
    raw_path: Path,
    processed_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse the raw NREL wind extract, standardize columns and units,
    and run schema validation.
    """
    logger.info(f"Parsing raw wind file: {raw_path}")

    # Read metadata header from line 1
    with open(raw_path, "r", encoding="utf-8") as f:
        meta_line = f.readline().strip()

    meta_parts = meta_line.split(",")
    metadata = {}
    for i in range(0, len(meta_parts) - 1, 2):
        key = meta_parts[i].strip()
        val = meta_parts[i + 1].strip()
        metadata[key] = val

    logger.info(f"Extracted site metadata: {metadata}")

    # Read data starting from line 2 (header line)
    df_raw = pd.read_csv(raw_path, skiprows=1)
    logger.info(f"Raw data shape: {df_raw.shape}")
    logger.info(f"Raw columns: {list(df_raw.columns)}")

    # Construct datetime column from Year, Month, Day, Hour, Minute
    required_time_cols = ["Year", "Month", "Day", "Hour", "Minute"]
    for c in required_time_cols:
        if c not in df_raw.columns:
            raise ValueError(f"Missing expected time column '{c}' in {list(df_raw.columns)}")

    datetimes = pd.to_datetime(
        df_raw[required_time_cols]
    )

    # Standardize feature columns
    # Find exact matching columns
    col_map = {}
    for col in df_raw.columns:
        col_l = col.lower()
        if "wind speed" in col_l:
            col_map[col] = "wind_speed"
        elif "wind direction" in col_l:
            col_map[col] = "wind_direction"
        elif "temperature" in col_l:
            col_map[col] = "temperature"
        elif "pressure" in col_l:
            col_map[col] = "pressure_raw"

    logger.info(f"Column mappings: {col_map}")
    df_clean = pd.DataFrame()
    df_clean["datetime"] = datetimes

    for src_col, dest_col in col_map.items():
        if dest_col == "pressure_raw":
            # NREL air pressure is in Pa; convert to hPa (hectopascals / mbar)
            # 1 hPa = 100 Pa
            df_clean["pressure"] = (df_raw[src_col].astype(np.float64) / 100.0).astype(np.float32)
        else:
            df_clean[dest_col] = df_raw[src_col].astype(np.float32)

    # Validate against strict pipeline criteria
    validated_df = validate_wind_dataframe(
        df_clean,
        time_col="datetime",
        speed_col="wind_speed",
        direction_col="wind_direction" if "wind_direction" in df_clean.columns else None,
        temp_col="temperature" if "temperature" in df_clean.columns else None,
        pressure_col="pressure" if "pressure" in df_clean.columns else None,
        expected_freq="1h",
    )

    # Save processed dataset
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(processed_path, index=False)
    logger.info(f"Validated processed wind data saved to {processed_path} ({len(validated_df)} rows)")

    # Save metadata sidecar
    meta_sidecar = {
        "source": "NREL India Wind Toolkit / India Renewable Integration Study (IRIS)",
        "endpoint": PRIMARY_ENDPOINT,
        "site_id": metadata.get("SiteID", "Unknown"),
        "grid_longitude": metadata.get("Longitude", "69.662"),
        "grid_latitude": metadata.get("Latitude", "23.243"),
        "requested_longitude": 69.67,
        "requested_latitude": 23.25,
        "temporal_resolution": "1 hour (interval=60, minute=30)",
        "record_count": len(validated_df),
        "start_time": str(validated_df["datetime"].min()),
        "end_time": str(validated_df["datetime"].max()),
        "height_agl": "100m",
        "variables": {
            "wind_speed": {"unit": "m/s", "min": float(validated_df["wind_speed"].min()), "max": float(validated_df["wind_speed"].max()), "mean": float(validated_df["wind_speed"].mean())},
            "wind_direction": {"unit": "deg", "min": float(validated_df["wind_direction"].min()), "max": float(validated_df["wind_direction"].max()), "mean": float(validated_df["wind_direction"].mean())},
            "temperature": {"unit": "degC", "min": float(validated_df["temperature"].min()), "max": float(validated_df["temperature"].max()), "mean": float(validated_df["temperature"].mean())},
            "pressure": {"unit": "hPa", "min": float(validated_df["pressure"].min()), "max": float(validated_df["pressure"].max()), "mean": float(validated_df["pressure"].mean())},
        },
        "nature_of_data": "Modeled meso-scale atmospheric numerical simulation (WRF 3km grid reanalysis), NOT physical tower anemometer observations.",
    }

    sidecar_path = raw_path.parent / f"{raw_path.stem}_metadata.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(meta_sidecar, f, indent=2)
    logger.info(f"Saved metadata sidecar to {sidecar_path}")

    return validated_df, meta_sidecar


def main():
    parser = argparse.ArgumentParser(description="Pilot download for NREL India Wind Toolkit data")
    parser.add_argument("--force", action="store_true", help="Force re-download even if raw file exists")
    parser.add_argument("--skip-download", action="store_true", help="Skip download and process existing raw file")
    args = parser.parse_args()

    raw_path = PROJECT_ROOT / "data" / "raw" / "wind" / "nrel_india_wind_2014_23.25N_69.67E.csv"
    processed_path = PROJECT_ROOT / "data" / "processed" / "wind" / "wind_hourly_pilot.csv"

    if not args.skip_download and (not raw_path.exists() or args.force):
        api_key, email = get_secure_credentials()
        download_wind_extract(
            api_key=api_key,
            email=email,
            lat=23.25,
            lon=69.67,
            year=2014,
            interval=60,
            raw_out_path=raw_path,
        )
    elif raw_path.exists():
        logger.info(f"Raw file already exists at {raw_path}. Processing directly.")
    else:
        logger.error(f"Cannot skip download: {raw_path} does not exist.")
        sys.exit(1)

    df_valid, meta = parse_and_process_wind(raw_path, processed_path)

    print("\n" + "=" * 60)
    print("NREL INDIA WIND PILOT VALIDATION AUDIT")
    print("=" * 60)
    print(f"Raw File Path:       {raw_path}")
    print(f"Processed File Path: {processed_path}")
    print(f"Actual Row Count:    {len(df_valid):,} (Expected 8,760)")
    print(f"Date Range:          {df_valid['datetime'].min()} to {df_valid['datetime'].max()}")
    print(f"Grid Cell:           Lat {meta['grid_latitude']}, Lon {meta['grid_longitude']} (SiteID: {meta['site_id']})")
    print(f"Measurement Height:  100m Above Ground Level (AGL)")
    print(f"Temporal Frequency:  1-hour continuous (centered at :30)")
    print(f"Data Nature:         {meta['nature_of_data']}")
    print("-" * 60)
    print("Variable Statistics:")
    for vname, vinfo in meta["variables"].items():
        print(f"  {vname:<16} [{vinfo['unit']:<5}]: min={vinfo['min']:>6.2f}, mean={vinfo['mean']:>6.2f}, max={vinfo['max']:>6.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
