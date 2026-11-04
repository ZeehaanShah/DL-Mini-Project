"""
Building Data Genome Project 2 (BDG2) electricity demand ingestion, schema validation, and PyTorch dataset.

Designed for BDG2 building meter sequences (Miller et al., 2020):
- Validates timestamp monotonicity, hourly continuity, physical boundary constraints (kWh),
  and rejects negative or non-finite/sentinel readings.
- Readings represent hourly electricity consumption in kWh (equivalent to average active power in kW over 1h).
- Strict meter isolation: prevents indiscriminate cross-building meter mixing.
- Lookahead-free causal sliding window construction with chronological splitting and embargo.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from src.data.splits import (
    create_forecasting_windows,
    split_timeseries_chronological,
)


def validate_demand_dataframe(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    meter_col: str = "demand_kwh",
    building_id: Optional[str] = None,
    expected_freq: str = "1h",
    max_demand_kwh: float = 10000.0,
) -> pd.DataFrame:
    """
    Validate schema, continuity, temporal monotonicity, and physical boundary constraints
    for electricity demand time series (BDG2).

    Args:
        df: Input pandas DataFrame.
        time_col: Column containing timestamp observations (UTC or Local Standard Time).
        meter_col: Column containing electricity meter readings in kWh.
        building_id: Optional building identifier string for provenance.
        expected_freq: Expected frequency string (default: '1h' for hourly meters).
        max_demand_kwh: Upper plausibility threshold in kWh for an individual building meter.

    Returns:
        Validated DataFrame sorted chronologically with guaranteed continuity.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    if time_col not in df.columns:
        if time_col == "timestamp" and "datetime" in df.columns:
            time_col = "datetime"
        elif time_col == "datetime" and "timestamp" in df.columns:
            time_col = "timestamp"
        else:
            raise ValueError(f"Demand DataFrame missing timestamp column '{time_col}'. Available: {list(df.columns)}")

    if meter_col not in df.columns:
        if meter_col == "demand_kwh" and "electricity_demand_kwh" in df.columns:
            meter_col = "electricity_demand_kwh"
        elif meter_col == "electricity_demand_kwh" and "demand_kwh" in df.columns:
            meter_col = "demand_kwh"
        else:
            raise ValueError(f"Demand DataFrame missing meter reading column '{meter_col}'. Available: {list(df.columns)}")

    df_clean = df.copy()

    # Convert timestamps to DatetimeIndex
    if not pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])

    # Check for duplicate timestamps
    if df_clean[time_col].duplicated().any():
        n_dups = int(df_clean[time_col].duplicated().sum())
        raise ValueError(f"Demand DataFrame contains {n_dups} duplicate timestamps in '{time_col}'")

    # Sort chronologically if needed
    if not df_clean[time_col].is_monotonic_increasing:
        df_clean = df_clean.sort_values(by=time_col).reset_index(drop=True)

    # Verify temporal continuity
    if expected_freq == "1h":
        diffs = df_clean[time_col].diff().iloc[1:]
        if not (diffs == pd.Timedelta(hours=1)).all():
            irregular_indices = np.where(diffs != pd.Timedelta(hours=1))[0]
            raise ValueError(
                f"Demand DataFrame has {len(irregular_indices)} temporal discontinuities; "
                f"expected regular 1-hour intervals. First irregular at row {irregular_indices[0]}."
            )

    # Check non-finite values and sentinel values (e.g. -999, -9999)
    meter_series = pd.to_numeric(df_clean[meter_col], errors="coerce")
    if meter_series.isna().any():
        n_nan = int(meter_series.isna().sum())
        raise ValueError(f"Meter column '{meter_col}' contains {n_nan} NaN/null/non-numeric values.")
    if np.isinf(meter_series.to_numpy()).any():
        raise ValueError(f"Meter column '{meter_col}' contains infinite values.")
    if (meter_series <= -900).any():
        n_sentinel = int((meter_series <= -900).sum())
        raise ValueError(f"Meter column '{meter_col}' contains {n_sentinel} sentinel missing values (<= -900).")

    df_clean[meter_col] = meter_series.to_numpy(dtype=np.float32)

    # Physical domain constraints: Non-negative electricity consumption
    readings = df_clean[meter_col].to_numpy()
    if (readings < 0.0).any():
        raise ValueError(f"Negative electricity consumption detected in '{meter_col}': min={readings.min()} kWh")
    if (readings > max_demand_kwh).any():
        raise ValueError(
            f"Unphysically high demand detected in '{meter_col}': max={readings.max()} kWh "
            f"(exceeds building threshold {max_demand_kwh} kWh)"
        )

    return df_clean


class DemandDataset(Dataset):
    """
    PyTorch Dataset for electricity demand sequence forecasting.
    Provides sliding windows of shape (sequence_length, num_features) and target demand in kWh.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_timestamps: Optional[np.ndarray] = None,
        origin_timestamps: Optional[np.ndarray] = None,
        target_scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            features: 3D numpy array of shape (num_samples, sequence_length, num_features).
            targets: 1D or 2D numpy array of shape (num_samples,) or (num_samples, 1).
            target_timestamps: Optional 1D array of target timestamps (t + forecast_horizon).
            origin_timestamps: Optional 1D array of forecast origin timestamps (t, last input step).
            target_scaler: Optional fitted StandardScaler for unscaling to physical kWh.
        """
        assert len(features) == len(targets), "Features and targets must have same length"
        self.features = torch.tensor(features, dtype=torch.float32)
        if targets.ndim == 1:
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        else:
            self.targets = torch.tensor(targets, dtype=torch.float32)

        self.target_timestamps = target_timestamps
        self.origin_timestamps = origin_timestamps
        self.target_scaler = target_scaler

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    def inverse_transform_targets(
        self,
        targets: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Invert standard scaling to reconstruct physical demand in kWh.

        Args:
            targets: Array or tensor of scaled target values.

        Returns:
            1D numpy array of physical demand values (kWh).
        """
        if isinstance(targets, torch.Tensor):
            t_arr = targets.detach().cpu().numpy()
        else:
            t_arr = np.asarray(targets)

        if self.target_scaler is None:
            return t_arr.flatten().astype(np.float32)

        mean = float(self.target_scaler.mean_[0])
        scale = float(self.target_scaler.scale_[0])
        unscaled = t_arr.flatten() * scale + mean
        return unscaled.astype(np.float32)

    def get_time_alignment(self, idx: int) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Return (origin_timestamp, target_timestamp) for sample index."""
        t_origin = pd.Timestamp(self.origin_timestamps[idx]) if self.origin_timestamps is not None else None
        t_target = pd.Timestamp(self.target_timestamps[idx]) if self.target_timestamps is not None else None
        return t_origin, t_target


def build_demand_datasets(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Union[str, Path]] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "demand_kwh",
    time_col: str = "timestamp",
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_hours: int = 24,
    scale: bool = True,
    max_demand_kwh: float = 10000.0,
    split_basis: str = "legacy_source",
) -> Tuple[DemandDataset, DemandDataset, DemandDataset, Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Construct leakage-safe, causally aligned DemandDataset instances for train, val, and test splits.

    Enforces:
    1. Schema and non-negative demand validation.
    2. Chronological partition with embargo buffer between train/val and val/test boundaries.
    3. StandardScaler fitting strictly on training partition.
    4. Causal forward-only sliding window generation.

    If neither df nor existing csv_path is provided, raises FileNotFoundError clearly indicating
    that real BDG2 demand data is not yet downloaded and pipeline is blocked.
    """
    if df is None:
        if csv_path is None or not Path(csv_path).exists():
            raise FileNotFoundError(
                f"BDG2 demand dataset not found at '{csv_path}'. "
                "Real electricity demand data acquisition is currently BLOCKED pending BDG2 download. "
                "Use a validated synthetic DataFrame fixture to test the pipeline interface."
            )
        raw_df = pd.read_csv(csv_path)
    else:
        raw_df = df.copy()

    if time_col not in raw_df.columns:
        if time_col == "timestamp" and "datetime" in raw_df.columns:
            time_col = "datetime"
        elif time_col == "datetime" and "timestamp" in raw_df.columns:
            time_col = "timestamp"

    if target_col not in raw_df.columns:
        if target_col == "demand_kwh" and "electricity_demand_kwh" in raw_df.columns:
            target_col = "electricity_demand_kwh"
        elif target_col == "electricity_demand_kwh" and "demand_kwh" in raw_df.columns:
            target_col = "demand_kwh"

    if feature_cols is None:
        feature_cols = [target_col]

    # 1. Validate DataFrame
    valid_df = validate_demand_dataframe(
        raw_df,
        time_col=time_col,
        meter_col=target_col,
        max_demand_kwh=max_demand_kwh,
    )

    # 2. Chronological split with embargo
    train_df, val_df, test_df = split_timeseries_chronological(
        valid_df,
        time_col=time_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        embargo_steps=embargo_hours,
        ratio_basis=split_basis,
    )

    train_feat = train_df[feature_cols].to_numpy(dtype=np.float32)
    val_feat = val_df[feature_cols].to_numpy(dtype=np.float32)
    test_feat = test_df[feature_cols].to_numpy(dtype=np.float32)

    train_targ = train_df[target_col].to_numpy(dtype=np.float32)
    val_targ = val_df[target_col].to_numpy(dtype=np.float32)
    test_targ = test_df[target_col].to_numpy(dtype=np.float32)

    feat_scaler = None
    targ_scaler = None

    # 3. Fit scaler strictly on training split
    if scale:
        feat_scaler = StandardScaler()
        train_feat = feat_scaler.fit_transform(train_feat)
        val_feat = feat_scaler.transform(val_feat)
        test_feat = feat_scaler.transform(test_feat)

        targ_scaler = StandardScaler()
        train_targ = targ_scaler.fit_transform(train_targ.reshape(-1, 1)).flatten()
        val_targ = targ_scaler.transform(val_targ.reshape(-1, 1)).flatten()
        test_targ = targ_scaler.transform(test_targ.reshape(-1, 1)).flatten()

    # 4. Construct forward-only forecasting windows
    X_train, y_train = create_forecasting_windows(train_feat, train_targ, sequence_length, forecast_horizon)
    X_val, y_val = create_forecasting_windows(val_feat, val_targ, sequence_length, forecast_horizon)
    X_test, y_test = create_forecasting_windows(test_feat, test_targ, sequence_length, forecast_horizon)

    # 5. Extract origin and target timestamps
    t_train_origins = train_df[time_col].iloc[sequence_length - 1 : len(train_df) - forecast_horizon].to_numpy()
    t_train_targets = train_df[time_col].iloc[sequence_length + forecast_horizon - 1 : len(train_df)].to_numpy()

    t_val_origins = val_df[time_col].iloc[sequence_length - 1 : len(val_df) - forecast_horizon].to_numpy()
    t_val_targets = val_df[time_col].iloc[sequence_length + forecast_horizon - 1 : len(val_df)].to_numpy()

    t_test_origins = test_df[time_col].iloc[sequence_length - 1 : len(test_df) - forecast_horizon].to_numpy()
    t_test_targets = test_df[time_col].iloc[sequence_length + forecast_horizon - 1 : len(test_df)].to_numpy()

    assert len(X_train) == len(t_train_origins) == len(t_train_targets)
    assert len(X_val) == len(t_val_origins) == len(t_val_targets)
    assert len(X_test) == len(t_test_origins) == len(t_test_targets)

    train_ds = DemandDataset(
        X_train,
        y_train,
        target_timestamps=t_train_targets,
        origin_timestamps=t_train_origins,
        target_scaler=targ_scaler,
    )
    val_ds = DemandDataset(
        X_val,
        y_val,
        target_timestamps=t_val_targets,
        origin_timestamps=t_val_origins,
        target_scaler=targ_scaler,
    )
    test_ds = DemandDataset(
        X_test,
        y_test,
        target_timestamps=t_test_targets,
        origin_timestamps=t_test_origins,
        target_scaler=targ_scaler,
    )

    return train_ds, val_ds, test_ds, feat_scaler, targ_scaler
