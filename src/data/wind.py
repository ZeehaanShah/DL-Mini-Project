"""
Wind resource data ingestion, schema validation, turbine power estimation, and PyTorch dataset.

Designed for NREL Wind Toolkit / India Wind Toolkit and meteorological wind sequences:
- Validates timestamp monotonicity, hourly continuity, physical boundary constraints (m/s),
  and rejects non-finite/sentinel values.
- Implements an illustrative turbine power curve (m/s -> estimated kW) based on a piecewise
  aerodynamic cubic engineering approximation. The primary forecasting target is physical wind speed;
  estimated power is an illustrative proxy, NOT measured wind-farm generation.
- Lookahead-free causal sliding window construction with chronological splitting and embargo.
"""
from dataclasses import dataclass
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


@dataclass
class TurbinePowerCurve:
    """
    Illustrative engineering approximation for utility-scale wind turbine power estimation.

    Converts hub-height wind speed (m/s) to estimated electrical power output proxy (kW).
    Follows standard aerodynamic power principles:
    - Cut-in speed (v_in): Turbine begins rotating and generating power.
    - Rated speed (v_rated): Turbine reaches nominal rated capacity (P_rated).
    - Cut-out speed (v_out): Turbine shuts down for storm protection.
    - Cubic power region: P(v) proportional to (v^3 - v_in^3) / (v_rated^3 - v_in^3).
    - Optional air density correction: v_eff = v * (rho / rho_0)^(1/3) where rho_0 = 1.225 kg/m^3.

    IMPORTANT:
    - The sequence model forecasts meteorological wind speed (m/s) and converts it to estimated power.
    - This generic 2.0 MW curve is an illustrative engineering approximation and power proxy,
      NOT a validated turbine specification under IEC 61400-12-1, nor an empirical model of any
      specific commercial turbine.
    - Estimated power output must NEVER be described as measured wind-farm generation or actual
      SCADA production data.
    """

    turbine_model: str = "Generic_2.0MW_Illustrative_Proxy"
    rated_power_kw: float = 2000.0
    cut_in_speed: float = 3.0       # m/s
    rated_speed: float = 12.0      # m/s
    cut_out_speed: float = 25.0    # m/s
    hub_height_m: float = 100.0    # meters
    reference_density: float = 1.225  # kg/m^3 standard sea-level air density

    def __post_init__(self):
        if not (0.0 < self.cut_in_speed < self.rated_speed < self.cut_out_speed):
            raise ValueError(
                f"Invalid speed thresholds: cut_in ({self.cut_in_speed}) < "
                f"rated ({self.rated_speed}) < cut_out ({self.cut_out_speed})"
            )
        if self.rated_power_kw <= 0:
            raise ValueError(f"rated_power_kw must be positive, got {self.rated_power_kw}")
        if self.hub_height_m <= 0:
            raise ValueError(f"hub_height_m must be positive, got {self.hub_height_m}")

    def wind_to_power(
        self,
        wind_speed: Union[float, np.ndarray, torch.Tensor],
        air_density: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Convert wind speed in m/s to estimated electrical power proxy in kW.

        Args:
            wind_speed: Hub-height wind speed in m/s (scalar or array).
            air_density: Optional air density in kg/m^3 for air density normalization.

        Returns:
            Estimated power output proxy in kW (scalar or numpy array).
        """
        is_torch = isinstance(wind_speed, torch.Tensor)
        if is_torch:
            ws_arr = wind_speed.detach().cpu().numpy()
        elif isinstance(wind_speed, (int, float)):
            ws_arr = np.array([wind_speed], dtype=np.float64)
        else:
            ws_arr = np.asarray(wind_speed, dtype=np.float64)

        if np.any(ws_arr < 0.0):
            raise ValueError("Wind speed must be non-negative (>= 0 m/s)")

        # Apply density correction if density values provided
        if air_density is not None:
            if isinstance(air_density, torch.Tensor):
                rho_arr = air_density.detach().cpu().numpy()
            elif isinstance(air_density, (int, float)):
                rho_arr = np.array([air_density], dtype=np.float64)
            else:
                rho_arr = np.asarray(air_density, dtype=np.float64)
            if np.any(rho_arr <= 0.0):
                raise ValueError("Air density must be strictly positive (> 0 kg/m^3)")
            # Standard air density normalization formula: v_eff = v * (rho / rho_0)^(1/3)
            effective_v = ws_arr * np.cbrt(rho_arr / self.reference_density)
        else:
            effective_v = ws_arr

        # Vectorized piecewise power calculation
        power = np.zeros_like(effective_v, dtype=np.float64)

        # Region 2: Between cut-in and rated (aerodynamic cubic regime)
        cubic_mask = (effective_v >= self.cut_in_speed) & (effective_v < self.rated_speed)
        v_diff_cube = effective_v[cubic_mask] ** 3 - self.cut_in_speed ** 3
        denom_cube = self.rated_speed ** 3 - self.cut_in_speed ** 3
        power[cubic_mask] = self.rated_power_kw * (v_diff_cube / denom_cube)

        # Region 3: Between rated and cut-out (constant rated generation)
        rated_mask = (effective_v >= self.rated_speed) & (effective_v < self.cut_out_speed)
        power[rated_mask] = self.rated_power_kw

        # Region 4: Beyond cut-out speed (storm shutdown) -> 0.0 kW (already zero)

        if isinstance(wind_speed, (int, float)):
            return float(power[0])
        return power.astype(np.float32)

    __call__ = wind_to_power
    estimate_power = wind_to_power


def validate_wind_dataframe(
    df: pd.DataFrame,
    time_col: str = "datetime",
    speed_col: str = "wind_speed",
    direction_col: Optional[str] = None,
    temp_col: Optional[str] = None,
    pressure_col: Optional[str] = None,
    expected_freq: str = "1h",
) -> pd.DataFrame:
    """
    Validate schema, continuity, temporal monotonicity, and physical boundary constraints
    for wind resource time series.

    Args:
        df: Input pandas DataFrame.
        time_col: Column containing timestamp observations.
        speed_col: Column containing wind speed observations in m/s.
        direction_col: Optional column for wind direction in degrees.
        temp_col: Optional column for temperature (in °C or K).
        pressure_col: Optional column for air pressure (in Pa or hPa).
        expected_freq: Expected frequency string (e.g. '1h' for hourly, '15min' for 15-minute).

    Returns:
        Validated DataFrame sorted chronologically with guaranteed continuity.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    # Required columns presence
    required = [time_col, speed_col]
    if direction_col is not None:
        required.append(direction_col)
    if temp_col is not None:
        required.append(temp_col)
    if pressure_col is not None:
        required.append(pressure_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Wind DataFrame is missing required columns: {missing}. Available: {list(df.columns)}")

    df_clean = df.copy()

    # Convert timestamps to DatetimeIndex
    if not pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])

    # Check for duplicate timestamps
    if df_clean[time_col].duplicated().any():
        n_dups = int(df_clean[time_col].duplicated().sum())
        raise ValueError(f"Wind DataFrame contains {n_dups} duplicate timestamps in '{time_col}'")

    # Sort chronologically if needed
    if not df_clean[time_col].is_monotonic_increasing:
        df_clean = df_clean.sort_values(by=time_col).reset_index(drop=True)

    # Verify temporal continuity
    if expected_freq == "1h":
        diffs = df_clean[time_col].diff().iloc[1:]
        if not (diffs == pd.Timedelta(hours=1)).all():
            irregular_indices = np.where(diffs != pd.Timedelta(hours=1))[0]
            raise ValueError(
                f"Wind DataFrame has {len(irregular_indices)} temporal discontinuities; "
                f"expected regular 1-hour intervals. First irregular at row {irregular_indices[0]}."
            )
    elif expected_freq == "15min":
        diffs = df_clean[time_col].diff().iloc[1:]
        if not (diffs == pd.Timedelta(minutes=15)).all():
            irregular_indices = np.where(diffs != pd.Timedelta(minutes=15))[0]
            raise ValueError(
                f"Wind DataFrame has {len(irregular_indices)} temporal discontinuities; "
                f"expected regular 15-minute intervals."
            )

    # Check non-finite values and sentinel values (e.g. -999, -9999)
    numeric_cols = [speed_col]
    if direction_col:
        numeric_cols.append(direction_col)
    if temp_col:
        numeric_cols.append(temp_col)
    if pressure_col:
        numeric_cols.append(pressure_col)

    for col in numeric_cols:
        series = pd.to_numeric(df_clean[col], errors="coerce")
        if series.isna().any():
            n_nan = int(series.isna().sum())
            raise ValueError(f"Column '{col}' contains {n_nan} NaN/null/non-numeric values.")
        if np.isinf(series.to_numpy()).any():
            raise ValueError(f"Column '{col}' contains infinite values.")
        if (series <= -900).any():
            n_sentinel = int((series <= -900).sum())
            raise ValueError(f"Column '{col}' contains {n_sentinel} sentinel missing values (<= -900).")
        df_clean[col] = series.to_numpy(dtype=np.float32)

    # Physical domain constraints
    # Wind speed: 0.0 to 75.0 m/s (extreme hurricane limit)
    ws = df_clean[speed_col].to_numpy()
    if (ws < 0.0).any():
        raise ValueError(f"Negative wind speed detected in '{speed_col}': min={ws.min()} m/s")
    if (ws > 75.0).any():
        raise ValueError(f"Unphysically high wind speed detected in '{speed_col}': max={ws.max()} m/s (>75 m/s)")

    # Wind direction: 0.0 to 360.0 degrees
    if direction_col is not None:
        wd = df_clean[direction_col].to_numpy()
        if (wd < 0.0).any() or (wd > 360.0).any():
            raise ValueError(f"Wind direction out of bounds [0, 360] in '{direction_col}': range=[{wd.min()}, {wd.max()}]")

    return df_clean


class WindDataset(Dataset):
    """
    PyTorch Dataset for wind resource sequence forecasting.
    Provides sliding windows of shape (sequence_length, num_features) and target wind speed.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_timestamps: Optional[np.ndarray] = None,
        origin_timestamps: Optional[np.ndarray] = None,
        target_scaler: Optional[StandardScaler] = None,
        power_curve: Optional[TurbinePowerCurve] = None,
    ):
        """
        Args:
            features: 3D numpy array of shape (num_samples, sequence_length, num_features).
            targets: 1D or 2D numpy array of shape (num_samples,) or (num_samples, 1).
            target_timestamps: Optional 1D array of target timestamps (t + forecast_horizon).
            origin_timestamps: Optional 1D array of forecast origin timestamps (t, last input step).
            target_scaler: Optional fitted StandardScaler for unscaling to m/s.
            power_curve: Optional TurbinePowerCurve for wind power estimation (kW).
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
        self.power_curve = power_curve or TurbinePowerCurve()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    def inverse_transform_targets(
        self,
        targets: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Invert standard scaling to reconstruct physical wind speed in m/s.

        Args:
            targets: Array or tensor of scaled target values.

        Returns:
            1D numpy array of physical wind speed values (m/s).
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

    def estimate_power(
        self,
        wind_speeds: Union[np.ndarray, torch.Tensor],
        air_density: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Convert wind speeds (in physical m/s) to estimated turbine electrical power in kW.

        Args:
            wind_speeds: Physical wind speed array in m/s.
            air_density: Optional air density array in kg/m^3.

        Returns:
            Estimated turbine power in kW.
        """
        return self.power_curve.wind_to_power(wind_speeds, air_density=air_density)

    def get_time_alignment(self, idx: int) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Return (origin_timestamp, target_timestamp) for sample index."""
        t_origin = pd.Timestamp(self.origin_timestamps[idx]) if self.origin_timestamps is not None else None
        t_target = pd.Timestamp(self.target_timestamps[idx]) if self.target_timestamps is not None else None
        return t_origin, t_target


def build_wind_datasets(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Union[str, Path]] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "wind_speed",
    time_col: str = "datetime",
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_hours: int = 24,
    scale: bool = True,
    power_curve: Optional[TurbinePowerCurve] = None,
    split_basis: str = "legacy_source",
) -> Tuple[WindDataset, WindDataset, WindDataset, Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Construct leakage-safe, causally aligned WindDataset instances for train, val, and test splits.

    Enforces:
    1. Schema and physical bounds validation.
    2. Chronological partition with embargo buffer between train/val and val/test boundaries.
    3. StandardScaler fitting strictly on training partition.
    4. Causal forward-only sliding window generation.

    If neither df nor existing csv_path is provided, raises FileNotFoundError clearly indicating
    that real wind data is not yet downloaded and pipeline is blocked.
    """
    if df is None:
        if csv_path is None or not Path(csv_path).exists():
            raise FileNotFoundError(
                f"Wind dataset not found at '{csv_path}'. "
                "Real wind data acquisition is currently BLOCKED pending NREL download approval. "
                "Use a validated synthetic DataFrame fixture to test the pipeline interface."
            )
        raw_df = pd.read_csv(csv_path)
    else:
        raw_df = df.copy()

    if feature_cols is None:
        feature_cols = [target_col]

    # 1. Validate DataFrame
    valid_df = validate_wind_dataframe(
        raw_df,
        time_col=time_col,
        speed_col=target_col,
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

    curve = power_curve or TurbinePowerCurve()

    train_ds = WindDataset(
        X_train,
        y_train,
        target_timestamps=t_train_targets,
        origin_timestamps=t_train_origins,
        target_scaler=targ_scaler,
        power_curve=curve,
    )
    val_ds = WindDataset(
        X_val,
        y_val,
        target_timestamps=t_val_targets,
        origin_timestamps=t_val_origins,
        target_scaler=targ_scaler,
        power_curve=curve,
    )
    test_ds = WindDataset(
        X_test,
        y_test,
        target_timestamps=t_test_targets,
        origin_timestamps=t_test_origins,
        target_scaler=targ_scaler,
        power_curve=curve,
    )

    return train_ds, val_ds, test_ds, feat_scaler, targ_scaler
