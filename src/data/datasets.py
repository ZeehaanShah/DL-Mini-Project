"""
PyTorch Dataset classes and data pipeline builders for DL MiniProject.

Provides:
- SkippdDataset: Efficient Parquet-backed dataset for sky images and concurrent PV output.
- NasaWeatherDataset: Windowed time-series dataset for next-hour solar irradiance prediction.
- High-level builders ensuring leakage-free calendar-day splitting and chronological embargoes.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import io

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

from .splits import (
    split_skippd_by_day,
    split_timeseries_chronological,
    create_forecasting_windows,
)

# Standard ImageNet normalization parameters for CNN transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_skippd_transform(normalize: bool = True) -> Callable:
    """Return default torchvision transform for SKIPP'D sky images."""
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(transform_list)


class SkippdDataset(Dataset):
    """
    PyTorch Dataset for SKIPP'D (SKy Images and Photovoltaic Power generation Dataset).

    Supports contemporaneous PV nowcasting (estimating instantaneous PV output in kW
    at time t from a single fish-eye sky image at time t).

    Reads directly from in-memory PyArrow table buffers for zero disk-overhead during epochs.
    """

    def __init__(
        self,
        table_or_path: Union[str, Path, pa.Table],
        indices: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            table_or_path: Filepath to parquet file or an existing PyArrow Table.
            indices: Optional 1D integer array specifying which subset rows to use.
            transform: Optional torchvision transform callable.
            return_metadata: If True, returns dict with image, pv, timestamp, and index.
        """
        if isinstance(table_or_path, pa.Table):
            self.table = table_or_path
        else:
            path = Path(table_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Parquet file not found: {path}")
            self.table = pq.read_table(path)

        self.num_total_rows = len(self.table)
        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
            if len(self.indices) > 0:
                assert self.indices.min() >= 0, "Negative indices not allowed"
                assert self.indices.max() < self.num_total_rows, "Index out of bounds"
        else:
            self.indices = np.arange(self.num_total_rows, dtype=np.int64)

        self.image_col = self.table.column("image")
        self.time_col = self.table.column("time")
        self.pv_col = self.table.column("pv")

        self.transform = transform if transform is not None else get_default_skippd_transform(normalize=False)
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Union[torch.Tensor, str]]]:
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.indices)}")

        row_idx = int(self.indices[idx])

        # Extract image bytes from Arrow struct
        img_entry = self.image_col[row_idx].as_py()
        raw_bytes = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # Apply image transform (e.g. ToTensor -> (3, 64, 64))
        image_tensor = self.transform(pil_image)

        # Target PV in kW
        pv_val = float(self.pv_col[row_idx].as_py())
        pv_tensor = torch.tensor(pv_val, dtype=torch.float32)

        if self.return_metadata:
            timestamp = self.time_col[row_idx].as_py()
            return {
                "image": image_tensor,
                "pv": pv_tensor,
                "timestamp": str(timestamp),
                "index": row_idx,
            }

        return image_tensor, pv_tensor

    def get_timestamps(self) -> pd.Series:
        """Return pandas DatetimeIndex of timestamps corresponding to this dataset's samples."""
        selected_times = [self.time_col[int(i)].as_py() for i in self.indices]
        return pd.Series(selected_times, name="time")

    def get_pv_values(self) -> np.ndarray:
        """Return numpy array of target PV values in kW for this dataset's samples."""
        return np.array([self.pv_col[int(i)].as_py() for i in self.indices], dtype=np.float32)


class NasaWeatherDataset(Dataset):
    """
    PyTorch Dataset for NASA POWER hourly weather and solar irradiance forecasting.
    Provides sliding windows of shape (sequence_length, num_features) and target irradiance.
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
            target_scaler: Optional fitted StandardScaler for target unscaling / physical reconstruction.
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

    def get_time_alignment(self, idx: int) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Return (origin_timestamp, target_timestamp) as pandas Timestamps for sample idx."""
        t_origin = pd.to_datetime(self.origin_timestamps[idx]) if self.origin_timestamps is not None else None
        t_target = pd.to_datetime(self.target_timestamps[idx]) if self.target_timestamps is not None else None
        return t_origin, t_target

    def inverse_transform_targets(self, targets: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform scaled target or prediction values back to original physical units (Wh/m²).

        Args:
            targets: Array or tensor of scaled target predictions.

        Returns:
            Numpy array of unscaled values in physical units (Wh/m²).
        """
        if self.target_scaler is None:
            if isinstance(targets, torch.Tensor):
                return targets.detach().cpu().numpy()
            return np.asarray(targets)

        if isinstance(targets, torch.Tensor):
            arr = targets.detach().cpu().numpy()
        else:
            arr = np.asarray(targets)

        orig_shape = arr.shape
        arr_2d = arr.reshape(-1, 1)
        unscaled_2d = self.target_scaler.inverse_transform(arr_2d)
        return unscaled_2d.reshape(orig_shape)


# =====================================================================
# High-Level Builders & Validation
# =====================================================================

def validate_and_load_nasa_dataframe(
    csv_path: Union[str, Path],
    feature_cols: Optional[List[str]] = None,
    target_col: str = "ALLSKY_SFC_SW_DWN",
) -> pd.DataFrame:
    """
    Validate and parse NASA POWER CSV file into a strictly continuous hourly DataFrame.

    Checks:
    - File exists and is non-empty.
    - Required calendar and meteorological columns exist.
    - Datetime column is monotonically increasing with zero duplicate timestamps.
    - Timestamps are strictly hourly continuous (zero missing hours or gaps).
    - No NaN, infinite, or sentinel missing values (e.g. <= -900) exist in features/target.

    Returns:
        Validated pd.DataFrame with sorted, monotonic DatetimeIndex.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"NASA POWER CSV not found: {path}")

    if feature_cols is None:
        feature_cols = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]

    df = pd.read_csv(path, skiprows=12)

    # 1. Column existence
    required_date_cols = {"YEAR", "MO", "DY", "HR"}
    missing_date = required_date_cols - set(df.columns)
    if missing_date:
        raise ValueError(f"NASA CSV missing required date columns: {missing_date}")

    required_val_cols = set(feature_cols) | {target_col}
    missing_val = required_val_cols - set(df.columns)
    if missing_val:
        raise ValueError(f"NASA CSV missing required feature/target columns: {missing_val}")

    # 2. Datetime parsing
    df["datetime"] = pd.to_datetime(
        df[["YEAR", "MO", "DY"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day"}
        ).assign(hour=df["HR"])
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    # 3. Monotonicity and duplicates
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("NASA timestamps are not monotonically increasing")
    if df["datetime"].duplicated().any():
        num_dup = int(df["datetime"].duplicated().sum())
        raise ValueError(f"NASA CSV contains {num_dup} duplicate timestamps")

    # 4. Hourly continuity
    time_diffs = df["datetime"].diff().iloc[1:]
    is_continuous_hourly = (time_diffs == pd.Timedelta(hours=1)).all()
    if not is_continuous_hourly:
        gap_indices = df.index[1:][time_diffs != pd.Timedelta(hours=1)]
        raise ValueError(f"NASA timestamps are not continuous hourly; gaps found at indices: {gap_indices.tolist()[:5]}")

    # 5. Finite values & sentinel check
    all_check_cols = list(dict.fromkeys(feature_cols + [target_col]))
    for col in all_check_cols:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN / null values")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Column '{col}' contains non-finite values (Inf/-Inf)")
        if (df[col] <= -900).any():
            raise ValueError(f"Column '{col}' contains NASA sentinel missing values (<= -900)")

    return df


def build_skippd_datasets(
    train_parquet_path: Union[str, Path],
    test_parquet_path: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.8,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    chronological: bool = True,
    random_state: int = 42,
) -> Tuple[SkippdDataset, SkippdDataset, Optional[SkippdDataset]]:
    """
    Construct leakage-free SKIPP'D PyTorch Datasets.

    Partitions train_parquet by full calendar day into train and validation sets,
    sharing the underlying in-memory table to avoid duplicating memory buffers.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_path = Path(train_parquet_path)
    train_table = pq.read_table(train_path)

    # Extract timestamps to compute calendar day split
    train_times = train_table.column("time").to_pandas()
    train_indices, val_indices = split_skippd_by_day(
        timestamps=train_times,
        train_ratio=train_ratio,
        chronological=chronological,
        random_state=random_state,
    )

    t_train = transform_train if transform_train is not None else get_default_skippd_transform(normalize=True)
    t_val = transform_val if transform_val is not None else get_default_skippd_transform(normalize=True)

    train_ds = SkippdDataset(table_or_path=train_table, indices=train_indices, transform=t_train)
    val_ds = SkippdDataset(table_or_path=train_table, indices=val_indices, transform=t_val)

    test_ds = None
    if test_parquet_path is not None:
        test_path = Path(test_parquet_path)
        test_table = pq.read_table(test_path)
        test_ds = SkippdDataset(table_or_path=test_table, transform=t_val)

    return train_ds, val_ds, test_ds


def build_nasa_datasets(
    csv_path: Union[str, Path],
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_hours: int = 24,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "ALLSKY_SFC_SW_DWN",
    scale: bool = True,
    split_basis: str = "legacy_source",
) -> Tuple[NasaWeatherDataset, NasaWeatherDataset, NasaWeatherDataset, Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Construct leakage-free NASA POWER PyTorch Datasets for sequence forecasting.

    Enforces:
    - Validation of column schema, continuous hourly timestamps, and finite values.
    - Chronological 70/15/15 partitioning with embargo gap between splits.
    - Preprocessing scaling fitted strictly on training partition.
    - Causal sliding windows guaranteeing t_input < t_target.
    - Verification of exact origin (t) and target (t + H) timestamp alignment.

    Returns:
        (train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler)
    """
    if feature_cols is None:
        feature_cols = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]

    # 1. Validate and load continuous hourly DataFrame
    df = validate_and_load_nasa_dataframe(
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    # 2. Chronological split with embargo
    train_df, val_df, test_df = split_timeseries_chronological(
        df,
        time_col="datetime",
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

    if scale:
        feat_scaler = StandardScaler()
        train_feat = feat_scaler.fit_transform(train_feat)
        val_feat = feat_scaler.transform(val_feat)
        test_feat = feat_scaler.transform(test_feat)

        targ_scaler = StandardScaler()
        train_targ = targ_scaler.fit_transform(train_targ.reshape(-1, 1)).flatten()
        val_targ = targ_scaler.transform(val_targ.reshape(-1, 1)).flatten()
        test_targ = targ_scaler.transform(test_targ.reshape(-1, 1)).flatten()

    # 3. Construct forward-only forecasting windows on each split
    X_train, y_train = create_forecasting_windows(train_feat, train_targ, sequence_length, forecast_horizon)
    X_val, y_val = create_forecasting_windows(val_feat, val_targ, sequence_length, forecast_horizon)
    X_test, y_test = create_forecasting_windows(test_feat, test_targ, sequence_length, forecast_horizon)

    # 4. Extract window origin (last input) and target timestamps for each split
    t_train_origins = train_df["datetime"].iloc[sequence_length - 1 : len(train_df) - forecast_horizon].to_numpy()
    t_train_targets = train_df["datetime"].iloc[sequence_length + forecast_horizon - 1 : len(train_df)].to_numpy()

    t_val_origins = val_df["datetime"].iloc[sequence_length - 1 : len(val_df) - forecast_horizon].to_numpy()
    t_val_targets = val_df["datetime"].iloc[sequence_length + forecast_horizon - 1 : len(val_df)].to_numpy()

    t_test_origins = test_df["datetime"].iloc[sequence_length - 1 : len(test_df) - forecast_horizon].to_numpy()
    t_test_targets = test_df["datetime"].iloc[sequence_length + forecast_horizon - 1 : len(test_df)].to_numpy()

    assert len(X_train) == len(t_train_origins) == len(t_train_targets)
    assert len(X_val) == len(t_val_origins) == len(t_val_targets)
    assert len(X_test) == len(t_test_origins) == len(t_test_targets)

    train_dataset = NasaWeatherDataset(
        X_train,
        y_train,
        target_timestamps=t_train_targets,
        origin_timestamps=t_train_origins,
        target_scaler=targ_scaler,
    )
    val_dataset = NasaWeatherDataset(
        X_val,
        y_val,
        target_timestamps=t_val_targets,
        origin_timestamps=t_val_origins,
        target_scaler=targ_scaler,
    )
    test_dataset = NasaWeatherDataset(
        X_test,
        y_test,
        target_timestamps=t_test_targets,
        origin_timestamps=t_test_origins,
        target_scaler=targ_scaler,
    )

    return train_dataset, val_dataset, test_dataset, feat_scaler, targ_scaler
