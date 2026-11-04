"""
Leakage-safe data splitting and windowing utilities.

Provides functions to partition datasets while strictly preventing data leakage:
- Calendar-day splitting for sky image datasets (prevents same-day sky correlation)
- Chronological splitting with embargo gaps for time-series data
- Forward-only window creation for multi-step / single-step forecasting
- Scaler fitting strictly on training partitions
"""
from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_skippd_by_day(
    timestamps: Union[pd.Series, pd.DatetimeIndex],
    train_ratio: float = 0.8,
    random_state: Optional[int] = 42,
    chronological: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split SKIPP'D timestamps into train and validation sets by calendar day.

    Ensures that ALL samples from any given calendar day are allocated entirely
    to either the training set or the validation set, preventing same-day
    sky condition correlation leakage (Leakage Risk L1).

    Args:
        timestamps: Series or DatetimeIndex of sample timestamps.
        train_ratio: Fraction of days allocated to training (default: 0.8).
        random_state: Random seed if chronological is False.
        chronological: If True, partitions the earliest days to train and later days to val.

    Returns:
        Tuple of (train_indices, val_indices) as numpy integer arrays.
    """
    if isinstance(timestamps, pd.DatetimeIndex):
        ts_series = pd.Series(timestamps)
    else:
        ts_series = timestamps.reset_index(drop=True)

    dates = ts_series.dt.date
    unique_days = np.array(sorted(dates.unique()))
    n_days = len(unique_days)
    n_train_days = int(np.floor(n_days * train_ratio))

    if chronological:
        train_days = set(unique_days[:n_train_days])
        val_days = set(unique_days[n_train_days:])
    else:
        rng = np.random.RandomState(random_state)
        shuffled_days = unique_days.copy()
        rng.shuffle(shuffled_days)
        train_days = set(shuffled_days[:n_train_days])
        val_days = set(shuffled_days[n_train_days:])

    # Verify zero day intersection
    assert len(train_days & val_days) == 0, "Leakage detected: shared days between train and val"

    train_mask = dates.isin(train_days).to_numpy()
    val_mask = dates.isin(val_days).to_numpy()

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    return train_indices, val_indices


def split_timeseries_chronological(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_steps: int = 24,
    ratio_basis: str = "retained",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series dataframe chronologically into train, val, and test partitions
    with an embargo gap between splits to prevent boundary overlap leakage (Leakage Risk L6).

    Args:
        df: Input DataFrame, sorted chronologically.
        time_col: Optional name of datetime column to verify monotonicity.
        train_ratio: Fraction of rows for training.
        val_ratio: Fraction of rows for validation.
        test_ratio: Fraction of rows for testing.
        embargo_steps: Number of steps to purge between split boundaries.
        ratio_basis: "retained" allocates the requested ratios across rows after removing
            both embargo gaps. "legacy_source" preserves historical checkpoint boundaries.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if not np.isfinite(ratios).all() or np.any(ratios <= 0.0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must be finite, positive, and sum to 1.0")
    if embargo_steps < 0:
        raise ValueError("embargo_steps must be non-negative")
    if ratio_basis not in {"retained", "legacy_source"}:
        raise ValueError("ratio_basis must be 'retained' or 'legacy_source'")
    n = len(df)

    if time_col is not None:
        assert df[time_col].is_monotonic_increasing, f"{time_col} must be monotonically increasing"
        assert not df[time_col].duplicated().any(), f"{time_col} contains duplicate timestamps"

    if ratio_basis == "retained":
        retained = n - 2 * embargo_steps
        if retained <= 0:
            raise ValueError("Series is too short for the requested embargo gaps")
        train_size = int(np.floor(retained * train_ratio))
        val_size = int(np.floor(retained * val_ratio))
        test_size = retained - train_size - val_size
        train_end = train_size
        val_start = train_end + embargo_steps
        val_end = val_start + val_size
        test_start = val_end + embargo_steps
        test_end = test_start + test_size
    else:
        # Preserve the historical boundaries used to train existing checkpoints.
        train_end = int(n * train_ratio)
        val_start = train_end + embargo_steps
        val_end = val_start + int(n * val_ratio)
        test_start = val_end + embargo_steps
        test_end = n

    if train_end <= 0 or val_start >= val_end or test_start >= test_end:
        raise ValueError("Series is too short for the requested ratios and embargo")

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    return train_df, val_df, test_df


def create_forecasting_windows(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct sliding windows for sequence forecasting without lookahead bias (Leakage Risk L2).

    Given sequence length L and forecast horizon H:
    Input window i uses features[i : i + L]
    Target i uses targets[i + L + H - 1]

    This guarantees that the input strictly precedes the target timestep.

    Args:
        features: 2D array of shape (N, num_features).
        targets: 1D or 2D array of targets, length N.
        sequence_length: Number of past time steps (L).
        forecast_horizon: Number of steps ahead to predict (H >= 1).

    Returns:
        X: 3D array of shape (num_samples, sequence_length, num_features).
        y: Target array of shape (num_samples, ...).
    """
    assert sequence_length >= 1, "sequence_length must be >= 1"
    assert forecast_horizon >= 1, "forecast_horizon must be >= 1"
    assert len(features) == len(targets), "features and targets must have same length"

    num_samples = len(features) - sequence_length - forecast_horizon + 1
    assert num_samples > 0, "Dataset too short for given sequence_length and forecast_horizon"

    X = np.empty((num_samples, sequence_length, features.shape[1]), dtype=features.dtype)
    y = np.empty((num_samples, *targets.shape[1:]), dtype=targets.dtype)

    for i in range(num_samples):
        X[i] = features[i : i + sequence_length]
        target_idx = i + sequence_length + forecast_horizon - 1
        y[i] = targets[target_idx]

    return X, y


def fit_scaler_train_only(
    train_data: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    test_data: Optional[np.ndarray] = None,
) -> Tuple[StandardScaler, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit a StandardScaler strictly on the training partition and apply to val and test
    partitions without updating scaler parameters (Leakage Risk L3).

    Args:
        train_data: Training partition array.
        val_data: Optional validation partition array.
        test_data: Optional test partition array.

    Returns:
        Tuple of (fitted_scaler, train_scaled, val_scaled, test_scaled).
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)

    val_scaled = scaler.transform(val_data) if val_data is not None else None
    test_scaled = scaler.transform(test_data) if test_data is not None else None

    return scaler, train_scaled, val_scaled, test_scaled
