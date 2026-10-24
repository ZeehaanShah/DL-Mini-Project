"""
Evaluation metrics and utility functions for DL MiniProject.

Provides:
- mean_absolute_error (MAE)
- root_mean_squared_error (RMSE)
- r2_score (R²)
- calculate_regression_metrics (combined dictionary of metrics)
"""
from typing import Dict, Optional, Union
import numpy as np
import torch


def _to_1d_numpy(arr: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Convert input array-like to a flat 1D float64 numpy array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64)
    return arr.ravel()


def mean_absolute_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE = (1 / N) * sum(|y_true - y_pred|)

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Float MAE value in the same physical units as the target.
    """
    yt = _to_1d_numpy(y_true)
    yp = _to_1d_numpy(y_pred)
    assert len(yt) == len(yp), f"Shape mismatch: y_true length {len(yt)} != y_pred length {len(yp)}"
    assert len(yt) > 0, "Cannot compute MAE on empty arrays"
    assert np.isfinite(yt).all(), "y_true contains non-finite values (NaN or Inf)"
    assert np.isfinite(yp).all(), "y_pred contains non-finite values (NaN or Inf)"

    return float(np.mean(np.abs(yt - yp)))


def root_mean_squared_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE = sqrt((1 / N) * sum((y_true - y_pred)^2))

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Float RMSE value in the same physical units as the target.
    """
    yt = _to_1d_numpy(y_true)
    yp = _to_1d_numpy(y_pred)
    assert len(yt) == len(yp), f"Shape mismatch: y_true length {len(yt)} != y_pred length {len(yp)}"
    assert len(yt) > 0, "Cannot compute RMSE on empty arrays"
    assert np.isfinite(yt).all(), "y_true contains non-finite values (NaN or Inf)"
    assert np.isfinite(yp).all(), "y_pred contains non-finite values (NaN or Inf)"

    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Calculate Coefficient of Determination (R²).

    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_true - y_pred)^2)
          SS_tot = sum((y_true - mean(y_true))^2)

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Float R² score. Returns 0.0 or NaN if SS_tot is zero (constant ground truth).
    """
    yt = _to_1d_numpy(y_true)
    yp = _to_1d_numpy(y_pred)
    assert len(yt) == len(yp), f"Shape mismatch: y_true length {len(yt)} != y_pred length {len(yp)}"
    assert len(yt) > 0, "Cannot compute R² on empty arrays"
    assert np.isfinite(yt).all(), "y_true contains non-finite values (NaN or Inf)"
    assert np.isfinite(yp).all(), "y_pred contains non-finite values (NaN or Inf)"

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)

    if np.isclose(ss_tot, 0.0):
        # When all true values are identical, SS_tot is zero
        return float("nan")

    return float(1.0 - (ss_res / ss_tot))


def calculate_regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    prefix: str = "",
) -> Dict[str, Union[float, int]]:
    """
    Compute full suite of regression metrics: MAE, RMSE, R², and sample count.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        prefix: Optional prefix for dictionary keys (e.g. 'val_').

    Returns:
        Dictionary mapping metric names to values:
        - {prefix}mae: Mean Absolute Error
        - {prefix}rmse: Root Mean Squared Error
        - {prefix}r2: Coefficient of Determination
        - {prefix}n_samples: Number of samples evaluated
    """
    yt = _to_1d_numpy(y_true)
    yp = _to_1d_numpy(y_pred)

    mae_val = mean_absolute_error(yt, yp)
    rmse_val = root_mean_squared_error(yt, yp)
    r2_val = r2_score(yt, yp)

    return {
        f"{prefix}mae": round(mae_val, 4),
        f"{prefix}rmse": round(rmse_val, 4),
        f"{prefix}r2": round(r2_val, 4) if not np.isnan(r2_val) else float("nan"),
        f"{prefix}n_samples": int(len(yt)),
    }
