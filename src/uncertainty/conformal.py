"""
Split Conformal Prediction for time-series forecasting models.

Provides distribution-free prediction intervals with finite-sample coverage guarantees
under the assumption of exchangeability.

References:
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
  Distribution-free predictive inference for regression. JASA, 113(523), 1094-1111.
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction
  and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction,
  and estimation. JASA, 102(477), 359-378.
"""
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


def _to_1d_numpy(arr: Union[np.ndarray, torch.Tensor, list], name: str = "array") -> np.ndarray:
    """Convert input array-like to a flat 1D float64 numpy array with validation."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim > 1 and arr.size != max(arr.shape):
        raise ValueError(f"'{name}' must be 1-dimensional; got shape {arr.shape}")
    flat = arr.ravel()
    if len(flat) == 0:
        raise ValueError(f"'{name}' cannot be empty")
    if not np.isfinite(flat).all():
        n_invalid = int(np.isnan(flat).sum() + np.isinf(flat).sum())
        raise ValueError(f"'{name}' contains {n_invalid} non-finite (NaN or Inf) values")
    return flat


def calculate_interval_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    lower: Union[np.ndarray, torch.Tensor],
    upper: Union[np.ndarray, torch.Tensor],
    alpha: float,
) -> Dict[str, float]:
    """
    Compute empirical prediction interval evaluation metrics.

    Calculates:
    - coverage: fraction of true values falling within [lower, upper]
    - mpiw: Mean Prediction Interval Width (mean(upper - lower))
    - winkler_score: Mean Winkler interval score (penalizes width and under/over-coverage)
    - undercoverage_rate: fraction of observations falling below lower bound
    - overcoverage_rate: fraction of observations falling above upper bound
    - nominal_coverage: 1.0 - alpha
    - coverage_gap: empirical coverage minus nominal coverage

    Winkler Score definition (Gneiting & Raftery 2007):
      W_alpha(l, u, y) = (u - l) + (2 / alpha) * (l - y) * I(y < l)
                                 + (2 / alpha) * (y - u) * I(y > u)

    Args:
        y_true: Ground truth target values.
        lower: Lower prediction interval bounds.
        upper: Upper prediction interval bounds.
        alpha: Miscoverage significance level in (0, 1).

    Returns:
        Dictionary of interval performance metrics.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Significance level alpha must be strictly between 0 and 1; got {alpha}")

    yt = _to_1d_numpy(y_true, "y_true")
    l_arr = _to_1d_numpy(lower, "lower")
    u_arr = _to_1d_numpy(upper, "upper")

    if not (len(yt) == len(l_arr) == len(u_arr)):
        raise ValueError(
            f"Shape mismatch: y_true ({len(yt)}), lower ({len(l_arr)}), upper ({len(u_arr)}) "
            f"must have identical lengths."
        )

    # Check for invalid interval bounds where lower > upper
    inverted = l_arr > (u_arr + 1e-9)
    if np.any(inverted):
        n_inv = int(np.sum(inverted))
        raise ValueError(f"Found {n_inv} prediction intervals where lower > upper.")

    # Empirical coverage: y in [lower, upper]
    covered = (yt >= l_arr) & (yt <= u_arr)
    coverage = float(np.mean(covered))

    # Mean Prediction Interval Width (MPIW)
    widths = u_arr - l_arr
    mpiw = float(np.mean(widths))

    # Under-coverage (y < lower) and Over-coverage (y > upper)
    miss_lower = np.maximum(0.0, l_arr - yt)
    miss_upper = np.maximum(0.0, yt - u_arr)

    undercoverage_rate = float(np.mean(yt < l_arr))
    overcoverage_rate = float(np.mean(yt > u_arr))

    # Winkler Interval Score
    penalty_lower = (2.0 / alpha) * miss_lower
    penalty_upper = (2.0 / alpha) * miss_upper
    winkler_scores = widths + penalty_lower + penalty_upper
    mean_winkler = float(np.mean(winkler_scores))

    nominal_cov = 1.0 - alpha

    return {
        "coverage": round(coverage, 4),
        "nominal_coverage": round(nominal_cov, 4),
        "coverage_gap": round(coverage - nominal_cov, 4),
        "mpiw": round(mpiw, 4),
        "winkler_score": round(mean_winkler, 4),
        "undercoverage_rate": round(undercoverage_rate, 4),
        "overcoverage_rate": round(overcoverage_rate, 4),
        "n_samples": int(len(yt)),
    }


class SplitConformalCalibrator:
    """
    Split Conformal Prediction calibrator for regression models.

    Computes nonconformity scores using absolute residuals R_i = |y_i - y_hat_i|
    from a dedicated calibration split.

    Finite-Sample Adjustment:
    Given n calibration samples and nominal miscoverage level alpha, the nonconformity
    quantile index is determined by:
        k = ceil((n + 1) * (1 - alpha))
    clamped to n. The critical nonconformity score q is the k-th order statistic
    R_(k), guaranteeing P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha under exchangeability.

    Time-Series Nuance & Limitation:
    Conformal prediction guarantees rely strictly on the exchangeability (or i.i.d.)
    assumption of calibration and test residuals. For sequential time-series data,
    temporal auto-correlation, diurnal patterns, and non-stationary distribution shifts
    can violate exchangeability. Conformal intervals should therefore be treated as
    empirically calibrated margins rather than exact finite-sample guarantees.
    """

    def __init__(self, alpha: float = 0.10):
        """
        Initialize Split Conformal Calibrator.

        Args:
            alpha: Default significance level in (0, 1). Default 0.10 (90% nominal coverage).
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be strictly between 0 and 1; got {alpha}")
        self.alpha = float(alpha)
        self._residuals: Optional[np.ndarray] = None
        self._sorted_residuals: Optional[np.ndarray] = None
        self._n_calibration: int = 0
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return True if calibrator has been fitted on calibration data."""
        return self._is_fitted

    @property
    def n_calibration_samples(self) -> int:
        """Return number of calibration samples."""
        return self._n_calibration

    @property
    def residuals(self) -> np.ndarray:
        """Return copy of absolute calibration residuals."""
        if not self._is_fitted:
            raise ValueError("Calibrator is not fitted yet. Call fit() first.")
        return self._residuals.copy()

    def fit(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
    ) -> "SplitConformalCalibrator":
        """
        Compute absolute calibration residuals from dedicated calibration split.

        Args:
            y_true: Ground truth target values on calibration partition.
            y_pred: Point forecasts on calibration partition.

        Returns:
            Self (fitted calibrator).
        """
        yt = _to_1d_numpy(y_true, "y_true")
        yp = _to_1d_numpy(y_pred, "y_pred")

        if len(yt) != len(yp):
            raise ValueError(
                f"Shape mismatch: y_true length {len(yt)} != y_pred length {len(yp)}"
            )

        residuals = np.abs(yt - yp)
        self._residuals = residuals
        self._sorted_residuals = np.sort(residuals)
        self._n_calibration = len(residuals)
        self._is_fitted = True
        return self

    def compute_quantile(self, alpha: Optional[float] = None) -> float:
        """
        Compute nonconformity quantile score q using finite-sample correction.

        Formula:
            k = ceil((n + 1) * (1 - alpha))
            q = sorted_residuals[k - 1] (1-indexed k converted to 0-indexed)

        Args:
            alpha: Optional significance level. If None, uses self.alpha.

        Returns:
            Nonconformity threshold q (float).
        """
        if not self._is_fitted:
            raise ValueError("Calibrator is not fitted. Call fit() before computing quantile.")

        eff_alpha = float(alpha) if alpha is not None else self.alpha
        if not (0.0 < eff_alpha < 1.0):
            raise ValueError(f"Significance level alpha must be strictly between 0 and 1; got {eff_alpha}")

        n = self._n_calibration
        k = int(np.ceil((n + 1) * (1.0 - eff_alpha)))

        # Finite-sample index bound: clamp k between 1 and n
        k_clamped = min(max(k, 1), n)

        # 1-indexed k_clamped corresponds to 0-indexed index k_clamped - 1
        q = float(self._sorted_residuals[k_clamped - 1])
        return q

    def predict_interval(
        self,
        y_pred: Union[np.ndarray, torch.Tensor],
        alpha: Optional[float] = None,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for point forecasts.

        Interval: [y_pred - q, y_pred + q] optionally clipped to [clip_min, clip_max].

        Args:
            y_pred: Point predictions to generate intervals for.
            alpha: Significance level in (0, 1). Defaults to self.alpha.
            clip_min: Optional minimum physical lower bound (e.g., 0.0 for speed/demand).
            clip_max: Optional maximum physical upper bound.

        Returns:
            Tuple (lower_bound, upper_bound) as 1D float64 numpy arrays.
        """
        yp = _to_1d_numpy(y_pred, "y_pred")
        q = self.compute_quantile(alpha)

        lower = yp - q
        upper = yp + q

        if clip_min is not None:
            lower = np.maximum(lower, float(clip_min))
            upper = np.maximum(upper, float(clip_min))

        if clip_max is not None:
            lower = np.minimum(lower, float(clip_max))
            upper = np.minimum(upper, float(clip_max))

        return lower, upper


def evaluate_conformal_intervals(
    calibrator: SplitConformalCalibrator,
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    alpha: float,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate conformal prediction intervals with and without physical clipping.

    Reports:
    - raw_metrics: metrics without clipping
    - clipped_metrics: metrics with physical clipping applied
    - clipping_effect: difference in width and coverage due to clipping
    - quantile_q: critical nonconformity score q

    Args:
        calibrator: Fitted SplitConformalCalibrator.
        y_true: Ground truth target observations.
        y_pred: Point forecasts.
        alpha: Miscoverage significance level.
        clip_min: Optional lower clipping threshold.
        clip_max: Optional upper clipping threshold.

    Returns:
        Structured evaluation dictionary.
    """
    yt = _to_1d_numpy(y_true, "y_true")
    yp = _to_1d_numpy(y_pred, "y_pred")

    q = calibrator.compute_quantile(alpha)

    # 1. Raw unclipped interval
    lower_raw, upper_raw = calibrator.predict_interval(yp, alpha=alpha, clip_min=None, clip_max=None)
    raw_metrics = calculate_interval_metrics(yt, lower_raw, upper_raw, alpha=alpha)

    # 2. Clipped interval (if bounds supplied)
    has_clipping = (clip_min is not None) or (clip_max is not None)
    if has_clipping:
        lower_clip, upper_clip = calibrator.predict_interval(
            yp, alpha=alpha, clip_min=clip_min, clip_max=clip_max
        )
        clipped_metrics = calculate_interval_metrics(yt, lower_clip, upper_clip, alpha=alpha)
        width_reduction = round(raw_metrics["mpiw"] - clipped_metrics["mpiw"], 4)
        coverage_diff = round(clipped_metrics["coverage"] - raw_metrics["coverage"], 4)
    else:
        lower_clip, upper_clip = lower_raw, upper_raw
        clipped_metrics = raw_metrics
        width_reduction = 0.0
        coverage_diff = 0.0

    return {
        "alpha": alpha,
        "nominal_coverage": round(1.0 - alpha, 4),
        "quantile_q": round(q, 4),
        "raw_metrics": raw_metrics,
        "clipped_metrics": clipped_metrics,
        "clipping_effect": {
            "has_clipping": has_clipping,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "width_reduction": width_reduction,
            "coverage_diff": coverage_diff,
        },
        "lower_bounds": lower_clip,
        "upper_bounds": upper_clip,
        "lower_bounds_raw": lower_raw,
        "upper_bounds_raw": upper_raw,
    }
