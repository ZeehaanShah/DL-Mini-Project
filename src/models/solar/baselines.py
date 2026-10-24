"""
Baseline models for SKIPP'D solar PV nowcasting task.

Provides:
- GlobalMeanBaseline: Predicts global training mean for all samples.
- GlobalMedianBaseline: Predicts global training median for all samples.
- HourOfDayClimatologyBaseline: Predicts diurnal hourly mean based on local time.
- ConditionalPersistenceBaseline: Predicts preceding PV value strictly when delta_t <= 2 minutes.

All baselines:
- Fit statistics strictly on training partition.
- Return predictions in physical units (kW).
- Are non-deep-learning statistical heuristics.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class GlobalMeanBaseline:
    """Predicts the mean training PV output for every sample."""

    def __init__(self):
        self.mean_: Optional[float] = None

    def fit(self, y_train: Union[np.ndarray, list, pd.Series]) -> "GlobalMeanBaseline":
        """Fit global mean on training partition targets."""
        arr = np.asarray(y_train, dtype=np.float64)
        assert len(arr) > 0, "Cannot fit on empty training array"
        assert np.isfinite(arr).all(), "Training array contains non-finite values"
        self.mean_ = float(np.mean(arr))
        return self

    def predict(self, n_or_inputs: Union[int, np.ndarray, pd.Series]) -> np.ndarray:
        """Return predictions filled with training mean."""
        if self.mean_ is None:
            raise RuntimeError("GlobalMeanBaseline must be fitted before predict()")
        n = n_or_inputs if isinstance(n_or_inputs, int) else len(n_or_inputs)
        return np.full(shape=(n,), fill_value=self.mean_, dtype=np.float32)


class GlobalMedianBaseline:
    """Predicts the median training PV output for every sample."""

    def __init__(self):
        self.median_: Optional[float] = None

    def fit(self, y_train: Union[np.ndarray, list, pd.Series]) -> "GlobalMedianBaseline":
        """Fit global median on training partition targets."""
        arr = np.asarray(y_train, dtype=np.float64)
        assert len(arr) > 0, "Cannot fit on empty training array"
        assert np.isfinite(arr).all(), "Training array contains non-finite values"
        self.median_ = float(np.median(arr))
        return self

    def predict(self, n_or_inputs: Union[int, np.ndarray, pd.Series]) -> np.ndarray:
        """Return predictions filled with training median."""
        if self.median_ is None:
            raise RuntimeError("GlobalMedianBaseline must be fitted before predict()")
        n = n_or_inputs if isinstance(n_or_inputs, int) else len(n_or_inputs)
        return np.full(shape=(n,), fill_value=self.median_, dtype=np.float32)


class HourOfDayClimatologyBaseline:
    """
    Predicts the mean training PV output for the local hour of the day.

    Uses timestamps directly from the dataset (preserving US/Pacific local time).
    Hours with no training observations fall back to the global training mean.
    """

    def __init__(self):
        self.hourly_means_: Dict[int, float] = {}
        self.global_mean_: Optional[float] = None

    def fit(
        self,
        timestamps: Union[pd.Series, pd.DatetimeIndex],
        y_train: Union[np.ndarray, list, pd.Series],
    ) -> "HourOfDayClimatologyBaseline":
        """
        Fit hourly climatology on training partition.

        Args:
            timestamps: Series or DatetimeIndex of training timestamps in local time.
            y_train: Array of training PV power output (kW).
        """
        ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
        y = np.asarray(y_train, dtype=np.float64)

        assert len(ts) == len(y), f"Length mismatch: {len(ts)} timestamps != {len(y)} targets"
        assert len(y) > 0, "Cannot fit climatology on empty data"
        assert np.isfinite(y).all(), "Training array contains non-finite values"

        self.global_mean_ = float(np.mean(y))
        hours = ts.dt.hour

        df_train = pd.DataFrame({"hour": hours, "pv": y})
        grouped = df_train.groupby("hour")["pv"].mean()

        self.hourly_means_ = {int(h): float(val) for h, val in grouped.items()}
        return self

    def predict(self, timestamps: Union[pd.Series, pd.DatetimeIndex]) -> np.ndarray:
        """
        Predict PV output based on sample timestamps' local hour.

        Args:
            timestamps: Series or DatetimeIndex of evaluation timestamps.

        Returns:
            Numpy array of predicted PV power in kW.
        """
        if self.global_mean_ is None:
            raise RuntimeError("HourOfDayClimatologyBaseline must be fitted before predict()")

        ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
        hours = ts.dt.hour

        preds = np.empty(len(hours), dtype=np.float32)
        for i, h in enumerate(hours):
            preds[i] = self.hourly_means_.get(int(h), self.global_mean_)

        return preds


class ConditionalPersistenceBaseline:
    """
    Persistence baseline with explicit time-gap enforcement (Leakage / Realism Guard).

    Given a chronological stream of observations:
    - If delta_t = t_k - t_{k-1} <= max_gap_seconds (default 120s / 2 min),
      predicts P_k = P_{k-1}.
    - If delta_t > max_gap_seconds (night gap, sensor downtime), the persistence
      assumption is violated and the baseline falls back to climatology/mean or flags
      the sample as invalid for persistence evaluation.
    """

    def __init__(self, max_gap_seconds: float = 120.0):
        self.max_gap_seconds = float(max_gap_seconds)

    def evaluate_valid_pairs(
        self,
        timestamps: Union[pd.Series, pd.DatetimeIndex],
        pv_values: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter to only valid consecutive pairs where delta_t <= max_gap_seconds.

        Returns:
            (valid_indices, y_true_subset, y_pred_subset)
        """
        ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
        y = np.asarray(pv_values, dtype=np.float32)

        assert len(ts) == len(y), f"Length mismatch: {len(ts)} timestamps != {len(y)} pv values"
        assert len(y) >= 2, "Need at least 2 samples for persistence"
        assert ts.is_monotonic_increasing, "Timestamps must be monotonically increasing for persistence"

        time_deltas = ts.diff().dt.total_seconds().to_numpy()

        # Enforce same calendar day to strictly prevent pairing across overnight gaps
        dates = ts.dt.date.to_numpy()
        same_day = np.zeros(len(ts), dtype=bool)
        same_day[1:] = (dates[1:] == dates[:-1])

        # Valid condition: predecessor exists, delta_t is within 0 < delta_t <= max_gap_seconds, and on same day
        valid_mask = (time_deltas <= self.max_gap_seconds) & (time_deltas > 0) & same_day
        valid_indices = np.where(valid_mask)[0]

        # Prediction for sample k is sample k-1 (telemetry reading from preceding timestamp)
        y_true_subset = y[valid_indices]
        y_pred_subset = y[valid_indices - 1]

        return valid_indices, y_true_subset, y_pred_subset

    def predict_with_fallback(
        self,
        timestamps: Union[pd.Series, pd.DatetimeIndex],
        pv_values: Union[np.ndarray, pd.Series],
        fallback_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Produce predictions across ALL samples, using persistence when delta_t <= max_gap_seconds
        on the same day, and fallback_predictions when delta_t > max_gap_seconds, across midnight,
        or for the first sample of a session.
        """
        ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
        y = np.asarray(pv_values, dtype=np.float32)
        fb = np.asarray(fallback_predictions, dtype=np.float32)

        assert len(ts) == len(y) == len(fb), "All inputs must have identical length"
        assert ts.is_monotonic_increasing, "Timestamps must be monotonically increasing for persistence"

        time_deltas = ts.diff().dt.total_seconds().to_numpy()
        dates = ts.dt.date.to_numpy()
        same_day = np.zeros(len(ts), dtype=bool)
        same_day[1:] = (dates[1:] == dates[:-1])

        preds = fb.copy()

        # Valid condition: predecessor on same day within max_gap_seconds
        valid_mask = (time_deltas <= self.max_gap_seconds) & (time_deltas > 0) & same_day
        valid_indices = np.where(valid_mask)[0]

        preds[valid_indices] = y[valid_indices - 1]
        return preds
