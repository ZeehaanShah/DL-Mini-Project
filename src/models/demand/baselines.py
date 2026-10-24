"""
Baseline models for electricity demand forecasting (BDG2).

Provides:
- DemandPersistenceBaseline: Predicts electricity demand observed at forecast origin (t) for t + H.
- DemandSeasonalNaiveBaseline: Predicts same-hour-yesterday demand (t + H - 24h).
- DemandWeeklySeasonalNaiveBaseline: Predicts same-hour-last-week demand (t + H - 168h).

Requirements:
- Predict in original physical units (kWh or kW).
- Support both raw (unscaled) input tensors and standardized tensors with inverse scaling.
- Causal and lookahead-free: strictly use observations available at the forecast origin.
"""
from typing import Optional, Union
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class DemandPersistenceBaseline:
    """
    Persistence baseline for electricity demand forecasting.

    Given a past window ending at forecast origin t:
    Predicts y_{t+H} = y_t (the final observation in the input window).
    """

    def __init__(
        self,
        meter_feature_idx: int = 0,
        feature_scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            meter_feature_idx: Index of demand variable in feature dimension (default: 0).
            feature_scaler: Optional fitted StandardScaler used on input features.
        """
        self.meter_feature_idx = meter_feature_idx
        self.feature_scaler = feature_scaler

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Predict next demand from the final observation in the window.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).

        Returns:
            1D numpy array of predictions in physical units (kWh).
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X)

        if X_arr.ndim != 3:
            raise ValueError(f"Expected 3D input (N, L, D), got shape {X_arr.shape}")
        if X_arr.shape[1] < 1:
            raise ValueError("Sequence length must be >= 1")

        # Final observation at forecast origin t
        last_obs = X_arr[:, -1, self.meter_feature_idx]
        if not np.isfinite(last_obs).all():
            raise ValueError("Input contains non-finite observations at forecast origin")

        if self.feature_scaler is not None:
            mean = self.feature_scaler.mean_[self.meter_feature_idx]
            scale = self.feature_scaler.scale_[self.meter_feature_idx]
            unscaled = last_obs * scale + mean
            return np.asarray(unscaled, dtype=np.float32)

        return np.asarray(last_obs, dtype=np.float32)


class DemandSeasonalNaiveBaseline:
    """
    Same-hour-yesterday seasonal naive baseline for electricity demand forecasting.

    For a target at t + H, predicts the demand from 24 hours prior: (t + H - 24h).
    For sequence_length = 24 and horizon = 1, observation at (t + 1) - 24 = t - 23 corresponds
    to index 0 of the input window.
    """

    def __init__(
        self,
        sequence_length: int = 24,
        forecast_horizon: int = 1,
        seasonal_period: int = 24,
        meter_feature_idx: int = 0,
        feature_scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            sequence_length: Past window length L (default: 24).
            forecast_horizon: Forecast horizon H (default: 1).
            seasonal_period: Daily seasonal period S in hours (default: 24).
            meter_feature_idx: Index of demand variable in feature dimension.
            feature_scaler: Optional fitted StandardScaler.
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.seasonal_period = seasonal_period
        self.meter_feature_idx = meter_feature_idx
        self.feature_scaler = feature_scaler

        # Relative index in the past window:
        # Timesteps in window: [t - (L - 1), ..., t]
        # Target timestamp: t + H
        # Target - S: (t + H) - S
        # Index in window: (t + H - S) - (t - (L - 1)) = L - 1 + H - S
        self.seasonal_idx = sequence_length - 1 + forecast_horizon - seasonal_period
        if not (0 <= self.seasonal_idx < sequence_length):
            raise ValueError(
                f"Seasonal lag index {self.seasonal_idx} falls outside window [0, {sequence_length - 1}]. "
                f"Window length {sequence_length} is too short for seasonal period {seasonal_period}."
            )

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Predict next demand using the 24h seasonal lag observation.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).

        Returns:
            1D numpy array of predictions in physical units (kWh).
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X)

        if X_arr.ndim != 3:
            raise ValueError(f"Expected 3D input (N, L, D), got shape {X_arr.shape}")

        seasonal_obs = X_arr[:, self.seasonal_idx, self.meter_feature_idx]
        if not np.isfinite(seasonal_obs).all():
            raise ValueError(
                f"Input contains non-finite observations at seasonal lag index {self.seasonal_idx}"
            )

        if self.feature_scaler is not None:
            mean = self.feature_scaler.mean_[self.meter_feature_idx]
            scale = self.feature_scaler.scale_[self.meter_feature_idx]
            unscaled = seasonal_obs * scale + mean
            return np.asarray(unscaled, dtype=np.float32)

        return np.asarray(seasonal_obs, dtype=np.float32)


class DemandWeeklySeasonalNaiveBaseline:
    """
    Same-hour-last-week seasonal naive baseline for electricity demand forecasting.

    Commercial building electricity demand exhibits strong 7-day cyclical patterns (weekdays vs weekends).
    For a target at t + H, predicts demand from 168 hours prior: (t + H - 168h).
    Requires sequence_length >= 168 + 1 - H.
    """

    def __init__(
        self,
        sequence_length: int = 168,
        forecast_horizon: int = 1,
        seasonal_period: int = 168,
        meter_feature_idx: int = 0,
        feature_scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            sequence_length: Past window length L (e.g. 168 for 1 week).
            forecast_horizon: Forecast horizon H (default: 1).
            seasonal_period: Weekly seasonal period in hours (default: 168 = 7 * 24).
            meter_feature_idx: Index of demand variable in feature dimension.
            feature_scaler: Optional fitted StandardScaler.
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.seasonal_period = seasonal_period
        self.meter_feature_idx = meter_feature_idx
        self.feature_scaler = feature_scaler

        self.seasonal_idx = sequence_length - 1 + forecast_horizon - seasonal_period
        if not (0 <= self.seasonal_idx < sequence_length):
            raise ValueError(
                f"Weekly seasonal lag index {self.seasonal_idx} falls outside window [0, {sequence_length - 1}]. "
                f"Window length {sequence_length} must be >= {seasonal_period + 1 - forecast_horizon} hours."
            )

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Predict demand using the 168h (weekly) seasonal lag observation.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).

        Returns:
            1D numpy array of predictions in physical units (kWh).
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X)

        if X_arr.ndim != 3:
            raise ValueError(f"Expected 3D input (N, L, D), got shape {X_arr.shape}")

        seasonal_obs = X_arr[:, self.seasonal_idx, self.meter_feature_idx]
        if not np.isfinite(seasonal_obs).all():
            raise ValueError(
                f"Input contains non-finite observations at weekly lag index {self.seasonal_idx}"
            )

        if self.feature_scaler is not None:
            mean = self.feature_scaler.mean_[self.meter_feature_idx]
            scale = self.feature_scaler.scale_[self.meter_feature_idx]
            unscaled = seasonal_obs * scale + mean
            return np.asarray(unscaled, dtype=np.float32)

        return np.asarray(seasonal_obs, dtype=np.float32)
