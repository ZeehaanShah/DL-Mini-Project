"""
Baseline models for NASA POWER hourly solar irradiance forecasting.

Provides:
- NasaOneHourPersistenceBaseline: Predicts irradiance observed at forecast origin (t) for t + 1.
- NasaSeasonalNaiveBaseline: Predicts same-hour-yesterday irradiance (t + 1 - 24h = t - 23h).

Requirements:
- Predict in original physical units (Wh/m²).
- Support both raw (unscaled) input tensors and standardized tensors with inverse scaling.
- Causal and lookahead-free: strictly use observations available at the forecast origin.
"""
from typing import Optional, Union
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class NasaOneHourPersistenceBaseline:
    """
    One-hour persistence baseline for next-hour solar irradiance forecasting.

    Given a 24-hour input window ending at forecast origin t:
    Predicts y_{t+1} = y_t (the final observation in the input window).
    """

    def __init__(
        self,
        irradiance_feature_idx: int = 0,
        feature_scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            irradiance_feature_idx: Index of irradiance variable in feature dimension (default: 0).
            feature_scaler: Optional fitted StandardScaler used on input features.
        """
        self.irradiance_feature_idx = irradiance_feature_idx
        self.feature_scaler = feature_scaler

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Predict next-hour irradiance from the final observation in the window.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).

        Returns:
            1D numpy array of predictions in physical units (Wh/m²).
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X)

        assert X_arr.ndim == 3, f"Expected 3D input (N, L, D), got shape {X_arr.shape}"
        assert X_arr.shape[1] >= 1, "Sequence length must be >= 1"

        # Final observation at forecast origin t
        last_obs = X_arr[:, -1, self.irradiance_feature_idx]

        # If features were standardized, invert scaling to obtain physical Wh/m²
        if self.feature_scaler is not None:
            mean = self.feature_scaler.mean_[self.irradiance_feature_idx]
            scale = self.feature_scaler.scale_[self.irradiance_feature_idx]
            unscaled = last_obs * scale + mean
            return np.asarray(unscaled, dtype=np.float32)

        return np.asarray(last_obs, dtype=np.float32)


class NasaSeasonalNaiveBaseline:
    """
    Same-hour-yesterday seasonal naive baseline for next-hour solar irradiance forecasting.

    For a target at t + 1 hour, predicts the irradiance from 24 hours prior (t + 1 - 24 = t - 23).
    For sequence_length = 24 and horizon = 1, observation at t - 23 corresponds to index 0 of the input window.
    """

    def __init__(
        self,
        sequence_length: int = 24,
        forecast_horizon: int = 1,
        seasonal_period: int = 24,
        irradiance_feature_idx: int = 0,
        feature_scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            sequence_length: Past window length L (default: 24).
            forecast_horizon: Forecast horizon H (default: 1).
            seasonal_period: Seasonal period S in hours (default: 24).
            irradiance_feature_idx: Index of irradiance variable in feature dimension.
            feature_scaler: Optional fitted StandardScaler.
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.seasonal_period = seasonal_period
        self.irradiance_feature_idx = irradiance_feature_idx
        self.feature_scaler = feature_scaler

        # Relative index in the past window:
        # Window timesteps: [t - (L - 1), ..., t]
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
        Predict next-hour irradiance from the observation 24 hours before the target.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).

        Returns:
            1D numpy array of predictions in physical units (Wh/m²).
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X)

        assert X_arr.ndim == 3, f"Expected 3D input (N, L, D), got shape {X_arr.shape}"

        seasonal_obs = X_arr[:, self.seasonal_idx, self.irradiance_feature_idx]

        if self.feature_scaler is not None:
            mean = self.feature_scaler.mean_[self.irradiance_feature_idx]
            scale = self.feature_scaler.scale_[self.irradiance_feature_idx]
            unscaled = seasonal_obs * scale + mean
            return np.asarray(unscaled, dtype=np.float32)

        return np.asarray(seasonal_obs, dtype=np.float32)
