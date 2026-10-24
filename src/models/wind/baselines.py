"""
Baseline models for wind resource forecasting and turbine power estimation.

Provides:
- WindPersistenceBaseline: Predicts wind speed observed at forecast origin (t) for t + H.
- WindSeasonalNaiveBaseline: Predicts same-hour-yesterday wind speed (t + H - 24h).
- Optional turbine power conversion via TurbinePowerCurve to benchmark estimated power generation (kW).

Requirements:
- Predict in original physical units (m/s or estimated kW).
- Support both raw (unscaled) input tensors and standardized tensors with inverse scaling.
- Causal and lookahead-free: strictly use observations available at the forecast origin.
"""
from typing import Optional, Union
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.data.wind import TurbinePowerCurve


class WindPersistenceBaseline:
    """
    Persistence baseline for wind speed forecasting.

    Given a past window ending at forecast origin t:
    Predicts v_{t+H} = v_t (the final observation in the input window).
    Optionally computes estimated turbine power output in kW.
    """

    def __init__(
        self,
        speed_feature_idx: int = 0,
        feature_scaler: Optional[StandardScaler] = None,
        power_curve: Optional[TurbinePowerCurve] = None,
    ):
        """
        Args:
            speed_feature_idx: Index of wind speed variable in feature dimension (default: 0).
            feature_scaler: Optional fitted StandardScaler used on input features.
            power_curve: Optional TurbinePowerCurve to convert predicted wind speed to estimated power (kW).
        """
        self.speed_feature_idx = speed_feature_idx
        self.feature_scaler = feature_scaler
        self.power_curve = power_curve or TurbinePowerCurve()

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Predict next wind speed from the final observation in the window.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).

        Returns:
            1D numpy array of predictions in physical units (m/s).
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
        last_obs = X_arr[:, -1, self.speed_feature_idx]
        if not np.isfinite(last_obs).all():
            raise ValueError("Input contains non-finite observations at forecast origin")

        if self.feature_scaler is not None:
            mean = self.feature_scaler.mean_[self.speed_feature_idx]
            scale = self.feature_scaler.scale_[self.speed_feature_idx]
            unscaled = last_obs * scale + mean
            return np.asarray(unscaled, dtype=np.float32)

        return np.asarray(last_obs, dtype=np.float32)

    def predict_power(
        self,
        X: Union[np.ndarray, torch.Tensor],
        air_density: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Predict estimated turbine power output in kW using persistence predicted wind speeds.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).
            air_density: Optional air density in kg/m^3.

        Returns:
            1D numpy array of estimated turbine power predictions in kW.
        """
        speed_preds = self.predict(X)
        return self.power_curve.wind_to_power(speed_preds, air_density=air_density)


class WindSeasonalNaiveBaseline:
    """
    Same-hour-yesterday seasonal naive baseline for wind speed forecasting.

    For a target at t + H, predicts the wind speed from 24 hours prior: (t + H - 24h).
    For sequence_length = 24 and horizon = 1, observation at (t + 1) - 24 = t - 23 corresponds
    to index 0 of the input window.
    """

    def __init__(
        self,
        sequence_length: int = 24,
        forecast_horizon: int = 1,
        seasonal_period: int = 24,
        speed_feature_idx: int = 0,
        feature_scaler: Optional[StandardScaler] = None,
        power_curve: Optional[TurbinePowerCurve] = None,
    ):
        """
        Args:
            sequence_length: Past window length L (default: 24).
            forecast_horizon: Forecast horizon H (default: 1).
            seasonal_period: Seasonal period S in hours (default: 24).
            speed_feature_idx: Index of wind speed variable in feature dimension.
            feature_scaler: Optional fitted StandardScaler.
            power_curve: Optional TurbinePowerCurve for power estimation (kW).
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.seasonal_period = seasonal_period
        self.speed_feature_idx = speed_feature_idx
        self.feature_scaler = feature_scaler
        self.power_curve = power_curve or TurbinePowerCurve()

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
        Predict next wind speed using the seasonal lag observation.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).

        Returns:
            1D numpy array of predictions in physical units (m/s).
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X)

        if X_arr.ndim != 3:
            raise ValueError(f"Expected 3D input (N, L, D), got shape {X_arr.shape}")

        seasonal_obs = X_arr[:, self.seasonal_idx, self.speed_feature_idx]
        if not np.isfinite(seasonal_obs).all():
            raise ValueError(
                f"Input contains non-finite observations at seasonal lag index {self.seasonal_idx}"
            )

        if self.feature_scaler is not None:
            mean = self.feature_scaler.mean_[self.speed_feature_idx]
            scale = self.feature_scaler.scale_[self.speed_feature_idx]
            unscaled = seasonal_obs * scale + mean
            return np.asarray(unscaled, dtype=np.float32)

        return np.asarray(seasonal_obs, dtype=np.float32)

    def predict_power(
        self,
        X: Union[np.ndarray, torch.Tensor],
        air_density: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Predict estimated turbine power output in kW using seasonal naive wind speed.

        Args:
            X: Input tensor of shape (N, sequence_length, num_features).
            air_density: Optional air density in kg/m^3.

        Returns:
            1D numpy array of estimated turbine power predictions in kW.
        """
        speed_preds = self.predict(X)
        return self.power_curve.wind_to_power(speed_preds, air_density=air_density)
