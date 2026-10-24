"""
Cached model loaders, dataset accessors, and inference routines for the Streamlit dashboard.

Ensures:
- Windows PyTorch DLLs are initialized safely before torch imports.
- Neural network checkpoints are cached using @st.cache_resource (loaded once, zero re-loading per widget).
- DataFrames and evaluation series are cached using @st.cache_data.
- Clean fallback handling if optional checkpoints or raw datasets are absent.
"""
from pathlib import Path
import json
import io
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image

# 1. Early Windows PyTorch DLL initialization
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import src.utils.torch_loader  # noqa: F401
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import streamlit as st

from src.models.solar.resnet import ResNet18Nowcaster
from src.models.nasa_weather.sequence_models import WeatherLSTM
from src.models.wind.sequence_models import WindLSTM
from src.models.wind.baselines import TurbinePowerCurve
from src.models.demand.sequence_models import DemandLSTM
from src.data.datasets import (
    build_nasa_datasets,
    get_default_skippd_transform,
)
from src.data.wind import build_wind_datasets
from src.data.demand import build_demand_datasets
from src.uncertainty.conformal import SplitConformalCalibrator
from src.decision_support.energy_balance import (
    BatteryConfig,
    simulate_energy_dispatch,
    DispatchTimeSeries,
)
from src.decision_support.scenarios import (
    convert_irradiance_to_solar_pv_kwh,
    convert_wind_speed_to_generation_kwh,
)
from src.decision_support.optimizer import (
    CostModel,
    OptimizationObjective,
    optimize_energy_mix,
)
from src.decision_support.uncertainty_eval import evaluate_uncertainty_feasibility


def get_project_root() -> Path:
    """Return absolute path to project root."""
    return PROJECT_ROOT


@st.cache_data
def load_registry_data() -> pd.DataFrame:
    """Load and cache the complete experiment registry."""
    path = PROJECT_ROOT / "experiments" / "experiment_registry.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_uncertainty_summary() -> Dict[str, Any]:
    """Load and cache the Stage 4 conformal uncertainty summary."""
    path = PROJECT_ROOT / "reports" / "tables" / "uncertainty_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_decision_support_summary() -> Dict[str, Any]:
    """Load and cache the Stage 5 decision support summary."""
    path = PROJECT_ROOT / "reports" / "tables" / "decision_support_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_feasibility_scenarios_table() -> pd.DataFrame:
    """Load and cache the Stage 5 120-candidate scenario table."""
    path = PROJECT_ROOT / "reports" / "tables" / "energy_feasibility_scenarios.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# MODEL A: Solar Sky-Image Nowcaster (ResNet-18)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_solar_nowcaster() -> Optional[nn.Module]:
    """Load and cache Model A (ResNet-18 Fine-tuned) checkpoint."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / "EXP_SOLAR_DL_S3" / "best_model.pt"
    if not ckpt_path.exists():
        return None
    try:
        model = ResNet18Nowcaster(pretrained=False, mode="fine_tune", dropout=0.2)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
    except Exception as exc:
        st.warning(f"Could not load Solar ResNet checkpoint: {exc}")
        return None


def predict_solar_pv_from_image(image_input: Union[Image.Image, bytes, Path]) -> float:
    """
    Run contemporaneous PV power inference on a sky image.

    Args:
        image_input: PIL Image, raw image bytes, or Path.

    Returns:
        Estimated instantaneous PV power in kW (non-negative).
    """
    model = load_solar_nowcaster()
    if model is None:
        raise RuntimeError("Solar Nowcaster model checkpoint is unavailable.")

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Ensure 64x64 size
    if img.size != (64, 64):
        img = img.resize((64, 64), Image.Resampling.BILINEAR)

    transform = get_default_skippd_transform(normalize=True)
    tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 64, 64)

    with torch.no_grad():
        pred_kw = model(tensor).item()

    return float(max(0.0, pred_kw))


@st.cache_data
def get_solar_nowcasting_test_evaluation(max_samples: int = 260) -> Dict[str, Any]:
    """
    Return evaluated actual vs predicted test sequence for Model A (SKIPP'D Test Set).
    Extracts chronological test morning sequence from test parquet.
    """
    path = PROJECT_ROOT / "data" / "raw" / "skippd" / "test-00000-of-00001.parquet"
    if not path.exists():
        return {}
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        n = min(len(table), max_samples)
        df = table.slice(0, n).to_pandas()
        model = load_solar_nowcaster()
        if model is None:
            return {}

        transform = get_default_skippd_transform(normalize=True)
        tensors = []
        for raw in df["image"]:
            img_bytes = raw["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensors.append(transform(img))

        batch = torch.stack(tensors)
        with torch.no_grad():
            preds = model(batch).cpu().numpy().flatten()

        actuals = np.maximum(0.0, df["pv"].values)
        preds = np.maximum(0.0, preds)
        times = [t.strftime("%H:%M") for t in df["time"]]

        mae = float(np.mean(np.abs(actuals - preds)))
        rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))

        return {
            "timestamps": times,
            "y_true": actuals,
            "y_pred": preds,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "unit": "kW",
            "model_name": "ResNet-18 (Fine-Tuned)",
            "experiment_id": "EXP_SOLAR_DL_S3",
            "overall_test_mae": 2.301,
        }
    except Exception as exc:
        st.warning(f"Could not load Solar nowcasting test series: {exc}")
        return {}



# -----------------------------------------------------------------------------
# MODEL B: Solar Irradiance Forecasting (WeatherLSTM)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_weather_forecaster() -> Optional[Tuple[nn.Module, Any, Any]]:
    """Load and cache Model B (WeatherLSTM) with its test dataset and scaler."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / "EXP_NASA_DL_W1" / "best_model.pt"
    csv_path = PROJECT_ROOT / "data" / "raw" / "nasa_power" / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"
    if not ckpt_path.exists() or not csv_path.exists():
        return None
    try:
        train_ds, val_ds, test_ds, scaler, _ = build_nasa_datasets(
            csv_path=csv_path,
            sequence_length=24,
            forecast_horizon=1,
        )
        model = WeatherLSTM(input_dim=4, hidden_dim=32, num_layers=1, dropout=0.0)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, test_ds, scaler
    except Exception as exc:
        st.warning(f"Could not load WeatherLSTM checkpoint or data: {exc}")
        return None


@st.cache_data
def get_weather_test_evaluation(max_samples: int = 500) -> Dict[str, Any]:
    """Return evaluated actual vs predicted test sequences for Model B."""
    bundle = load_weather_forecaster()
    if bundle is None:
        return {}
    model, test_ds, scaler = bundle
    n = min(len(test_ds), max_samples)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    preds_scaled = []
    targets_scaled = []
    with torch.no_grad():
        for x_b, y_b in loader:
            p_b = model(x_b)
            preds_scaled.append(p_b.cpu().numpy().flatten())
            targets_scaled.append(y_b.cpu().numpy().flatten())

    p_arr = np.concatenate(preds_scaled)[:n]
    t_arr = np.concatenate(targets_scaled)[:n]

    y_pred = np.maximum(0.0, test_ds.inverse_transform_targets(p_arr))
    y_true = np.maximum(0.0, test_ds.inverse_transform_targets(t_arr))
    timestamps = test_ds.target_timestamps[:n]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else 0.0

    return {
        "timestamps": timestamps,
        "y_true": y_true,
        "y_pred": y_pred,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "unit": "Wh/m²",
    }


# -----------------------------------------------------------------------------
# MODEL C: Wind Speed Forecasting (WindLSTM)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_wind_forecaster() -> Optional[Tuple[nn.Module, Any, SplitConformalCalibrator]]:
    """Load and cache Model C (WindLSTM) with its test dataset and conformal calibrator."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / "EXP_WIND_DL_C2" / "best_model.pt"
    csv_path = PROJECT_ROOT / "data" / "processed" / "wind" / "wind_hourly_pilot.csv"
    if not ckpt_path.exists() or not csv_path.exists():
        return None
    try:
        curve = TurbinePowerCurve()
        _, cal_ds_w, test_ds_w, scaler, _ = build_wind_datasets(
            csv_path=csv_path,
            feature_cols=["wind_speed"],
            target_col="wind_speed",
            time_col="datetime",
            sequence_length=24,
            forecast_horizon=1,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_hours=24,
            scale=True,
            power_curve=curve,
        )
        model = WindLSTM(input_dim=1, hidden_dim=64, num_layers=1, dropout=0.1)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Compute calibration residuals
        cal_loader = DataLoader(cal_ds_w, batch_size=64, shuffle=False)
        y_true_cal = []
        y_pred_cal = []
        with torch.no_grad():
            for x_b, y_b in cal_loader:
                p_b = model(x_b)
                y_pred_cal.append(p_b.cpu().numpy().flatten())
                y_true_cal.append(y_b.cpu().numpy().flatten())
        y_true_c = cal_ds_w.inverse_transform_targets(np.concatenate(y_true_cal))
        y_pred_c = np.maximum(0.0, cal_ds_w.inverse_transform_targets(np.concatenate(y_pred_cal)))

        calibrator = SplitConformalCalibrator().fit(y_true_c, y_pred_c)
        return model, test_ds_w, calibrator
    except Exception as exc:
        st.warning(f"Could not load WindLSTM: {exc}")
        return None


@st.cache_data
def get_wind_test_evaluation(max_samples: int = 500) -> Dict[str, Any]:
    """Return evaluated actual vs predicted test sequences with conformal bounds for Model C."""
    bundle = load_wind_forecaster()
    if bundle is None:
        return {}
    model, test_ds, calibrator = bundle
    n = min(len(test_ds), max_samples)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    preds_scaled = []
    targets_scaled = []
    with torch.no_grad():
        for x_b, y_b in loader:
            p_b = model(x_b)
            preds_scaled.append(p_b.cpu().numpy().flatten())
            targets_scaled.append(y_b.cpu().numpy().flatten())

    y_pred = np.maximum(0.0, test_ds.inverse_transform_targets(np.concatenate(preds_scaled)[:n]))
    y_true = np.maximum(0.0, test_ds.inverse_transform_targets(np.concatenate(targets_scaled)[:n]))
    timestamps = test_ds.target_timestamps[:n]

    q_90 = calibrator.compute_quantile(0.10)
    y_low_90, y_upp_90 = calibrator.predict_interval(y_pred, alpha=0.10, clip_min=0.0)

    # Convert to turbine power proxy
    curve = TurbinePowerCurve()
    p_true = np.array([curve.wind_to_power(v) for v in y_true])
    p_pred = np.array([curve.wind_to_power(v) for v in y_pred])
    p_low = np.array([curve.wind_to_power(v) for v in y_low_90])
    p_upp = np.array([curve.wind_to_power(v) for v in y_upp_90])

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else 0.0

    return {
        "timestamps": timestamps,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_low_90": y_low_90,
        "y_upp_90": y_upp_90,
        "p_true_kw": p_true,
        "p_pred_kw": p_pred,
        "p_low_kw": p_low,
        "p_upp_kw": p_upp,
        "q_90": round(q_90, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "unit": "m/s",
    }


# -----------------------------------------------------------------------------
# MODEL D: Building Electricity Demand Forecasting (DemandLSTM)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_demand_forecaster() -> Optional[Tuple[nn.Module, Any, SplitConformalCalibrator]]:
    """Load and cache Model D (DemandLSTM) with its test dataset and conformal calibrator."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / "EXP_DEMAND_DL_D2" / "best_model.pt"
    csv_path = PROJECT_ROOT / "data" / "processed" / "demand" / "demand_hourly_pilot.csv"
    if not ckpt_path.exists() or not csv_path.exists():
        return None
    try:
        _, cal_ds_d, test_ds_d, scaler, _ = build_demand_datasets(
            csv_path=csv_path,
            feature_cols=["demand_kwh"],
            target_col="demand_kwh",
            time_col="datetime",
            sequence_length=24,
            forecast_horizon=1,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            embargo_hours=24,
            scale=True,
        )
        model = DemandLSTM(input_dim=1, hidden_dim=64, num_layers=1, dropout=0.1)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        cal_loader = DataLoader(cal_ds_d, batch_size=64, shuffle=False)
        y_true_cal = []
        y_pred_cal = []
        with torch.no_grad():
            for x_b, y_b in cal_loader:
                p_b = model(x_b)
                y_pred_cal.append(p_b.cpu().numpy().flatten())
                y_true_cal.append(y_b.cpu().numpy().flatten())
        y_true_c = cal_ds_d.inverse_transform_targets(np.concatenate(y_true_cal))
        y_pred_c = np.maximum(0.0, cal_ds_d.inverse_transform_targets(np.concatenate(y_pred_cal)))

        calibrator = SplitConformalCalibrator().fit(y_true_c, y_pred_c)
        return model, test_ds_d, calibrator
    except Exception as exc:
        st.warning(f"Could not load DemandLSTM: {exc}")
        return None


@st.cache_data
def get_demand_test_evaluation(max_samples: int = 500) -> Dict[str, Any]:
    """Return evaluated actual vs predicted test sequences with conformal bounds for Model D."""
    bundle = load_demand_forecaster()
    if bundle is None:
        return {}
    model, test_ds, calibrator = bundle
    n = min(len(test_ds), max_samples)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    preds_scaled = []
    targets_scaled = []
    with torch.no_grad():
        for x_b, y_b in loader:
            p_b = model(x_b)
            preds_scaled.append(p_b.cpu().numpy().flatten())
            targets_scaled.append(y_b.cpu().numpy().flatten())

    y_pred = np.maximum(0.0, test_ds.inverse_transform_targets(np.concatenate(preds_scaled)[:n]))
    y_true = np.maximum(0.0, test_ds.inverse_transform_targets(np.concatenate(targets_scaled)[:n]))
    timestamps = test_ds.target_timestamps[:n]

    q_90 = calibrator.compute_quantile(0.10)
    y_low_90, y_upp_90 = calibrator.predict_interval(y_pred, alpha=0.10, clip_min=0.0)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else 0.0

    return {
        "timestamps": timestamps,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_low_90": y_low_90,
        "y_upp_90": y_upp_90,
        "q_90": round(q_90, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "unit": "kWh",
    }


# -----------------------------------------------------------------------------
# MODEL E: Interactive Energy-Mix Simulator & Optimizer
# -----------------------------------------------------------------------------
@st.cache_data
def get_benchmark_time_series(duration_hours: int = 240) -> Dict[str, np.ndarray]:
    """
    Get synchronized 240-hour illustrative benchmark time series combining
    real Model B (irradiance), Model C (wind), and Model D (demand) test predictions.
    """
    w_eval = get_wind_test_evaluation(max_samples=duration_hours)
    d_eval = get_demand_test_evaluation(max_samples=duration_hours)
    nasa_csv = PROJECT_ROOT / "data" / "raw" / "nasa_power" / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"

    if not nasa_csv.exists() or "y_pred" not in w_eval or "y_pred" not in d_eval:
        return {}

    nasa_df = pd.read_csv(nasa_csv, skiprows=12)
    irr = nasa_df["ALLSKY_SFC_SW_DWN"].to_numpy(dtype=np.float64)[:duration_hours]

    return {
        "solar_irradiance": irr,
        "wind_speed": w_eval["y_pred"][:duration_hours],
        "wind_speed_low": w_eval["y_low_90"][:duration_hours],
        "wind_speed_upp": w_eval["y_upp_90"][:duration_hours],
        "demand": d_eval["y_pred"][:duration_hours],
        "demand_low": d_eval["y_low_90"][:duration_hours],
        "demand_upp": d_eval["y_upp_90"][:duration_hours],
        "timestamps": d_eval["timestamps"][:duration_hours],
    }


def simulate_custom_system(
    solar_kw: float,
    wind_kw: float,
    battery_kwh: float,
    grid_import_limit_kw: float = 100.0,
    duration_hours: int = 240,
    cost_model: Optional[CostModel] = None,
) -> Tuple[DispatchTimeSeries, Dict[str, Any], Dict[str, Any]]:
    """
    Simulate customized renewable capacity mix on benchmark series.

    Returns:
        (dispatch_timeseries, metrics_dict, costs_dict)
    """
    ts_data = get_benchmark_time_series(duration_hours)
    if not ts_data:
        raise RuntimeError("Benchmark time series data is unavailable.")

    # 1. Physical Generation Conversion
    solar_gen = convert_irradiance_to_solar_pv_kwh(ts_data["solar_irradiance"], solar_kw)
    wind_gen = convert_wind_speed_to_generation_kwh(ts_data["wind_speed"], wind_kw)
    renewable_avail = solar_gen + wind_gen

    # 2. Battery Storage
    batt = BatteryConfig(capacity_kwh=battery_kwh) if battery_kwh > 0 else None

    # 3. Simulate Energy Dispatch
    dispatch, metrics = simulate_energy_dispatch(
        demand_kwh=ts_data["demand"],
        renewable_available_kwh=renewable_avail,
        grid_import_limit_kw=grid_import_limit_kw,
        battery=batt,
    )

    # 4. Compute Costs
    if cost_model is None:
        cost_model = CostModel(amortize_to_horizon_hours=float(duration_hours))
    costs = cost_model.compute_total_cost(
        solar_kw=solar_kw,
        wind_kw=wind_kw,
        battery_kwh=battery_kwh,
        grid_import_kwh=metrics["grid_import_kwh"],
        unmet_demand_kwh=metrics["unmet_demand_kwh"],
        curtailment_kwh=metrics["curtailment_kwh"],
    )

    return dispatch, metrics, costs
