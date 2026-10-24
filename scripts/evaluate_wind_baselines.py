"""
Evaluate baseline models on real NREL India Wind Toolkit dataset (Gujarat 100m AGL).

Models evaluated:
1. WindPersistenceBaseline: 1-hour persistence (predicts wind speed observation at forecast origin t).
2. WindSeasonalNaiveBaseline: 24-hour diurnal seasonal naive (predicts observation from t - 23h).

Evaluation Protocol:
- Partitioning: Chronological 70% Train, 15% Validation, 15% Test with 24-hour embargo between partitions.
- Zero Leakage: StandardScaler fitted strictly on Train partition.
- Units: Evaluated in physical units (m/s) and estimated power proxy (kW via generic 2.0MW turbine curve).
- Primary Model Comparison: Validation partition.
- Confirmatory Benchmark: Test partition (evaluated after protocol is fixed).
- Registry: Logs experiment records to experiments/experiment_registry.csv.
"""
from pathlib import Path
import sys
from typing import Dict, List, Any

# Ensure project root is in sys.path and pre-load PyTorch DLLs on Windows
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils.torch_loader
import numpy as np
import pandas as pd

from src.data.wind import build_wind_datasets, TurbinePowerCurve
from src.evaluation.metrics import calculate_regression_metrics
from src.evaluation.registry import ExperimentRegistry
from src.models.wind.baselines import (
    WindPersistenceBaseline,
    WindSeasonalNaiveBaseline,
)


def evaluate_wind_baselines(
    csv_path: Path = PROJECT_ROOT / "data" / "processed" / "wind" / "wind_hourly_pilot.csv",
    log_to_registry: bool = True,
) -> pd.DataFrame:
    """Run baseline evaluations on real Gujarat wind dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed wind pilot dataset not found at {csv_path}. "
            "Run scripts/download_wind_pilot.py first."
        )

    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")
    records = []
    curve = TurbinePowerCurve()

    print("\n" + "=" * 80)
    print("EVALUATING MODEL C: NREL GUJARAT REAL WIND RESOURCE FORECASTING BASELINES")
    print("=" * 80)

    # Canonical 24-Hour Sequence Window (L=24, H=1)
    train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_wind_datasets(
        csv_path=csv_path,
        sequence_length=24,
        forecast_horizon=1,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
        power_curve=curve,
    )

    t_val_start = str(pd.Timestamp(val_ds.target_timestamps[0]))
    t_val_end = str(pd.Timestamp(val_ds.target_timestamps[-1]))
    t_test_start = str(pd.Timestamp(test_ds.target_timestamps[0]))
    t_test_end = str(pd.Timestamp(test_ds.target_timestamps[-1]))

    print(f"Train Windows: {len(train_ds):,}")
    print(f"Val Windows:   {len(val_ds):,} ({t_val_start} to {t_val_end})")
    print(f"Test Windows:  {len(test_ds):,} ({t_test_start} to {t_test_end})")

    y_val_true = val_ds.inverse_transform_targets(val_ds.targets)
    y_test_true = test_ds.inverse_transform_targets(test_ds.targets)

    # 1. 1-Hour Persistence Baseline
    persist = WindPersistenceBaseline(speed_feature_idx=0, feature_scaler=targ_scaler, power_curve=curve)

    # Val speed
    y_val_pred_p = persist.predict(val_ds.features)
    m_val_p = calculate_regression_metrics(y_val_true, y_val_pred_p)
    print(f"\n  [1h Persistence - Val Speed]   MAE: {m_val_p['mae']:.4f} m/s | RMSE: {m_val_p['rmse']:.4f} m/s | R^2: {m_val_p['r2']:.4f}")

    # Val power proxy
    p_val_true = val_ds.estimate_power(y_val_true)
    p_val_pred_p = persist.predict_power(val_ds.features)
    mp_val_p = calculate_regression_metrics(p_val_true, p_val_pred_p)
    print(f"  [1h Persistence - Val Power]   MAE: {mp_val_p['mae']:.2f} kW  | RMSE: {mp_val_p['rmse']:.2f} kW  | R^2: {mp_val_p['r2']:.4f}")

    # Test speed
    y_test_pred_p = persist.predict(test_ds.features)
    m_test_p = calculate_regression_metrics(y_test_true, y_test_pred_p)
    print(f"  [1h Persistence - Test Speed]  MAE: {m_test_p['mae']:.4f} m/s | RMSE: {m_test_p['rmse']:.4f} m/s | R^2: {m_test_p['r2']:.4f}")

    # Test power proxy
    p_test_true = test_ds.estimate_power(y_test_true)
    p_test_pred_p = persist.predict_power(test_ds.features)
    mp_test_p = calculate_regression_metrics(p_test_true, p_test_pred_p)
    print(f"  [1h Persistence - Test Power]  MAE: {mp_test_p['mae']:.2f} kW  | RMSE: {mp_test_p['rmse']:.2f} kW  | R^2: {mp_test_p['r2']:.4f}")

    if log_to_registry:
        r1 = registry.log_experiment(
            experiment_id="EXP_WIND_BASE_PERSIST_VAL",
            dataset="NREL_WIND_TOOLKIT",
            task="wind_speed_forecasting",
            model_name="WindPersistenceBaseline",
            partition="val",
            features="wind_speed (24h past)",
            target="wind_speed (m/s)",
            sequence_length=24,
            forecast_horizon=1,
            n_samples=len(y_val_true),
            mae=m_val_p["mae"],
            rmse=m_val_p["rmse"],
            r2=m_val_p["r2"],
            metric_unit="m/s",
            config="configs/wind_resource.yaml",
            notes=f"Predicts observation at forecast origin t. Period: {t_val_start} to {t_val_end}",
        )
        records.append(r1)

        r2 = registry.log_experiment(
            experiment_id="EXP_WIND_BASE_PERSIST_TEST",
            dataset="NREL_WIND_TOOLKIT",
            task="wind_speed_forecasting",
            model_name="WindPersistenceBaseline",
            partition="test",
            features="wind_speed (24h past)",
            target="wind_speed (m/s)",
            sequence_length=24,
            forecast_horizon=1,
            n_samples=len(y_test_true),
            mae=m_test_p["mae"],
            rmse=m_test_p["rmse"],
            r2=m_test_p["r2"],
            metric_unit="m/s",
            config="configs/wind_resource.yaml",
            notes=f"Held-out test set evaluation. Period: {t_test_start} to {t_test_end}",
        )
        records.append(r2)

    # 2. 24-Hour Seasonal Naive Baseline (same-hour-yesterday)
    seasonal = WindSeasonalNaiveBaseline(
        sequence_length=24,
        forecast_horizon=1,
        seasonal_period=24,
        speed_feature_idx=0,
        feature_scaler=targ_scaler,
        power_curve=curve,
    )

    y_val_pred_s = seasonal.predict(val_ds.features)
    m_val_s = calculate_regression_metrics(y_val_true, y_val_pred_s)
    print(f"\n  [24h Seasonal Naive - Val Speed]  MAE: {m_val_s['mae']:.4f} m/s | RMSE: {m_val_s['rmse']:.4f} m/s | R^2: {m_val_s['r2']:.4f}")

    p_val_pred_s = seasonal.predict_power(val_ds.features)
    mp_val_s = calculate_regression_metrics(p_val_true, p_val_pred_s)
    print(f"  [24h Seasonal Naive - Val Power]  MAE: {mp_val_s['mae']:.2f} kW  | RMSE: {mp_val_s['rmse']:.2f} kW  | R^2: {mp_val_s['r2']:.4f}")

    y_test_pred_s = seasonal.predict(test_ds.features)
    m_test_s = calculate_regression_metrics(y_test_true, y_test_pred_s)
    print(f"  [24h Seasonal Naive - Test Speed] MAE: {m_test_s['mae']:.4f} m/s | RMSE: {m_test_s['rmse']:.4f} m/s | R^2: {m_test_s['r2']:.4f}")

    p_test_pred_s = seasonal.predict_power(test_ds.features)
    mp_test_s = calculate_regression_metrics(p_test_true, p_test_pred_s)
    print(f"  [24h Seasonal Naive - Test Power] MAE: {mp_test_s['mae']:.2f} kW  | RMSE: {mp_test_s['rmse']:.2f} kW  | R^2: {mp_test_s['r2']:.4f}")

    if log_to_registry:
        r3 = registry.log_experiment(
            experiment_id="EXP_WIND_BASE_SEASONAL_VAL",
            dataset="NREL_WIND_TOOLKIT",
            task="wind_speed_forecasting",
            model_name="WindSeasonalNaiveBaseline",
            partition="val",
            features="wind_speed (24h past)",
            target="wind_speed (m/s)",
            sequence_length=24,
            forecast_horizon=1,
            n_samples=len(y_val_true),
            mae=m_val_s["mae"],
            rmse=m_val_s["rmse"],
            r2=m_val_s["r2"],
            metric_unit="m/s",
            config="configs/wind_resource.yaml",
            notes=f"Predicts observation from t - 23h (index 0). Period: {t_val_start} to {t_val_end}",
        )
        records.append(r3)

        r4 = registry.log_experiment(
            experiment_id="EXP_WIND_BASE_SEASONAL_TEST",
            dataset="NREL_WIND_TOOLKIT",
            task="wind_speed_forecasting",
            model_name="WindSeasonalNaiveBaseline",
            partition="test",
            features="wind_speed (24h past)",
            target="wind_speed (m/s)",
            sequence_length=24,
            forecast_horizon=1,
            n_samples=len(y_test_true),
            mae=m_test_s["mae"],
            rmse=m_test_s["rmse"],
            r2=m_test_s["r2"],
            metric_unit="m/s",
            config="configs/wind_resource.yaml",
            notes=f"Held-out test set evaluation. Period: {t_test_start} to {t_test_end}",
        )
        records.append(r4)

    print("\n" + "=" * 80)
    print("WIND BASELINES EVALUATION COMPLETED")
    print("=" * 80 + "\n")

    return pd.DataFrame(records)


if __name__ == "__main__":
    evaluate_wind_baselines()
