"""
Evaluate baseline models on real BDG2 electricity demand dataset.

Models evaluated:
1. DemandPersistenceBaseline: 1-hour persistence (predicts observation at forecast origin t).
2. DemandSeasonalNaiveBaseline: 24-hour diurnal seasonal naive (predicts observation from t - 23h).
3. DemandWeeklySeasonalNaiveBaseline: 168-hour weekly seasonal naive (predicts observation from t - 167h).

Evaluation Protocol:
- Partitioning: Chronological 70% Train, 15% Validation, 15% Test with 24-hour embargo between partitions.
- Zero Leakage: StandardScaler fitted strictly on Train partition.
- Units: Evaluated in original physical units (kWh) after inverse scaling.
- Primary Model Comparison: Validation partition.
- Confirmatory Benchmark: Test partition (evaluated after protocol is fixed).
- Controlled Comparison: Both 24h sequence length and 168h sequence length (for weekly persistence eligibility).
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

from src.data.demand import build_demand_datasets
from src.evaluation.metrics import calculate_regression_metrics
from src.evaluation.registry import ExperimentRegistry
from src.models.demand.baselines import (
    DemandPersistenceBaseline,
    DemandSeasonalNaiveBaseline,
    DemandWeeklySeasonalNaiveBaseline,
)


def evaluate_demand_baselines(
    csv_path: Path = PROJECT_ROOT / "data" / "processed" / "demand" / "demand_hourly_pilot.csv",
    log_to_registry: bool = True,
) -> pd.DataFrame:
    """Run baseline evaluations on real BDG2 electricity demand dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed demand pilot dataset not found at {csv_path}. "
            "Run scripts/download_bdg2_meter.py first."
        )

    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")
    records = []

    print("\n" + "=" * 80)
    print("EVALUATING MODEL D: BDG2 REAL ELECTRICITY DEMAND FORECASTING BASELINES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Protocol A: Canonical 24-Hour Sequence Window (L=24, H=1)
    # -------------------------------------------------------------------------
    print("\n--- Protocol A: Canonical 24-Hour Window (sequence_length=24, horizon=1) ---")
    train_ds_24, val_ds_24, test_ds_24, feat_scaler_24, targ_scaler_24 = build_demand_datasets(
        csv_path=csv_path,
        sequence_length=24,
        forecast_horizon=1,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
    )

    t_val_start_24 = str(pd.Timestamp(val_ds_24.target_timestamps[0]))
    t_val_end_24 = str(pd.Timestamp(val_ds_24.target_timestamps[-1]))
    t_test_start_24 = str(pd.Timestamp(test_ds_24.target_timestamps[0]))
    t_test_end_24 = str(pd.Timestamp(test_ds_24.target_timestamps[-1]))

    print(f"Train Windows (24h): {len(train_ds_24):,}")
    print(f"Val Windows (24h):   {len(val_ds_24):,} ({t_val_start_24} to {t_val_end_24})")
    print(f"Test Windows (24h):  {len(test_ds_24):,} ({t_test_start_24} to {t_test_end_24})")

    y_val_true_24 = val_ds_24.inverse_transform_targets(val_ds_24.targets)
    y_test_true_24 = test_ds_24.inverse_transform_targets(test_ds_24.targets)

    # A1. 1-Hour Persistence (24h window)
    persist_24 = DemandPersistenceBaseline(meter_feature_idx=0, feature_scaler=targ_scaler_24)

    y_val_pred_p24 = persist_24.predict(val_ds_24.features)
    m_val_p24 = calculate_regression_metrics(y_val_true_24, y_val_pred_p24)
    print(f"  [1h Persistence - Val]       MAE: {m_val_p24['mae']:.4f} kWh | RMSE: {m_val_p24['rmse']:.4f} kWh | R^2: {m_val_p24['r2']:.4f}")

    y_test_pred_p24 = persist_24.predict(test_ds_24.features)
    m_test_p24 = calculate_regression_metrics(y_test_true_24, y_test_pred_p24)
    print(f"  [1h Persistence - Test]      MAE: {m_test_p24['mae']:.4f} kWh | RMSE: {m_test_p24['rmse']:.4f} kWh | R^2: {m_test_p24['r2']:.4f}")

    if log_to_registry:
        r1 = registry.log_experiment(
            experiment_id="EXP_DEMAND_BASE_PERSIST_VAL",
            dataset="BDG2",
            task="demand_forecasting",
            model_name="DemandPersistenceBaseline",
            partition="val",
            features="demand_kwh (24h past)",
            target="demand_kwh (kWh)",
            sequence_length=24,
            forecast_horizon=1,
            n_samples=len(y_val_true_24),
            mae=m_val_p24["mae"],
            rmse=m_val_p24["rmse"],
            r2=m_val_p24["r2"],
            metric_unit="kWh",
            config="configs/bdg2_demand.yaml",
            notes=f"Predicts observation at forecast origin t. Period: {t_val_start_24} to {t_val_end_24}",
        )
        records.append(r1)

        r2 = registry.log_experiment(
            experiment_id="EXP_DEMAND_BASE_PERSIST_TEST",
            dataset="BDG2",
            task="demand_forecasting",
            model_name="DemandPersistenceBaseline",
            partition="test",
            features="demand_kwh (24h past)",
            target="demand_kwh (kWh)",
            sequence_length=24,
            forecast_horizon=1,
            n_samples=len(y_test_true_24),
            mae=m_test_p24["mae"],
            rmse=m_test_p24["rmse"],
            r2=m_test_p24["r2"],
            metric_unit="kWh",
            config="configs/bdg2_demand.yaml",
            notes=f"Held-out test set evaluation. Period: {t_test_start_24} to {t_test_end_24}",
        )
        records.append(r2)

    # A2. 24-Hour Seasonal Naive (same-hour-yesterday)
    seasonal_24 = DemandSeasonalNaiveBaseline(
        sequence_length=24,
        forecast_horizon=1,
        seasonal_period=24,
        meter_feature_idx=0,
        feature_scaler=targ_scaler_24,
    )

    y_val_pred_s24 = seasonal_24.predict(val_ds_24.features)
    m_val_s24 = calculate_regression_metrics(y_val_true_24, y_val_pred_s24)
    print(f"  [24h Seasonal Naive - Val]   MAE: {m_val_s24['mae']:.4f} kWh | RMSE: {m_val_s24['rmse']:.4f} kWh | R^2: {m_val_s24['r2']:.4f}")

    y_test_pred_s24 = seasonal_24.predict(test_ds_24.features)
    m_test_s24 = calculate_regression_metrics(y_test_true_24, y_test_pred_s24)
    print(f"  [24h Seasonal Naive - Test]  MAE: {m_test_s24['mae']:.4f} kWh | RMSE: {m_test_s24['rmse']:.4f} kWh | R^2: {m_test_s24['r2']:.4f}")

    if log_to_registry:
        r3 = registry.log_experiment(
            experiment_id="EXP_DEMAND_BASE_SEASONAL_24H_VAL",
            dataset="BDG2",
            task="demand_forecasting",
            model_name="DemandSeasonalNaiveBaseline",
            partition="val",
            features="demand_kwh (24h past)",
            target="demand_kwh (kWh)",
            sequence_length=24,
            forecast_horizon=1,
            n_samples=len(y_val_true_24),
            mae=m_val_s24["mae"],
            rmse=m_val_s24["rmse"],
            r2=m_val_s24["r2"],
            metric_unit="kWh",
            config="configs/bdg2_demand.yaml",
            notes=f"Predicts observation from t - 23h (index 0). Period: {t_val_start_24} to {t_val_end_24}",
        )
        records.append(r3)

        r4 = registry.log_experiment(
            experiment_id="EXP_DEMAND_BASE_SEASONAL_24H_TEST",
            dataset="BDG2",
            task="demand_forecasting",
            model_name="DemandSeasonalNaiveBaseline",
            partition="test",
            features="demand_kwh (24h past)",
            target="demand_kwh (kWh)",
            sequence_length=24,
            forecast_horizon=1,
            n_samples=len(y_test_true_24),
            mae=m_test_s24["mae"],
            rmse=m_test_s24["rmse"],
            r2=m_test_s24["r2"],
            metric_unit="kWh",
            config="configs/bdg2_demand.yaml",
            notes=f"Held-out test set evaluation. Period: {t_test_start_24} to {t_test_end_24}",
        )
        records.append(r4)

    # -------------------------------------------------------------------------
    # Protocol B: Extended 168-Hour Sequence Window (L=168, H=1)
    # Allows weekly seasonal persistence (same hour last week)
    # -------------------------------------------------------------------------
    print("\n--- Protocol B: Extended 168-Hour Window (sequence_length=168, horizon=1) ---")
    train_ds_168, val_ds_168, test_ds_168, feat_scaler_168, targ_scaler_168 = build_demand_datasets(
        csv_path=csv_path,
        sequence_length=168,
        forecast_horizon=1,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
    )

    t_val_start_168 = str(pd.Timestamp(val_ds_168.target_timestamps[0]))
    t_val_end_168 = str(pd.Timestamp(val_ds_168.target_timestamps[-1]))
    t_test_start_168 = str(pd.Timestamp(test_ds_168.target_timestamps[0]))
    t_test_end_168 = str(pd.Timestamp(test_ds_168.target_timestamps[-1]))

    print(f"Train Windows (168h): {len(train_ds_168):,}")
    print(f"Val Windows (168h):   {len(val_ds_168):,} ({t_val_start_168} to {t_val_end_168})")
    print(f"Test Windows (168h):  {len(test_ds_168):,} ({t_test_start_168} to {t_test_end_168})")

    y_val_true_168 = val_ds_168.inverse_transform_targets(val_ds_168.targets)
    y_test_true_168 = test_ds_168.inverse_transform_targets(test_ds_168.targets)

    # B1. 1-Hour Persistence on 168h window (controlled comparison)
    persist_168 = DemandPersistenceBaseline(meter_feature_idx=0, feature_scaler=targ_scaler_168)
    y_val_pred_p168 = persist_168.predict(val_ds_168.features)
    m_val_p168 = calculate_regression_metrics(y_val_true_168, y_val_pred_p168)
    print(f"  [1h Persistence (168h) - Val] MAE: {m_val_p168['mae']:.4f} kWh | RMSE: {m_val_p168['rmse']:.4f} kWh | R^2: {m_val_p168['r2']:.4f}")

    # B2. 24-Hour Seasonal Naive on 168h window (controlled comparison)
    seasonal_168_24 = DemandSeasonalNaiveBaseline(
        sequence_length=168,
        forecast_horizon=1,
        seasonal_period=24,
        meter_feature_idx=0,
        feature_scaler=targ_scaler_168,
    )
    y_val_pred_s168_24 = seasonal_168_24.predict(val_ds_168.features)
    m_val_s168_24 = calculate_regression_metrics(y_val_true_168, y_val_pred_s168_24)
    print(f"  [24h Seasonal (168h) - Val]  MAE: {m_val_s168_24['mae']:.4f} kWh | RMSE: {m_val_s168_24['rmse']:.4f} kWh | R^2: {m_val_s168_24['r2']:.4f}")

    # B3. 168-Hour Weekly Seasonal Naive (same hour last week)
    weekly_168 = DemandWeeklySeasonalNaiveBaseline(
        sequence_length=168,
        forecast_horizon=1,
        seasonal_period=168,
        meter_feature_idx=0,
        feature_scaler=targ_scaler_168,
    )

    y_val_pred_w168 = weekly_168.predict(val_ds_168.features)
    m_val_w168 = calculate_regression_metrics(y_val_true_168, y_val_pred_w168)
    print(f"  [168h Weekly Naive - Val]    MAE: {m_val_w168['mae']:.4f} kWh | RMSE: {m_val_w168['rmse']:.4f} kWh | R^2: {m_val_w168['r2']:.4f}")

    y_test_pred_w168 = weekly_168.predict(test_ds_168.features)
    m_test_w168 = calculate_regression_metrics(y_test_true_168, y_test_pred_w168)
    print(f"  [168h Weekly Naive - Test]   MAE: {m_test_w168['mae']:.4f} kWh | RMSE: {m_test_w168['rmse']:.4f} kWh | R^2: {m_test_w168['r2']:.4f}")

    if log_to_registry:
        r5 = registry.log_experiment(
            experiment_id="EXP_DEMAND_BASE_WEEKLY_168H_VAL",
            dataset="BDG2",
            task="demand_forecasting",
            model_name="DemandWeeklySeasonalNaiveBaseline",
            partition="val",
            features="demand_kwh (168h past)",
            target="demand_kwh (kWh)",
            sequence_length=168,
            forecast_horizon=1,
            n_samples=len(y_val_true_168),
            mae=m_val_w168["mae"],
            rmse=m_val_w168["rmse"],
            r2=m_val_w168["r2"],
            metric_unit="kWh",
            config="configs/bdg2_demand.yaml",
            notes=f"Predicts observation from t - 167h (index 0). Period: {t_val_start_168} to {t_val_end_168}",
        )
        records.append(r5)

        r6 = registry.log_experiment(
            experiment_id="EXP_DEMAND_BASE_WEEKLY_168H_TEST",
            dataset="BDG2",
            task="demand_forecasting",
            model_name="DemandWeeklySeasonalNaiveBaseline",
            partition="test",
            features="demand_kwh (168h past)",
            target="demand_kwh (kWh)",
            sequence_length=168,
            forecast_horizon=1,
            n_samples=len(y_test_true_168),
            mae=m_test_w168["mae"],
            rmse=m_test_w168["rmse"],
            r2=m_test_w168["r2"],
            metric_unit="kWh",
            config="configs/bdg2_demand.yaml",
            notes=f"Held-out test set evaluation. Period: {t_test_start_168} to {t_test_end_168}",
        )
        records.append(r6)

    print("\n" + "=" * 80)
    print("DEMAND BASELINES EVALUATION COMPLETED")
    print("=" * 80 + "\n")

    return pd.DataFrame(records)


if __name__ == "__main__":
    evaluate_demand_baselines()
