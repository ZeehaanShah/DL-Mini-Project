"""
Master orchestration script for Stage 3: Train, Evaluate, and Compare WindLSTM and DemandLSTM.

Executes:
1. Model C (Wind): Trains C1-C4 on NREL India Wind Toolkit dataset (Gujarat 100m AGL).
2. Model D (Demand): Trains D1-D4 on BDG2 Panther_education_Rosalie electricity demand series.
3. Generates hyperparameter comparison bar charts for both domains.
4. Performs strict validation-based model selection (comparing against baselines on matched validation timestamps).
5. Evaluates the selected frozen best checkpoints ONCE on held-out test partitions.
6. Generates chronological test plots and updates experiment registry.
"""
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path and initialize Windows PyTorch DLLs first
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import src.utils.torch_loader

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.train_wind import train_wind_model, parse_args as parse_wind_args
from scripts.train_demand import train_demand_model, parse_args as parse_demand_args
from src.data.wind import build_wind_datasets, TurbinePowerCurve
from src.data.demand import build_demand_datasets
from src.models.wind.sequence_models import WindLSTM
from src.models.demand.sequence_models import DemandLSTM
from src.training.trainer import RegressionTrainer
from src.training.plotting import (
    plot_hyperparameter_comparison,
    plot_baseline_vs_model,
    plot_chronological_test_predictions,
)
from src.evaluation.registry import ExperimentRegistry
from src.evaluation.metrics import calculate_regression_metrics


class ArgsWrapper:
    def __init__(self, config: str, epochs: int = 15, no_plot: bool = False):
        self.config = config
        self.experiment_id = None
        self.epochs = epochs
        self.batch_size = None
        self.lr = None
        self.hidden_dim = None
        self.num_layers = None
        self.dropout = None
        self.seed = None
        self.no_plot = no_plot


def run_stage3():
    print("=" * 80)
    print("STAGE 3: NEURAL-NETWORK TRAINING & BENCHMARKING (WindLSTM & DemandLSTM)")
    print("=" * 80)

    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")

    # -------------------------------------------------------------------------
    # PART 1: MODEL C (WIND RESOURCE FORECASTING)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# PART 1: MODEL C — NREL INDIA WIND RESOURCE FORECASTING (WindLSTM)")
    print("#" * 80 + "\n")

    wind_configs = [
        "configs/experiments/wind_c1.yaml",
        "configs/experiments/wind_c2.yaml",
        "configs/experiments/wind_c3.yaml",
        "configs/experiments/wind_c4.yaml",
    ]
    wind_results = []

    for cfg_path in wind_configs:
        args = ArgsWrapper(config=cfg_path, epochs=15, no_plot=False)
        res = train_wind_model(args)
        wind_results.append(res)

    # Plot hyperparameter comparison for Wind
    wind_comp_data = [
        {
            "label": r["experiment_id"].replace("EXP_WIND_DL_", ""),
            "mae": r["val_speed"]["mae"],
            "rmse": r["val_speed"]["rmse"],
        }
        for r in wind_results
    ]
    plot_hyperparameter_comparison(
        wind_comp_data,
        fig_dir / "wind_hyperparameter_comparison.png",
        title="Model C (WindLSTM) - Validation Speed Error Across Architectures",
        unit="m/s",
    )

    # Select best Wind model based on validation MAE
    best_wind_res = min(wind_results, key=lambda r: r["val_speed"]["mae"])
    best_wind_id = best_wind_res["experiment_id"]
    print(f"\n>>> Best Wind Model Selected by Validation Performance: {best_wind_id}")
    print(f"    Val MAE: {best_wind_res['val_speed']['mae']:.4f} m/s | RMSE: {best_wind_res['val_speed']['rmse']:.4f} m/s | R^2: {best_wind_res['val_speed']['r2']:.4f}")

    # Comparative baseline plot for Wind (Validation)
    # Persistence: MAE 0.7644 m/s, RMSE 1.2845 m/s
    # Seasonal Naive: MAE 1.4190 m/s, RMSE 1.9749 m/s
    wind_baseline_comp = {
        "1h Persistence": {"mae": 0.7644, "rmse": 1.2845},
        "24h Seasonal": {"mae": 1.4190, "rmse": 1.9749},
        f"Best {best_wind_id.replace('EXP_WIND_DL_', '')}": {
            "mae": best_wind_res["val_speed"]["mae"],
            "rmse": best_wind_res["val_speed"]["rmse"],
        },
    }
    plot_baseline_vs_model(
        wind_baseline_comp,
        fig_dir / "wind_baseline_vs_model_val.png",
        title="Model C: Baseline vs Best WindLSTM (Validation Set)",
        unit="m/s",
    )

    # -------------------------------------------------------------------------
    # PART 2: MODEL D (ELECTRICITY DEMAND FORECASTING)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# PART 2: MODEL D — BDG2 ELECTRICITY DEMAND FORECASTING (DemandLSTM)")
    print("#" * 80 + "\n")

    demand_configs = [
        "configs/experiments/demand_d1.yaml",
        "configs/experiments/demand_d2.yaml",
        "configs/experiments/demand_d3.yaml",
        "configs/experiments/demand_d4.yaml",
    ]
    demand_results = []

    for cfg_path in demand_configs:
        args = ArgsWrapper(config=cfg_path, epochs=15, no_plot=False)
        res = train_demand_model(args)
        demand_results.append(res)

    # Plot hyperparameter comparison for Demand
    demand_comp_data = [
        {
            "label": r["experiment_id"].replace("EXP_DEMAND_DL_", ""),
            "mae": r["val_metrics"]["mae"],
            "rmse": r["val_metrics"]["rmse"],
        }
        for r in demand_results
    ]
    plot_hyperparameter_comparison(
        demand_comp_data,
        fig_dir / "demand_hyperparameter_comparison.png",
        title="Model D (DemandLSTM) - Validation Error Across Architectures",
        unit="kWh",
    )

    # Select best Demand model based on validation MAE
    best_demand_res = min(demand_results, key=lambda r: r["val_metrics"]["mae"])
    best_demand_id = best_demand_res["experiment_id"]
    print(f"\n>>> Best Demand Model Selected by Validation Performance: {best_demand_id}")
    print(f"    Val MAE: {best_demand_res['val_metrics']['mae']:.4f} kWh | RMSE: {best_demand_res['val_metrics']['rmse']:.4f} kWh | R^2: {best_demand_res['val_metrics']['r2']:.4f}")

    # Comparative baseline plot for Demand (Validation)
    # 1h Persistence: MAE 1.8052 kWh, RMSE 3.6647 kWh
    # 24h Seasonal: MAE 5.6797 kWh, RMSE 9.1857 kWh
    # 168h Weekly: MAE 6.5699 kWh, RMSE 9.7644 kWh
    demand_baseline_comp = {
        "1h Persistence": {"mae": 1.8052, "rmse": 3.6647},
        "24h Seasonal": {"mae": 5.6797, "rmse": 9.1857},
        "168h Weekly": {"mae": 6.5699, "rmse": 9.7644},
        f"Best {best_demand_id.replace('EXP_DEMAND_DL_', '')}": {
            "mae": best_demand_res["val_metrics"]["mae"],
            "rmse": best_demand_res["val_metrics"]["rmse"],
        },
    }
    plot_baseline_vs_model(
        demand_baseline_comp,
        fig_dir / "demand_baseline_vs_model_val.png",
        title="Model D: Baseline vs Best DemandLSTM (Validation Set)",
        unit="kWh",
    )

    # -------------------------------------------------------------------------
    # PART 3: HELD-OUT TEST EVALUATION (STRICTLY ONCE ON FROZEN CHECKPOINTS)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# PART 3: CONFIRMATORY HELD-OUT TEST EVALUATION")
    print("#" * 80 + "\n")

    # 1. Evaluate Best Wind Model on Test Set
    print(f"Evaluating {best_wind_id} on held-out wind test set...")
    import yaml
    with open(f"configs/experiments/{best_wind_id.lower().replace('exp_wind_dl_', 'wind_')}.yaml") as f:
        best_w_cfg = yaml.safe_load(f)

    curve = TurbinePowerCurve()
    _, _, test_ds_wind, _, targ_scaler_w = build_wind_datasets(
        csv_path=PROJECT_ROOT / "data" / "processed" / "wind" / "wind_hourly_pilot.csv",
        feature_cols=best_w_cfg.get("features_used", ["wind_speed"]),
        target_col="wind_speed",
        time_col="datetime",
        sequence_length=best_w_cfg.get("sequence_length", 24),
        forecast_horizon=best_w_cfg.get("forecast_horizon", 1),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
        power_curve=curve,
    )
    test_loader_w = DataLoader(test_ds_wind, batch_size=32, shuffle=False, num_workers=0)

    best_wind_model = WindLSTM(
        input_dim=best_w_cfg.get("input_dim", 1),
        hidden_dim=best_w_cfg.get("hidden_dim", 32),
        num_layers=best_w_cfg.get("num_layers", 1),
        dropout=best_w_cfg.get("dropout", 0.0),
    )
    ckpt_w_path = PROJECT_ROOT / "checkpoints" / best_wind_id / "best_model.pt"
    ckpt_w = torch.load(ckpt_w_path, map_location="cpu", weights_only=True)
    best_wind_model.load_state_dict(ckpt_w["model_state_dict"])
    best_wind_model.eval()

    unscaler_w = lambda arr: test_ds_wind.inverse_transform_targets(arr)
    trainer_w = RegressionTrainer(
        model=best_wind_model,
        optimizer=torch.optim.Adam(best_wind_model.parameters()),
        device=torch.device("cpu"),
        experiment_id=best_wind_id,
        target_unscaler=unscaler_w,
    )
    y_true_w_test, y_pred_w_test = trainer_w.predict(test_loader_w)
    m_wind_test = calculate_regression_metrics(y_true_w_test, y_pred_w_test)

    # Power proxy evaluation on test set
    p_true_w_test = test_ds_wind.estimate_power(y_true_w_test)
    p_pred_w_test = test_ds_wind.estimate_power(y_pred_w_test)
    mp_wind_test = calculate_regression_metrics(p_true_w_test, p_pred_w_test)

    t_w_test_start = str(pd.Timestamp(test_ds_wind.target_timestamps[0]))
    t_w_test_end = str(pd.Timestamp(test_ds_wind.target_timestamps[-1]))

    print(f"Wind Test Speed: MAE = {m_wind_test['mae']:.4f} m/s | RMSE = {m_wind_test['rmse']:.4f} m/s | R^2 = {m_wind_test['r2']:.4f} (N={len(y_true_w_test):,})")
    print(f"Wind Test Power: MAE = {mp_wind_test['mae']:.2f} kW  | RMSE = {mp_wind_test['rmse']:.2f} kW  | R^2 = {mp_wind_test['r2']:.4f}")

    # Plot chronological test predictions for Wind
    plot_chronological_test_predictions(
        test_ds_wind.target_timestamps,
        y_true_w_test,
        y_pred_w_test,
        fig_dir / f"{best_wind_id}_chronological_test.png",
        title=f"{best_wind_id} - Chronological Test Wind Speed Predictions",
        unit="m/s",
        max_points=240,
    )

    # Log Wind Test to registry
    registry.log_experiment(
        experiment_id=f"{best_wind_id}_TEST",
        dataset="NREL_WIND_TOOLKIT",
        task="wind_speed_forecasting",
        model_name="WindLSTM",
        partition="test",
        features=f"{','.join(best_w_cfg.get('features_used', ['wind_speed']))} (24h past)",
        target="wind_speed (m/s)",
        sequence_length=best_w_cfg.get("sequence_length", 24),
        forecast_horizon=best_w_cfg.get("forecast_horizon", 1),
        n_samples=len(y_true_w_test),
        mae=m_wind_test["mae"],
        rmse=m_wind_test["rmse"],
        r2=m_wind_test["r2"],
        metric_unit="m/s",
        config=f"configs/experiments/{best_wind_id.lower().replace('exp_wind_dl_', 'wind_')}.yaml",
        random_seed=best_w_cfg.get("random_seed", 42),
        notes=f"Confirmatory test evaluation for {best_wind_id}. Period: {t_w_test_start} to {t_w_test_end}",
    )

    # 2. Evaluate Best Demand Model on Test Set
    print(f"\nEvaluating {best_demand_id} on held-out demand test set...")
    with open(f"configs/experiments/{best_demand_id.lower().replace('exp_demand_dl_', 'demand_')}.yaml") as f:
        best_d_cfg = yaml.safe_load(f)

    _, _, test_ds_dem, _, targ_scaler_d = build_demand_datasets(
        csv_path=PROJECT_ROOT / "data" / "processed" / "demand" / "demand_hourly_pilot.csv",
        feature_cols=best_d_cfg.get("features_used", ["demand_kwh"]),
        target_col="demand_kwh",
        time_col="datetime",
        sequence_length=best_d_cfg.get("sequence_length", 24),
        forecast_horizon=best_d_cfg.get("forecast_horizon", 1),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
    )
    test_loader_d = DataLoader(test_ds_dem, batch_size=32, shuffle=False, num_workers=0)

    best_demand_model = DemandLSTM(
        input_dim=best_d_cfg.get("input_dim", 1),
        hidden_dim=best_d_cfg.get("hidden_dim", 32),
        num_layers=best_d_cfg.get("num_layers", 1),
        dropout=best_d_cfg.get("dropout", 0.0),
    )
    ckpt_d_path = PROJECT_ROOT / "checkpoints" / best_demand_id / "best_model.pt"
    ckpt_d = torch.load(ckpt_d_path, map_location="cpu", weights_only=True)
    best_demand_model.load_state_dict(ckpt_d["model_state_dict"])
    best_demand_model.eval()

    unscaler_d = lambda arr: test_ds_dem.inverse_transform_targets(arr)
    trainer_d = RegressionTrainer(
        model=best_demand_model,
        optimizer=torch.optim.Adam(best_demand_model.parameters()),
        device=torch.device("cpu"),
        experiment_id=best_demand_id,
        target_unscaler=unscaler_d,
    )
    y_true_d_test, y_pred_d_test = trainer_d.predict(test_loader_d)
    m_demand_test = calculate_regression_metrics(y_true_d_test, y_pred_d_test)

    t_d_test_start = str(pd.Timestamp(test_ds_dem.target_timestamps[0]))
    t_d_test_end = str(pd.Timestamp(test_ds_dem.target_timestamps[-1]))

    print(f"Demand Test: MAE = {m_demand_test['mae']:.4f} kWh | RMSE = {m_demand_test['rmse']:.4f} kWh | R^2 = {m_demand_test['r2']:.4f} (N={len(y_true_d_test):,})")

    # Plot chronological test predictions for Demand
    plot_chronological_test_predictions(
        test_ds_dem.target_timestamps,
        y_true_d_test,
        y_pred_d_test,
        fig_dir / f"{best_demand_id}_chronological_test.png",
        title=f"{best_demand_id} - Chronological Test Demand Predictions",
        unit="kWh",
        max_points=240,
    )

    # Log Demand Test to registry
    registry.log_experiment(
        experiment_id=f"{best_demand_id}_TEST",
        dataset="BDG2",
        task="demand_forecasting",
        model_name="DemandLSTM",
        partition="test",
        features=f"{','.join(best_d_cfg.get('features_used', ['demand_kwh']))} (24h past)",
        target="demand_kwh (kWh)",
        sequence_length=best_d_cfg.get("sequence_length", 24),
        forecast_horizon=best_d_cfg.get("forecast_horizon", 1),
        n_samples=len(y_true_d_test),
        mae=m_demand_test["mae"],
        rmse=m_demand_test["rmse"],
        r2=m_demand_test["r2"],
        metric_unit="kWh",
        config=f"configs/experiments/{best_demand_id.lower().replace('exp_demand_dl_', 'demand_')}.yaml",
        random_seed=best_d_cfg.get("random_seed", 42),
        notes=f"Confirmatory test evaluation for {best_demand_id}. Period: {t_d_test_start} to {t_d_test_end}",
    )

    print("\n" + "=" * 80)
    print("STAGE 3 MODEL TRAINING, EVALUATION, AND BENCHMARKING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_stage3()
