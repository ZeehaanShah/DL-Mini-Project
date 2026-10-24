"""
Executable training script for Model D: BDG2 Building Electricity Demand Forecasting (DemandLSTM).

Usage:
    python scripts/train_demand.py --config configs/experiments/demand_d1.yaml
    python scripts/train_demand.py --config configs/experiments/demand_d2.yaml --epochs 15
"""
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path and initialize Windows PyTorch DLLs first
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import src.utils.torch_loader

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.demand import build_demand_datasets
from src.models.demand.sequence_models import DemandLSTM
from src.training.trainer import RegressionTrainer
from src.training.plotting import (
    plot_loss_curve,
    plot_mae_curve,
    plot_actual_vs_predicted_scatter,
    plot_residuals_vs_predicted,
    plot_residual_distribution,
    plot_chronological_test_predictions,
)
from src.evaluation.registry import ExperimentRegistry
from src.evaluation.metrics import calculate_regression_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train DemandLSTM Electricity Forecaster")
    parser.add_argument("--config", type=str, default="configs/experiments/demand_d1.yaml", help="Path to YAML config")
    parser.add_argument("--experiment-id", type=str, default=None, help="Override experiment ID")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dim")
    parser.add_argument("--num-layers", type=int, default=None, help="Override num layers")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_demand_model(args) -> dict:
    cfg = load_config(args.config)

    experiment_id = args.experiment_id or cfg.get("experiment_id", "EXP_DEMAND_DL_D1")
    hidden_dim = args.hidden_dim or cfg.get("hidden_dim", 32)
    num_layers = args.num_layers or cfg.get("num_layers", 1)
    dropout = args.dropout if args.dropout is not None else cfg.get("dropout", 0.0)
    lr = args.lr or cfg.get("learning_rate", 1e-3)
    batch_size = args.batch_size or cfg.get("batch_size", 32)
    max_epochs = args.epochs or cfg.get("max_epochs", 15)
    patience = cfg.get("early_stopping_patience", 4)
    seed = args.seed if args.seed is not None else cfg.get("random_seed", 42)
    seq_len = cfg.get("sequence_length", 24)
    horizon = cfg.get("forecast_horizon", 1)
    input_dim = cfg.get("input_dim", 1)
    features_used = cfg.get("features_used", ["demand_kwh"])

    # Set deterministic random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 75)
    print(f"RUNNING MODEL D (DEMAND) EXPERIMENT: {experiment_id}")
    print(f"LSTM: input_dim={input_dim}, hidden={hidden_dim}, layers={num_layers}, dropout={dropout}, lr={lr}")
    print(f"Features: {features_used} -> Target: demand_kwh (kWh)")
    print(f"Batch Size: {batch_size} | Max Epochs: {max_epochs} | Seed: {seed}")
    print("=" * 75)

    demand_csv = PROJECT_ROOT / "data" / "processed" / "demand" / "demand_hourly_pilot.csv"

    # Build scaled datasets with 24-hour embargo
    train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_demand_datasets(
        csv_path=demand_csv,
        feature_cols=features_used,
        target_col="demand_kwh",
        time_col="datetime",
        sequence_length=seq_len,
        forecast_horizon=horizon,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    t_val_start = str(pd.Timestamp(val_ds.target_timestamps[0]))
    t_val_end = str(pd.Timestamp(val_ds.target_timestamps[-1]))
    print(f"Dataset: Train windows = {len(train_ds):,}, Val windows = {len(val_ds):,} ({t_val_start} to {t_val_end})")

    # Build DemandLSTM model
    model = DemandLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    trainable_p, total_p = model.get_parameter_counts()
    print(f"Model Parameters: {trainable_p:,} trainable / {total_p:,} total")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Physical unscaler callable for validation evaluation in kWh
    unscaler = lambda arr: val_ds.inverse_transform_targets(arr)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    trainer = RegressionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        experiment_id=experiment_id,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=patience,
        target_unscaler=unscaler,
        grad_clip_norm=5.0,
    )

    start_time = time.time()
    train_summary = trainer.train(train_loader, val_loader, max_epochs=max_epochs, verbose=True)
    runtime_sec = time.time() - start_time

    best_metrics = train_summary["best_metrics"]
    best_epoch = train_summary["best_epoch"]

    # Generate unscaled predictions for validation set
    y_true_val, y_pred_val = trainer.predict(val_loader)
    val_metrics = calculate_regression_metrics(y_true_val, y_pred_val)

    print("\n" + "=" * 75)
    print(f"EXPERIMENT {experiment_id} COMPLETED in {runtime_sec:.1f}s (Best Epoch: {best_epoch})")
    print(f"Val Demand: MAE: {val_metrics['mae']:.4f} kWh | RMSE: {val_metrics['rmse']:.4f} kWh | R^2: {val_metrics['r2']:.4f}")
    print("=" * 75)

    # Save validation predictions
    exp_dir = PROJECT_ROOT / "experiments" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        exp_dir / "val_predictions.npz",
        y_true=y_true_val,
        y_pred=y_pred_val,
        timestamps=val_ds.target_timestamps,
    )

    # Generate figures
    if not args.no_plot:
        fig_dir = PROJECT_ROOT / "reports" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        history = train_summary["history"]
        plot_loss_curve(history, fig_dir / f"{experiment_id}_loss.png", title=f"{experiment_id} - Train & Val Loss")
        plot_mae_curve(history, fig_dir / f"{experiment_id}_mae.png", title=f"{experiment_id} - Validation MAE Progression", unit="kWh")
        plot_actual_vs_predicted_scatter(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_scatter.png", title=f"{experiment_id} - Electricity Demand Actual vs Predicted", unit="kWh")
        plot_residuals_vs_predicted(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_residuals.png", title=f"{experiment_id} - Residuals vs Predicted", unit="kWh")
        plot_residual_distribution(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_dist.png", title=f"{experiment_id} - Error Distribution", unit="kWh")
        plot_chronological_test_predictions(
            val_ds.target_timestamps,
            y_true_val,
            y_pred_val,
            fig_dir / f"{experiment_id}_timeseries_val.png",
            title=f"{experiment_id} - Chronological Validation Trajectory",
            unit="kWh",
            max_points=240,
        )
        print(f"Diagnostic plots saved to {fig_dir}")

    # Log to experiment registry
    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")
    registry.log_experiment(
        experiment_id=experiment_id,
        dataset="BDG2",
        task="demand_forecasting",
        model_name="DemandLSTM",
        partition="val",
        features=f"{','.join(features_used)} ({seq_len}h past)",
        target="demand_kwh (kWh)",
        sequence_length=seq_len,
        forecast_horizon=horizon,
        n_samples=len(val_ds),
        mae=val_metrics["mae"],
        rmse=val_metrics["rmse"],
        r2=val_metrics["r2"],
        metric_unit="kWh",
        config=args.config,
        random_seed=seed,
        notes=f"Best epoch: {best_epoch}/{max_epochs}, hidden={hidden_dim}, layers={num_layers}, lr={lr}, dropout={dropout}, runtime={runtime_sec:.1f}s",
    )
    print(f"Logged to {PROJECT_ROOT / 'experiments' / 'experiment_registry.csv'}")

    return {
        "experiment_id": experiment_id,
        "best_epoch": best_epoch,
        "runtime_sec": runtime_sec,
        "val_metrics": val_metrics,
    }


if __name__ == "__main__":
    cli_args = parse_args()
    train_demand_model(cli_args)
