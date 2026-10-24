"""
Executable training script for Model B: NASA POWER Next-Hour Solar Irradiance Forecasting.

Usage:
    python scripts/train_nasa.py --config configs/experiments/nasa_w1.yaml
    python scripts/train_nasa.py --config configs/experiments/nasa_w3.yaml --epochs 15
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path and initialize Windows PyTorch DLLs first
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import src  # noqa: F401

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import build_nasa_datasets
from src.models.nasa_weather.sequence_models import WeatherLSTM
from src.training.trainer import RegressionTrainer
from src.training.plotting import (
    plot_loss_curve,
    plot_mae_curve,
    plot_actual_vs_predicted_scatter,
    plot_residuals_vs_predicted,
    plot_residual_distribution,
)
from src.evaluation.registry import ExperimentRegistry


def parse_args():
    parser = argparse.ArgumentParser(description="Train WeatherLSTM Irradiance Forecaster")
    parser.add_argument("--config", type=str, default="configs/experiments/nasa_w1.yaml", help="Path to YAML config")
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


def train_nasa_model(args) -> dict:
    cfg = load_config(args.config)

    experiment_id = args.experiment_id or cfg.get("experiment_id", "EXP_NASA_DL")
    hidden_dim = args.hidden_dim or cfg.get("hidden_dim", 64)
    num_layers = args.num_layers or cfg.get("num_layers", 1)
    dropout = args.dropout if args.dropout is not None else cfg.get("dropout", 0.0)
    lr = args.lr or cfg.get("learning_rate", 1e-3)
    batch_size = args.batch_size or cfg.get("batch_size", 32)
    max_epochs = args.epochs or cfg.get("max_epochs", 15)
    patience = cfg.get("early_stopping_patience", 4)
    seed = args.seed if args.seed is not None else cfg.get("random_seed", 42)
    seq_len = cfg.get("sequence_length", 24)
    horizon = cfg.get("forecast_horizon", 1)

    # Set deterministic random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"RUNNING NASA IRRADIANCE FORECASTING EXPERIMENT: {experiment_id}")
    print(f"LSTM: hidden={hidden_dim}, layers={num_layers}, dropout={dropout}, lr={lr}")
    print(f"Batch Size: {batch_size} | Max Epochs: {max_epochs} | Seed: {seed}")
    print("=" * 70)

    nasa_csv = PROJECT_ROOT / "data" / "raw" / "nasa_power" / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"

    # Build scaled datasets with 24-hour embargo
    train_ds, val_ds, test_ds, feat_scaler, targ_scaler = build_nasa_datasets(
        csv_path=nasa_csv,
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

    print(f"Dataset: Train windows = {len(train_ds):,}, Val windows = {len(val_ds):,}")

    # Build WeatherLSTM model
    model = WeatherLSTM(
        input_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    trainable_p, total_p = model.get_parameter_counts()
    print(f"Model Parameters: {trainable_p:,} trainable / {total_p:,} total")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Physical unscaler callable for validation evaluation in Wh/m²
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
    )

    train_summary = trainer.train(train_loader, val_loader, max_epochs=max_epochs, verbose=True)

    # Evaluate best model on validation set
    best_metrics = train_summary["best_metrics"]
    print("\n" + "=" * 70)
    print(f"EXPERIMENT {experiment_id} COMPLETED (Best Epoch: {train_summary['best_epoch']})")
    print(f"Val MAE:  {best_metrics['mae']:.2f} Wh/m^2")
    print(f"Val RMSE: {best_metrics['rmse']:.2f} Wh/m^2")
    print(f"Val R^2:  {best_metrics['r2']:.4f}")
    print("=" * 70)

    # Generate unscaled predictions for plotting and saving
    y_true_val, y_pred_val = trainer.predict(val_loader)

    # Save predictions
    exp_dir = PROJECT_ROOT / "experiments" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        exp_dir / "val_predictions.npz",
        y_true=y_true_val,
        y_pred=y_pred_val,
    )

    # Generate figures
    if not args.no_plot:
        fig_dir = PROJECT_ROOT / "reports" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        history = train_summary["history"]
        plot_loss_curve(history, fig_dir / f"{experiment_id}_loss.png", title=f"{experiment_id} - Loss Curve")
        plot_mae_curve(history, fig_dir / f"{experiment_id}_mae.png", title=f"{experiment_id} - Validation MAE Curve", unit="Wh/m^2")
        plot_actual_vs_predicted_scatter(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_scatter.png", title=f"{experiment_id} - Actual vs Predicted", unit="Wh/m^2")
        plot_residuals_vs_predicted(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_residuals.png", title=f"{experiment_id} - Residuals vs Predicted", unit="Wh/m^2")
        plot_residual_distribution(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_dist.png", title=f"{experiment_id} - Residual Distribution", unit="Wh/m^2")
        print(f"Plots saved to {fig_dir}")

    # Log to experiment registry
    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")
    registry.log_experiment(
        experiment_id=experiment_id,
        dataset="NASA_POWER",
        task="irradiance_forecasting",
        model_name="WeatherLSTM",
        partition="val",
        features="ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M (24h past)",
        target="ALLSKY_SFC_SW_DWN (Wh/m²)",
        sequence_length=seq_len,
        forecast_horizon=horizon,
        n_samples=len(val_ds),
        mae=best_metrics["mae"],
        rmse=best_metrics["rmse"],
        r2=best_metrics["r2"],
        metric_unit="Wh/m^2",
        config=args.config,
        random_seed=seed,
        notes=f"Best epoch: {train_summary['best_epoch']}/{max_epochs}, hidden={hidden_dim}, layers={num_layers}, lr={lr}, dropout={dropout}",
    )
    print(f"Logged to {PROJECT_ROOT / 'experiments' / 'experiment_registry.csv'}")

    return train_summary


if __name__ == "__main__":
    cli_args = parse_args()
    train_nasa_model(cli_args)
