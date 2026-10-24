"""
Executable training script for Model A: SKIPP'D ResNet-18 Solar PV Nowcasting.

Usage:
    python scripts/train_solar.py --config configs/experiments/solar_s1.yaml
    python scripts/train_solar.py --config configs/experiments/solar_s3.yaml --epochs 5 --batch-size 32
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
from torch.utils.data import DataLoader, Subset

from src.data.datasets import build_skippd_datasets
from src.models.solar.resnet import ResNet18Nowcaster
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
    parser = argparse.ArgumentParser(description="Train ResNet-18 Solar PV Nowcaster")
    parser.add_argument("--config", type=str, default="configs/experiments/solar_s1.yaml", help="Path to YAML config")
    parser.add_argument("--experiment-id", type=str, default=None, help="Override experiment ID")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    parser.add_argument("--mode", type=str, default=None, choices=["feature_extractor", "fine_tune", "scratch"])
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train sample cap for quick checks")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_solar_model(args) -> dict:
    cfg = load_config(args.config)

    experiment_id = args.experiment_id or cfg.get("experiment_id", "EXP_SOLAR_DL")
    mode = args.mode or cfg.get("mode", "feature_extractor")
    lr = args.lr or cfg.get("learning_rate", 1e-3)
    dropout = args.dropout or cfg.get("dropout", 0.2)
    batch_size = args.batch_size or cfg.get("batch_size", 32)
    max_epochs = args.epochs or cfg.get("max_epochs", 8)
    patience = cfg.get("early_stopping_patience", 3)
    seed = args.seed if args.seed is not None else cfg.get("random_seed", 42)
    pretrained = cfg.get("pretrained", True)

    # Set deterministic random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"RUNNING SOLAR NOWCASTING EXPERIMENT: {experiment_id}")
    print(f"Mode: {mode} | Pretrained: {pretrained} | LR: {lr} | Dropout: {dropout}")
    print(f"Batch Size: {batch_size} | Max Epochs: {max_epochs} | Seed: {seed}")
    print("=" * 70)

    train_parquet = PROJECT_ROOT / "data" / "raw" / "skippd" / "train-00000-of-00005.parquet"
    test_parquet = PROJECT_ROOT / "data" / "raw" / "skippd" / "test-00000-of-00001.parquet"

    # Build datasets (calendar-day split, normalized ImageNet tensors)
    train_ds, val_ds, _ = build_skippd_datasets(
        train_parquet_path=train_parquet,
        test_parquet_path=test_parquet,
        train_ratio=0.8,
        chronological=True,
    )

    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        indices = list(range(args.max_train_samples))
        train_ds = Subset(train_ds, indices)
        print(f"Subsampled training set to {len(train_ds):,} samples.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: Train samples = {len(train_ds):,}, Val samples = {len(val_ds):,}")

    # Build ResNet-18 model
    model = ResNet18Nowcaster(
        pretrained=pretrained,
        mode=mode,
        dropout=dropout,
    )

    trainable_p, total_p = model.get_parameter_counts()
    print(f"Model Parameters: {trainable_p:,} trainable / {total_p:,} total ({(trainable_p/total_p*100):.2f}% trainable)")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # Trainer
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    trainer = RegressionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        experiment_id=experiment_id,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=patience,
    )

    train_summary = trainer.train(train_loader, val_loader, max_epochs=max_epochs, verbose=True)

    # Evaluate best model on validation set
    best_metrics = train_summary["best_metrics"]
    print("\n" + "=" * 70)
    print(f"EXPERIMENT {experiment_id} COMPLETED (Best Epoch: {train_summary['best_epoch']})")
    print(f"Val MAE:  {best_metrics['mae']:.4f} kW")
    print(f"Val RMSE: {best_metrics['rmse']:.4f} kW")
    print(f"Val R^2:  {best_metrics['r2']:.4f}")
    print("=" * 70)

    # Generate predictions for analysis and plotting
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
        plot_mae_curve(history, fig_dir / f"{experiment_id}_mae.png", title=f"{experiment_id} - Validation MAE Curve", unit="kW")
        plot_actual_vs_predicted_scatter(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_scatter.png", title=f"{experiment_id} - Actual vs Predicted", unit="kW")
        plot_residuals_vs_predicted(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_residuals.png", title=f"{experiment_id} - Residuals vs Predicted", unit="kW")
        plot_residual_distribution(y_true_val, y_pred_val, fig_dir / f"{experiment_id}_dist.png", title=f"{experiment_id} - Residual Distribution", unit="kW")
        print(f"Plots saved to {fig_dir}")

    # Log to experiment registry
    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")
    registry.log_experiment(
        experiment_id=experiment_id,
        dataset="SKIPP'D",
        task="solar_nowcasting",
        model_name=f"ResNet18_{mode}",
        partition="val",
        features="single_sky_image_64x64",
        target="pv (kW)",
        n_samples=len(y_val_ds := val_ds),
        mae=best_metrics["mae"],
        rmse=best_metrics["rmse"],
        r2=best_metrics["r2"],
        metric_unit="kW",
        config=args.config,
        random_seed=seed,
        notes=f"Best epoch: {train_summary['best_epoch']}/{max_epochs}, lr={lr}, dropout={dropout}, early_stopped={train_summary['early_stopped']}",
    )
    print(f"Logged to {PROJECT_ROOT / 'experiments' / 'experiment_registry.csv'}")

    return train_summary


if __name__ == "__main__":
    cli_args = parse_args()
    train_solar_model(cli_args)
