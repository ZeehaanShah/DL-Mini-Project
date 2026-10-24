"""
Evaluate selected best models on official held-out test partitions.

Performs:
1. Final test evaluation for Model A (SKIPP'D test-00000-of-00001.parquet, 14,003 samples).
2. Final test evaluation for Model B (NASA POWER test split, 1,290 windows).
3. Generation of chronological actual-vs-predicted plots on the test set.
4. Logging test metrics to experiments/experiment_registry.csv.

Usage:
    python scripts/evaluate_test_set.py --solar-checkpoint checkpoints/EXP_SOLAR_DL_S3/best_model.pt --solar-mode fine_tune
    python scripts/evaluate_test_set.py --nasa-checkpoint checkpoints/EXP_NASA_DL_W3/best_model.pt --nasa-hidden-dim 64 --nasa-num-layers 2
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path and initialize Windows PyTorch DLLs first
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import src  # noqa: F401

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import build_skippd_datasets, build_nasa_datasets
from src.models.solar.resnet import ResNet18Nowcaster
from src.models.nasa_weather.sequence_models import WeatherLSTM
from src.training.trainer import RegressionTrainer
from src.training.plotting import plot_chronological_test_predictions
from src.evaluation.registry import ExperimentRegistry


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best models on held-out test sets")
    parser.add_argument("--solar-checkpoint", type=str, default=None, help="Path to best solar checkpoint .pt")
    parser.add_argument("--solar-mode", type=str, default="fine_tune", choices=["feature_extractor", "fine_tune", "scratch"])
    parser.add_argument("--nasa-checkpoint", type=str, default=None, help="Path to best NASA checkpoint .pt")
    parser.add_argument("--nasa-hidden-dim", type=int, default=64)
    parser.add_argument("--nasa-num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def evaluate_solar_test_set(checkpoint_path: Path, mode: str, batch_size: int = 32) -> dict:
    print("\n" + "=" * 70)
    print("EVALUATING MODEL A ON OFFICIAL HELD-OUT SKIPP'D TEST SET")
    print(f"Checkpoint: {checkpoint_path} | Mode: {mode}")
    print("=" * 70)

    train_parquet = PROJECT_ROOT / "data" / "raw" / "skippd" / "train-00000-of-00005.parquet"
    test_parquet = PROJECT_ROOT / "data" / "raw" / "skippd" / "test-00000-of-00001.parquet"

    _, _, test_ds = build_skippd_datasets(
        train_parquet_path=train_parquet,
        test_parquet_path=test_parquet,
        train_ratio=0.8,
        chronological=True,
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test Samples: {len(test_ds):,} across {len(set(test_ds.get_timestamps().dt.date))} calendar days")

    # Load model architecture and checkpoint weights
    model = ResNet18Nowcaster(pretrained=False, mode=mode)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    trainer = RegressionTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device("cpu"),
        experiment_id=checkpoint.get("experiment_id", "BEST_SOLAR_MODEL"),
    )

    test_metrics = trainer.evaluate(test_loader)
    print(f"Test MAE:  {test_metrics['mae']:.4f} kW")
    print(f"Test RMSE: {test_metrics['rmse']:.4f} kW")
    print(f"Test R^2:  {test_metrics['r2']:.4f}")

    y_true_test, y_pred_test = trainer.predict(test_loader)
    timestamps = test_ds.get_timestamps()

    # Generate chronological plot on test set
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_chronological_test_predictions(
        timestamps=timestamps,
        y_true=y_true_test,
        y_pred=y_pred_test,
        output_path=fig_dir / "solar_test_chronological.png",
        title=f"SKIPP'D Test Set Nowcasting: Actual vs ResNet-18 ({mode})",
        unit="kW",
    )
    print(f"Chronological test plot saved to {fig_dir / 'solar_test_chronological.png'}")

    # Log to registry under partition='test'
    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")
    exp_id = checkpoint.get("experiment_id", "EXP_SOLAR_DL_BEST") + "_TEST"
    registry.log_experiment(
        experiment_id=exp_id,
        dataset="SKIPP'D",
        task="solar_nowcasting",
        model_name=f"ResNet18_{mode}",
        partition="test",
        features="single_sky_image_64x64",
        target="pv (kW)",
        n_samples=len(test_ds),
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        r2=test_metrics["r2"],
        metric_unit="kW",
        notes=f"Held-out test set evaluation. Checkpoint from {checkpoint_path.name}",
    )
    return test_metrics


def evaluate_nasa_test_set(checkpoint_path: Path, hidden_dim: int, num_layers: int, batch_size: int = 32) -> dict:
    print("\n" + "=" * 70)
    print("EVALUATING MODEL B ON HELD-OUT NASA POWER TEST SET")
    print(f"Checkpoint: {checkpoint_path} | Hidden: {hidden_dim} | Layers: {num_layers}")
    print("=" * 70)

    nasa_csv = PROJECT_ROOT / "data" / "raw" / "nasa_power" / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"

    _, _, test_ds, _, _ = build_nasa_datasets(
        csv_path=nasa_csv,
        sequence_length=24,
        forecast_horizon=1,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test Windows: {len(test_ds):,}")

    model = WeatherLSTM(input_dim=4, hidden_dim=hidden_dim, num_layers=num_layers)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    unscaler = lambda arr: test_ds.inverse_transform_targets(arr)

    trainer = RegressionTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device("cpu"),
        experiment_id=checkpoint.get("experiment_id", "BEST_NASA_MODEL"),
        target_unscaler=unscaler,
    )

    test_metrics = trainer.evaluate(test_loader)
    print(f"Test MAE:  {test_metrics['mae']:.2f} Wh/m^2")
    print(f"Test RMSE: {test_metrics['rmse']:.2f} Wh/m^2")
    print(f"Test R^2:  {test_metrics['r2']:.4f}")

    y_true_test, y_pred_test = trainer.predict(test_loader)
    targets_ts = test_ds.target_timestamps

    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_chronological_test_predictions(
        timestamps=targets_ts,
        y_true=y_true_test,
        y_pred=y_pred_test,
        output_path=fig_dir / "nasa_test_chronological.png",
        title="NASA POWER Test Set Irradiance Forecasting: Actual vs WeatherLSTM",
        unit="Wh/m^2",
    )
    print(f"Chronological test plot saved to {fig_dir / 'nasa_test_chronological.png'}")

    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")
    exp_id = checkpoint.get("experiment_id", "EXP_NASA_DL_BEST") + "_TEST"
    registry.log_experiment(
        experiment_id=exp_id,
        dataset="NASA_POWER",
        task="irradiance_forecasting",
        model_name="WeatherLSTM",
        partition="test",
        features="ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M (24h past)",
        target="ALLSKY_SFC_SW_DWN (Wh/m²)",
        sequence_length=24,
        forecast_horizon=1,
        n_samples=len(test_ds),
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        r2=test_metrics["r2"],
        metric_unit="Wh/m^2",
        notes=f"Held-out test set evaluation. Checkpoint from {checkpoint_path.name}",
    )
    return test_metrics


def main():
    args = parse_args()

    if args.solar_checkpoint:
        evaluate_solar_test_set(Path(args.solar_checkpoint), args.solar_mode, args.batch_size)

    if args.nasa_checkpoint:
        evaluate_nasa_test_set(Path(args.nasa_checkpoint), args.nasa_hidden_dim, args.nasa_num_layers, args.batch_size)


if __name__ == "__main__":
    main()
