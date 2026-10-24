"""
Plotting and visualization utilities for deep learning models and baselines.

Generates:
1. Training vs Validation loss curve by epoch.
2. Validation MAE curve by epoch.
3. Actual vs Predicted scatter plot with identity line.
4. Residuals vs Predicted plot with zero reference line.
5. Residual error distribution histogram.
6. Baseline vs Model MAE and RMSE comparative bar chart.
7. Hyperparameter configuration comparison chart.
8. Frozen-feature-extraction vs Fine-tuning comparative plot.
9. Chronological actual vs predicted time series plot on test set.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless servers and CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style() -> None:
    """Set clean visual style for academic report figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    })


def plot_loss_curve(
    history: List[Dict],
    output_path: Union[str, Path],
    title: str = "Training and Validation Loss",
) -> None:
    """Plot training vs validation loss across epochs."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss (MSE)", color="#1f77b4")
    ax.plot(epochs, val_loss, marker="s", linewidth=2, label="Validation Loss (MSE)", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mae_curve(
    history: List[Dict],
    output_path: Union[str, Path],
    title: str = "Validation MAE Across Epochs",
    unit: str = "kW",
) -> None:
    """Plot validation MAE progression across epochs."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, val_mae, marker="^", linewidth=2, color="#2ca02c", label=f"Validation MAE ({unit})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"MAE ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_actual_vs_predicted_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Actual vs Predicted",
    unit: str = "kW",
    max_scatter_points: int = 2000,
) -> None:
    """Plot actual vs predicted scatter with 1:1 identity line."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    # Subsample points if dataset is huge for clean rendering
    if len(yt) > max_scatter_points:
        idx = np.random.RandomState(42).choice(len(yt), size=max_scatter_points, replace=False)
        yt_plot = yt[idx]
        yp_plot = yp[idx]
    else:
        yt_plot, yp_plot = yt, yp

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt_plot, yp_plot, alpha=0.35, edgecolors="none", s=22, color="#1f77b4", label="Predictions")

    min_val = min(float(yt.min()), float(yp.min()))
    max_val = max(float(yt.max()), float(yp.max()))
    pad = (max_val - min_val) * 0.05
    ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], "r--", linewidth=1.8, label="Ideal (y = x)")

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)
    ax.set_xlabel(f"Ground Truth Target ({unit})")
    ax.set_ylabel(f"Model Prediction ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Residuals vs Predicted",
    unit: str = "kW",
    max_scatter_points: int = 2000,
) -> None:
    """Plot residuals (y_true - y_pred) against predicted values."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    residuals = yt - yp

    if len(yt) > max_scatter_points:
        idx = np.random.RandomState(42).choice(len(yt), size=max_scatter_points, replace=False)
        yp_plot = yp[idx]
        res_plot = residuals[idx]
    else:
        yp_plot, res_plot = yp, residuals

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(yp_plot, res_plot, alpha=0.35, edgecolors="none", s=22, color="#9467bd")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.set_xlabel(f"Predicted Value ({unit})")
    ax.set_ylabel(f"Residual [True - Pred] ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Residual Error Distribution",
    unit: str = "kW",
) -> None:
    """Plot histogram of residuals."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    residuals = yt - yp

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=40, density=True, alpha=0.7, color="#17becf", edgecolor="black", label="Residuals")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Mean Residual = 0")
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    ax.set_xlabel(f"Residual Error ({unit})")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"{title} (Mean: {mean_res:.3f}, Std: {std_res:.3f})", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_baseline_vs_model(
    comparisons: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
    title: str = "Baseline vs Deep Learning Model Performance",
    unit: str = "kW",
) -> None:
    """
    Bar chart comparing baselines and final neural model.

    Args:
        comparisons: Dict mapping model_label to {'mae': float, 'rmse': float}.
    """
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(comparisons.keys())
    maes = [comparisons[k]["mae"] for k in labels]
    rmses = [comparisons[k]["rmse"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, maes, width, label=f"MAE ({unit})", color="#2ca02c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, rmses, width, label=f"RMSE ({unit})", color="#1f77b4", alpha=0.85)

    ax.set_ylabel(f"Error ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True)

    # Attach value labels above bars
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_hyperparameter_comparison(
    configs: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Hyperparameter Experiment Comparison",
    unit: str = "kW",
) -> None:
    """Bar chart comparing validation metrics across the 4 controlled configurations."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [c["label"] for c in configs]
    maes = [c["mae"] for c in configs]
    rmses = [c["rmse"] for c in configs]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - width / 2, maes, width, label=f"Val MAE ({unit})", color="#ff7f0e", alpha=0.85)
    b2 = ax.bar(x + width / 2, rmses, width, label=f"Val RMSE ({unit})", color="#2ca02c", alpha=0.85)

    ax.set_ylabel(f"Error Metric ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True)

    for bar in list(b1) + list(b2):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_frozen_vs_finetuned(
    frozen_metrics: Dict[str, float],
    finetuned_metrics: Dict[str, float],
    output_path: Union[str, Path],
    title: str = "ResNet-18 Feature Extraction vs Fine-Tuning",
) -> None:
    """Comparative bar chart between frozen feature extraction and fine-tuning."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = ["Frozen Backbone (Feature Extraction)", "Layer 4 Unfrozen (Fine-Tuning)"]
    maes = [frozen_metrics["mae"], finetuned_metrics["mae"]]
    rmses = [frozen_metrics["rmse"], finetuned_metrics["rmse"]]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.8))
    b1 = ax.bar(x - width / 2, maes, width, label="Val MAE (kW)", color="#3498db", alpha=0.85)
    b2 = ax.bar(x + width / 2, rmses, width, label="Val RMSE (kW)", color="#e74c3c", alpha=0.85)

    ax.set_ylabel("Error (kW)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True)

    for bar in list(b1) + list(b2):
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f} kW",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_chronological_test_predictions(
    timestamps: Union[pd.DatetimeIndex, np.ndarray, List],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Chronological Actual vs Predicted (Test Set)",
    unit: str = "kW",
    max_points: int = 250,
) -> None:
    """Plot time-series chronological segment comparing predictions with ground truth."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))

    # Slice a representative continuous window if series is large
    n = min(len(yt), max_points)
    yt_slice = yt[:n]
    yp_slice = yp[:n]
    ts_slice = ts[:n]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ts_slice, yt_slice, label=f"Ground Truth ({unit})", color="#2c3e50", linewidth=1.8)
    ax.plot(ts_slice, yp_slice, label=f"Model Prediction ({unit})", color="#e67e22", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"Power / Irradiance ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
