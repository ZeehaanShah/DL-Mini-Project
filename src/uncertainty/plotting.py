"""
Plotting and visualization utilities for Split Conformal Prediction.

Provides:
1. plot_calibration_coverage_curve: Calibration diagnostic plot comparing nominal vs empirical coverage.
2. plot_conformal_chronological_trajectory: Time-series actuals, point forecasts, and 80%/90% intervals.
3. plot_residual_calibration_distribution: Residual histogram with critical nonconformity quantile thresholds.
"""
from pathlib import Path
from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Headless backend
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


def plot_calibration_coverage_curve(
    nominal_levels: List[float],
    empirical_coverages_cal: List[float],
    empirical_coverages_test: Optional[List[float]],
    output_path: Union[str, Path],
    title: str = "Conformal Calibration Coverage Diagnostic",
) -> None:
    """
    Plot nominal coverage vs empirical coverage curve across multiple confidence levels.

    Args:
        nominal_levels: List of nominal coverage levels in [0.50, 0.99] (e.g. [0.50, ..., 0.95]).
        empirical_coverages_cal: Measured coverage on calibration/val set.
        empirical_coverages_test: Measured coverage on held-out test set.
        output_path: Destination image path.
        title: Plot title.
    """
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    nom_pct = [x * 100 for x in nominal_levels]
    cal_pct = [x * 100 for x in empirical_coverages_cal]

    # Perfect calibration 1:1 diagonal reference
    ax.plot([50, 100], [50, 100], linestyle="--", color="#7f8c8d", linewidth=1.8, label="Ideal Calibration (1:1)")

    # Calibration set empirical curve
    ax.plot(nom_pct, cal_pct, marker="o", color="#2980b9", linewidth=2.0, label="Calibration (Validation) Empirical")

    # Held-out test set empirical curve
    if empirical_coverages_test is not None:
        test_pct = [x * 100 for x in empirical_coverages_test]
        ax.plot(nom_pct, test_pct, marker="s", color="#e67e22", linewidth=2.0, label="Held-Out Test Empirical")

    ax.set_xlabel("Nominal Coverage Level (%)")
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(48, 100)
    ax.set_ylim(48, 102)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_conformal_chronological_trajectory(
    timestamps: Union[pd.DatetimeIndex, np.ndarray, List],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_80: np.ndarray,
    upper_80: np.ndarray,
    lower_90: np.ndarray,
    upper_90: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Chronological Conformal Prediction Intervals",
    unit: str = "m/s",
    max_points: int = 240,
) -> None:
    """
    Plot chronological trajectory showing ground truth, point forecast, and 80%/90% intervals.

    Args:
        timestamps: Series/array of timestamps.
        y_true: Ground truth observations.
        y_pred: Point forecasts.
        lower_80: Lower bound for 80% nominal interval.
        upper_80: Upper bound for 80% nominal interval.
        lower_90: Lower bound for 90% nominal interval.
        upper_90: Upper bound for 90% nominal interval.
        output_path: Destination image path.
        title: Plot title.
        unit: Physical measurement unit string.
        max_points: Number of consecutive timestamps to display for clarity.
    """
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    l80 = np.asarray(lower_80).ravel()
    u80 = np.asarray(upper_80).ravel()
    l90 = np.asarray(lower_90).ravel()
    u90 = np.asarray(upper_90).ravel()
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))

    n = min(len(yt), max_points)
    yt_s = yt[:n]
    yp_s = yp[:n]
    l80_s = l80[:n]
    u80_s = u80[:n]
    l90_s = l90[:n]
    u90_s = u90[:n]
    ts_s = ts[:n]

    fig, ax = plt.subplots(figsize=(12, 5.2))

    # Outer 90% band
    ax.fill_between(
        ts_s, l90_s, u90_s, color="#aed6f1", alpha=0.45, label="90% Conformal Interval"
    )
    # Inner 80% band
    ax.fill_between(
        ts_s, l80_s, u80_s, color="#5dade2", alpha=0.55, label="80% Conformal Interval"
    )

    # Point predictions and Ground truth
    ax.plot(ts_s, yp_s, color="#d35400", linestyle="--", linewidth=1.6, label=f"Point Forecast ({unit})")
    ax.plot(ts_s, yt_s, color="#1b2631", linewidth=1.8, label=f"Ground Truth ({unit})")

    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"Value ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")
    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_residual_calibration_distribution(
    residuals_cal: np.ndarray,
    q_80: float,
    q_90: float,
    output_path: Union[str, Path],
    title: str = "Calibration Nonconformity Scores",
    unit: str = "m/s",
) -> None:
    """
    Plot calibration absolute residual distribution with critical quantiles.
    """
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r = np.asarray(residuals_cal).ravel()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(r, bins=35, color="#34495e", alpha=0.75, edgecolor="white", density=True, label="Residual Density")
    ax.axvline(q_80, color="#2980b9", linestyle="--", linewidth=2.0, label=f"q (80%): {q_80:.3f} {unit}")
    ax.axvline(q_90, color="#c0392b", linestyle="-.", linewidth=2.0, label=f"q (90%): {q_90:.3f} {unit}")

    ax.set_xlabel(f"Absolute Nonconformity Score |y - y_hat| ({unit})")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
