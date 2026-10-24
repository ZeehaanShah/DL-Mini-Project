"""
Uncertainty quantification module for DL MiniProject.
Provides split conformal prediction, interval metrics, and coverage diagnostics.
"""
from src.uncertainty.conformal import (
    SplitConformalCalibrator,
    calculate_interval_metrics,
    evaluate_conformal_intervals,
)
from src.uncertainty.plotting import (
    plot_calibration_coverage_curve,
    plot_conformal_chronological_trajectory,
    plot_residual_calibration_distribution,
)

__all__ = [
    "SplitConformalCalibrator",
    "calculate_interval_metrics",
    "evaluate_conformal_intervals",
    "plot_calibration_coverage_curve",
    "plot_conformal_chronological_trajectory",
    "plot_residual_calibration_distribution",
]
