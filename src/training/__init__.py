"""Training and plotting package for DL MiniProject."""
from .trainer import RegressionTrainer
from .plotting import (
    plot_loss_curve,
    plot_mae_curve,
    plot_actual_vs_predicted_scatter,
    plot_residuals_vs_predicted,
    plot_residual_distribution,
    plot_baseline_vs_model,
    plot_hyperparameter_comparison,
    plot_frozen_vs_finetuned,
    plot_chronological_test_predictions,
)

__all__ = [
    "RegressionTrainer",
    "plot_loss_curve",
    "plot_mae_curve",
    "plot_actual_vs_predicted_scatter",
    "plot_residuals_vs_predicted",
    "plot_residual_distribution",
    "plot_baseline_vs_model",
    "plot_hyperparameter_comparison",
    "plot_frozen_vs_finetuned",
    "plot_chronological_test_predictions",
]
