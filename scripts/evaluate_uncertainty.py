"""
Orchestration script for Stage 4: Leakage-Safe Uncertainty Estimation.

Performs:
1. Loads frozen checkpoints for EXP_WIND_DL_C2 (Wind) and EXP_DEMAND_DL_D2 (Demand).
2. Uses dedicated chronological calibration partitions (Validation splits) strictly preceding test partitions.
3. Fits SplitConformalCalibrator with finite-sample adjusted nonconformity quantiles.
4. Generates 80% and 90% prediction intervals on held-out test partitions.
5. Applies and analyzes physically justified non-negative clipping (>= 0 m/s for wind, >= 0 kWh for demand).
6. Converts wind speed intervals to estimated turbine electrical power proxy intervals (in kW).
7. Evaluates empirical coverage, Mean Prediction Interval Width (MPIW), and Winkler interval scores.
8. Generates calibration diagnostic curves, chronological trajectory plots, and residual distributions.
9. Logs uncertainty experiments to experiments/experiment_registry.csv.
"""
import json
from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path and initialize Windows PyTorch DLLs first
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import src.utils.torch_loader

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.wind import build_wind_datasets, TurbinePowerCurve
from src.data.demand import build_demand_datasets
from src.models.wind.sequence_models import WindLSTM
from src.models.demand.sequence_models import DemandLSTM
from src.training.trainer import RegressionTrainer
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
from src.evaluation.registry import ExperimentRegistry


def run_wind_uncertainty(fig_dir: Path, registry: ExperimentRegistry) -> dict:
    print("\n" + "=" * 80)
    print("STAGE 4: UNCERTAINTY ESTIMATION FOR MODEL C (EXP_WIND_DL_C2)")
    print("=" * 80)

    cfg_path = PROJECT_ROOT / "configs" / "experiments" / "wind_c2.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    curve = TurbinePowerCurve()
    csv_path = PROJECT_ROOT / "data" / "processed" / "wind" / "wind_hourly_pilot.csv"

    # Build datasets with strict chronological splits and 24h embargos
    train_ds, cal_ds, test_ds, _, _ = build_wind_datasets(
        csv_path=csv_path,
        feature_cols=cfg.get("features_used", ["wind_speed"]),
        target_col="wind_speed",
        time_col="datetime",
        sequence_length=cfg.get("sequence_length", 24),
        forecast_horizon=cfg.get("forecast_horizon", 1),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
        power_curve=curve,
    )

    cal_loader = DataLoader(cal_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Document exact timestamps and counts
    t_cal_start = str(pd.Timestamp(cal_ds.target_timestamps[0]))
    t_cal_end = str(pd.Timestamp(cal_ds.target_timestamps[-1]))
    t_test_start = str(pd.Timestamp(test_ds.target_timestamps[0]))
    t_test_end = str(pd.Timestamp(test_ds.target_timestamps[-1]))

    print(f"Calibration partition (Validation): {len(cal_ds)} samples ({t_cal_start} to {t_cal_end})")
    print(f"Held-out Test partition:            {len(test_ds)} samples ({t_test_start} to {t_test_end})")
    print("Temporal ordering: Train < Embargo (24h) < Calibration < Embargo (24h) < Test. Zero leakage.")

    # Load frozen WindLSTM checkpoint
    ckpt_path = PROJECT_ROOT / "checkpoints" / "EXP_WIND_DL_C2" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Required checkpoint not found: {ckpt_path}")

    model = WindLSTM(
        input_dim=cfg.get("input_dim", 1),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.1),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    unscaler = lambda arr: cal_ds.inverse_transform_targets(arr)
    trainer = RegressionTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device("cpu"),
        experiment_id="EXP_WIND_DL_C2",
        target_unscaler=unscaler,
    )

    # Predictions in original physical units (m/s)
    y_true_cal, y_pred_cal = trainer.predict(cal_loader)
    y_true_test, y_pred_test = trainer.predict(test_loader)

    # 1. Fit Conformal Calibrator on Calibration Set
    calibrator = SplitConformalCalibrator()
    calibrator.fit(y_true_cal, y_pred_cal)
    q_80 = calibrator.compute_quantile(alpha=0.20)
    q_90 = calibrator.compute_quantile(alpha=0.10)

    print(f"Nonconformity score quantiles: q(80%) = {q_80:.4f} m/s | q(90%) = {q_90:.4f} m/s")

    # Diagnostic grid for coverage curve
    grid_alphas = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
    nom_levels = [round(1.0 - a, 4) for a in grid_alphas]
    cal_coverages = []
    test_coverages = []
    for a in grid_alphas:
        l_c, u_c = calibrator.predict_interval(y_pred_cal, alpha=a, clip_min=0.0)
        m_c = calculate_interval_metrics(y_true_cal, l_c, u_c, alpha=a)
        cal_coverages.append(m_c["coverage"])

        l_t, u_t = calibrator.predict_interval(y_pred_test, alpha=a, clip_min=0.0)
        m_t = calculate_interval_metrics(y_true_test, l_t, u_t, alpha=a)
        test_coverages.append(m_t["coverage"])

    # 2. Evaluate 80% and 90% Conformal Prediction Intervals on Test Set
    eval_80 = evaluate_conformal_intervals(
        calibrator=calibrator,
        y_true=y_true_test,
        y_pred=y_pred_test,
        alpha=0.20,
        clip_min=0.0,
    )
    eval_90 = evaluate_conformal_intervals(
        calibrator=calibrator,
        y_true=y_true_test,
        y_pred=y_pred_test,
        alpha=0.10,
        clip_min=0.0,
    )

    print("\n--- Model C Wind Speed Interval Evaluation (Test Set) ---")
    print(f"80% Nominal: Coverage = {eval_80['clipped_metrics']['coverage']*100:.2f}% | "
          f"MPIW = {eval_80['clipped_metrics']['mpiw']:.4f} m/s (Raw MPIW: {eval_80['raw_metrics']['mpiw']:.4f} m/s, "
          f"Width Reduction: {eval_80['clipping_effect']['width_reduction']:.4f} m/s) | "
          f"Winkler Score = {eval_80['clipped_metrics']['winkler_score']:.4f}")
    print(f"90% Nominal: Coverage = {eval_90['clipped_metrics']['coverage']*100:.2f}% | "
          f"MPIW = {eval_90['clipped_metrics']['mpiw']:.4f} m/s (Raw MPIW: {eval_90['raw_metrics']['mpiw']:.4f} m/s, "
          f"Width Reduction: {eval_90['clipping_effect']['width_reduction']:.4f} m/s) | "
          f"Winkler Score = {eval_90['clipped_metrics']['winkler_score']:.4f}")

    # 3. Power Proxy Interval Evaluation (in kW via TurbinePowerCurve)
    p_true_test = curve.wind_to_power(y_true_test)
    p_pred_test = curve.wind_to_power(y_pred_test)

    # Convert speed bounds to power bounds
    p_lower_80 = curve.wind_to_power(eval_80["lower_bounds"])
    p_upper_80 = curve.wind_to_power(eval_80["upper_bounds"])
    m_p_80 = calculate_interval_metrics(p_true_test, p_lower_80, p_upper_80, alpha=0.20)

    p_lower_90 = curve.wind_to_power(eval_90["lower_bounds"])
    p_upper_90 = curve.wind_to_power(eval_90["upper_bounds"])
    m_p_90 = calculate_interval_metrics(p_true_test, p_lower_90, p_upper_90, alpha=0.10)

    print("\n--- Model C Turbine Power Proxy Intervals (Estimated kW) ---")
    print(f"80% Power: Coverage = {m_p_80['coverage']*100:.2f}% | MPIW = {m_p_80['mpiw']:.2f} kW | Winkler = {m_p_80['winkler_score']:.2f} kW")
    print(f"90% Power: Coverage = {m_p_90['coverage']*100:.2f}% | MPIW = {m_p_90['mpiw']:.2f} kW | Winkler = {m_p_90['winkler_score']:.2f} kW")

    # 4. Generate Visualizations
    plot_calibration_coverage_curve(
        nominal_levels=nom_levels,
        empirical_coverages_cal=cal_coverages,
        empirical_coverages_test=test_coverages,
        output_path=fig_dir / "wind_conformal_calibration_curve.png",
        title="Model C (WindLSTM) - Split Conformal Calibration Diagnostic",
    )
    plot_conformal_chronological_trajectory(
        timestamps=test_ds.target_timestamps,
        y_true=y_true_test,
        y_pred=y_pred_test,
        lower_80=eval_80["lower_bounds"],
        upper_80=eval_80["upper_bounds"],
        lower_90=eval_90["lower_bounds"],
        upper_90=eval_90["upper_bounds"],
        output_path=fig_dir / "wind_conformal_intervals_test.png",
        title="Model C (WindLSTM) - Held-Out Test Prediction Intervals (m/s)",
        unit="m/s",
        max_points=240,
    )
    plot_conformal_chronological_trajectory(
        timestamps=test_ds.target_timestamps,
        y_true=p_true_test,
        y_pred=p_pred_test,
        lower_80=p_lower_80,
        upper_80=p_upper_80,
        lower_90=p_lower_90,
        upper_90=p_upper_90,
        output_path=fig_dir / "wind_power_conformal_intervals_test.png",
        title="Model C (WindLSTM) - Estimated Turbine Power Proxy Intervals (kW)",
        unit="kW",
        max_points=240,
    )
    plot_residual_calibration_distribution(
        residuals_cal=calibrator.residuals,
        q_80=q_80,
        q_90=q_90,
        output_path=fig_dir / "wind_conformal_residuals.png",
        title="Model C (WindLSTM) - Calibration Nonconformity Scores",
        unit="m/s",
    )

    # 5. Log to Experiment Registry
    registry.log_experiment(
        experiment_id="EXP_WIND_CP_80",
        dataset="NREL_WIND_TOOLKIT",
        task="uncertainty_estimation",
        model_name="SplitConformal_WindLSTM",
        partition="test",
        features="wind_speed (24h past)",
        target="wind_speed (m/s)",
        sequence_length=cfg.get("sequence_length", 24),
        forecast_horizon=cfg.get("forecast_horizon", 1),
        n_samples=len(y_true_test),
        mae=eval_80["clipped_metrics"]["mpiw"],
        rmse=eval_80["clipped_metrics"]["winkler_score"],
        r2=eval_80["clipped_metrics"]["coverage"],
        metric_unit="m/s",
        config=str(cfg_path),
        random_seed=cfg.get("random_seed", 42),
        notes=(
            f"Nominal 80% split conformal interval. Empirical cov={eval_80['clipped_metrics']['coverage']*100:.2f}%, "
            f"MPIW={eval_80['clipped_metrics']['mpiw']:.4f} m/s, Winkler={eval_80['clipped_metrics']['winkler_score']:.4f}, "
            f"q={q_80:.4f} m/s. Clipped >= 0 m/s (width red: {eval_80['clipping_effect']['width_reduction']:.4f} m/s). "
            f"Cal N={len(cal_ds)} ({t_cal_start} to {t_cal_end})"
        ),
    )
    registry.log_experiment(
        experiment_id="EXP_WIND_CP_90",
        dataset="NREL_WIND_TOOLKIT",
        task="uncertainty_estimation",
        model_name="SplitConformal_WindLSTM",
        partition="test",
        features="wind_speed (24h past)",
        target="wind_speed (m/s)",
        sequence_length=cfg.get("sequence_length", 24),
        forecast_horizon=cfg.get("forecast_horizon", 1),
        n_samples=len(y_true_test),
        mae=eval_90["clipped_metrics"]["mpiw"],
        rmse=eval_90["clipped_metrics"]["winkler_score"],
        r2=eval_90["clipped_metrics"]["coverage"],
        metric_unit="m/s",
        config=str(cfg_path),
        random_seed=cfg.get("random_seed", 42),
        notes=(
            f"Nominal 90% split conformal interval. Empirical cov={eval_90['clipped_metrics']['coverage']*100:.2f}%, "
            f"MPIW={eval_90['clipped_metrics']['mpiw']:.4f} m/s, Winkler={eval_90['clipped_metrics']['winkler_score']:.4f}, "
            f"q={q_90:.4f} m/s. Clipped >= 0 m/s (width red: {eval_90['clipping_effect']['width_reduction']:.4f} m/s). "
            f"Cal N={len(cal_ds)} ({t_cal_start} to {t_cal_end})"
        ),
    )

    return {
        "cal_samples": len(cal_ds),
        "test_samples": len(test_ds),
        "t_cal_start": t_cal_start,
        "t_cal_end": t_cal_end,
        "t_test_start": t_test_start,
        "t_test_end": t_test_end,
        "q_80": q_80,
        "q_90": q_90,
        "eval_80": eval_80,
        "eval_90": eval_90,
        "power_80": m_p_80,
        "power_90": m_p_90,
    }


def run_demand_uncertainty(fig_dir: Path, registry: ExperimentRegistry) -> dict:
    print("\n" + "=" * 80)
    print("STAGE 4: UNCERTAINTY ESTIMATION FOR MODEL D (EXP_DEMAND_DL_D2)")
    print("=" * 80)

    cfg_path = PROJECT_ROOT / "configs" / "experiments" / "demand_d2.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    csv_path = PROJECT_ROOT / "data" / "processed" / "demand" / "demand_hourly_pilot.csv"

    # Build datasets with strict chronological splits and 24h embargos
    train_ds, cal_ds, test_ds, _, _ = build_demand_datasets(
        csv_path=csv_path,
        feature_cols=cfg.get("features_used", ["demand_kwh"]),
        target_col="demand_kwh",
        time_col="datetime",
        sequence_length=cfg.get("sequence_length", 24),
        forecast_horizon=cfg.get("forecast_horizon", 1),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
    )

    cal_loader = DataLoader(cal_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    t_cal_start = str(pd.Timestamp(cal_ds.target_timestamps[0]))
    t_cal_end = str(pd.Timestamp(cal_ds.target_timestamps[-1]))
    t_test_start = str(pd.Timestamp(test_ds.target_timestamps[0]))
    t_test_end = str(pd.Timestamp(test_ds.target_timestamps[-1]))

    print(f"Calibration partition (Validation): {len(cal_ds)} samples ({t_cal_start} to {t_cal_end})")
    print(f"Held-out Test partition:            {len(test_ds)} samples ({t_test_start} to {t_test_end})")
    print("Temporal ordering: Train < Embargo (24h) < Calibration < Embargo (24h) < Test. Zero leakage.")

    # Load frozen DemandLSTM checkpoint
    ckpt_path = PROJECT_ROOT / "checkpoints" / "EXP_DEMAND_DL_D2" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Required checkpoint not found: {ckpt_path}")

    model = DemandLSTM(
        input_dim=cfg.get("input_dim", 1),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.1),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    unscaler = lambda arr: cal_ds.inverse_transform_targets(arr)
    trainer = RegressionTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device("cpu"),
        experiment_id="EXP_DEMAND_DL_D2",
        target_unscaler=unscaler,
    )

    # Predictions in original physical units (kWh)
    y_true_cal, y_pred_cal = trainer.predict(cal_loader)
    y_true_test, y_pred_test = trainer.predict(test_loader)

    # 1. Fit Conformal Calibrator on Calibration Set
    calibrator = SplitConformalCalibrator()
    calibrator.fit(y_true_cal, y_pred_cal)
    q_80 = calibrator.compute_quantile(alpha=0.20)
    q_90 = calibrator.compute_quantile(alpha=0.10)

    print(f"Nonconformity score quantiles: q(80%) = {q_80:.4f} kWh | q(90%) = {q_90:.4f} kWh")

    # Diagnostic grid for coverage curve
    grid_alphas = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
    nom_levels = [round(1.0 - a, 4) for a in grid_alphas]
    cal_coverages = []
    test_coverages = []
    for a in grid_alphas:
        l_c, u_c = calibrator.predict_interval(y_pred_cal, alpha=a, clip_min=0.0)
        m_c = calculate_interval_metrics(y_true_cal, l_c, u_c, alpha=a)
        cal_coverages.append(m_c["coverage"])

        l_t, u_t = calibrator.predict_interval(y_pred_test, alpha=a, clip_min=0.0)
        m_t = calculate_interval_metrics(y_true_test, l_t, u_t, alpha=a)
        test_coverages.append(m_t["coverage"])

    # 2. Evaluate 80% and 90% Conformal Prediction Intervals on Test Set
    eval_80 = evaluate_conformal_intervals(
        calibrator=calibrator,
        y_true=y_true_test,
        y_pred=y_pred_test,
        alpha=0.20,
        clip_min=0.0,
    )
    eval_90 = evaluate_conformal_intervals(
        calibrator=calibrator,
        y_true=y_true_test,
        y_pred=y_pred_test,
        alpha=0.10,
        clip_min=0.0,
    )

    print("\n--- Model D Electricity Demand Interval Evaluation (Test Set) ---")
    print(f"80% Nominal: Coverage = {eval_80['clipped_metrics']['coverage']*100:.2f}% | "
          f"MPIW = {eval_80['clipped_metrics']['mpiw']:.4f} kWh (Raw MPIW: {eval_80['raw_metrics']['mpiw']:.4f} kWh, "
          f"Width Reduction: {eval_80['clipping_effect']['width_reduction']:.4f} kWh) | "
          f"Winkler Score = {eval_80['clipped_metrics']['winkler_score']:.4f}")
    print(f"90% Nominal: Coverage = {eval_90['clipped_metrics']['coverage']*100:.2f}% | "
          f"MPIW = {eval_90['clipped_metrics']['mpiw']:.4f} kWh (Raw MPIW: {eval_90['raw_metrics']['mpiw']:.4f} kWh, "
          f"Width Reduction: {eval_90['clipping_effect']['width_reduction']:.4f} kWh) | "
          f"Winkler Score = {eval_90['clipped_metrics']['winkler_score']:.4f}")

    # 3. Generate Visualizations
    plot_calibration_coverage_curve(
        nominal_levels=nom_levels,
        empirical_coverages_cal=cal_coverages,
        empirical_coverages_test=test_coverages,
        output_path=fig_dir / "demand_conformal_calibration_curve.png",
        title="Model D (DemandLSTM) - Split Conformal Calibration Diagnostic",
    )
    plot_conformal_chronological_trajectory(
        timestamps=test_ds.target_timestamps,
        y_true=y_true_test,
        y_pred=y_pred_test,
        lower_80=eval_80["lower_bounds"],
        upper_80=eval_80["upper_bounds"],
        lower_90=eval_90["lower_bounds"],
        upper_90=eval_90["upper_bounds"],
        output_path=fig_dir / "demand_conformal_intervals_test.png",
        title="Model D (DemandLSTM) - Held-Out Test Prediction Intervals (kWh)",
        unit="kWh",
        max_points=240,
    )
    plot_residual_calibration_distribution(
        residuals_cal=calibrator.residuals,
        q_80=q_80,
        q_90=q_90,
        output_path=fig_dir / "demand_conformal_residuals.png",
        title="Model D (DemandLSTM) - Calibration Nonconformity Scores",
        unit="kWh",
    )

    # 4. Log to Experiment Registry
    registry.log_experiment(
        experiment_id="EXP_DEMAND_CP_80",
        dataset="BDG2",
        task="uncertainty_estimation",
        model_name="SplitConformal_DemandLSTM",
        partition="test",
        features="demand_kwh (24h past)",
        target="demand_kwh (kWh)",
        sequence_length=cfg.get("sequence_length", 24),
        forecast_horizon=cfg.get("forecast_horizon", 1),
        n_samples=len(y_true_test),
        mae=eval_80["clipped_metrics"]["mpiw"],
        rmse=eval_80["clipped_metrics"]["winkler_score"],
        r2=eval_80["clipped_metrics"]["coverage"],
        metric_unit="kWh",
        config=str(cfg_path),
        random_seed=cfg.get("random_seed", 42),
        notes=(
            f"Nominal 80% split conformal interval. Empirical cov={eval_80['clipped_metrics']['coverage']*100:.2f}%, "
            f"MPIW={eval_80['clipped_metrics']['mpiw']:.4f} kWh, Winkler={eval_80['clipped_metrics']['winkler_score']:.4f}, "
            f"q={q_80:.4f} kWh. Clipped >= 0 kWh (width red: {eval_80['clipping_effect']['width_reduction']:.4f} kWh). "
            f"Cal N={len(cal_ds)} ({t_cal_start} to {t_cal_end})"
        ),
    )
    registry.log_experiment(
        experiment_id="EXP_DEMAND_CP_90",
        dataset="BDG2",
        task="uncertainty_estimation",
        model_name="SplitConformal_DemandLSTM",
        partition="test",
        features="demand_kwh (24h past)",
        target="demand_kwh (kWh)",
        sequence_length=cfg.get("sequence_length", 24),
        forecast_horizon=cfg.get("forecast_horizon", 1),
        n_samples=len(y_true_test),
        mae=eval_90["clipped_metrics"]["mpiw"],
        rmse=eval_90["clipped_metrics"]["winkler_score"],
        r2=eval_90["clipped_metrics"]["coverage"],
        metric_unit="kWh",
        config=str(cfg_path),
        random_seed=cfg.get("random_seed", 42),
        notes=(
            f"Nominal 90% split conformal interval. Empirical cov={eval_90['clipped_metrics']['coverage']*100:.2f}%, "
            f"MPIW={eval_90['clipped_metrics']['mpiw']:.4f} kWh, Winkler={eval_90['clipped_metrics']['winkler_score']:.4f}, "
            f"q={q_90:.4f} kWh. Clipped >= 0 kWh (width red: {eval_90['clipping_effect']['width_reduction']:.4f} kWh). "
            f"Cal N={len(cal_ds)} ({t_cal_start} to {t_cal_end})"
        ),
    )

    return {
        "cal_samples": len(cal_ds),
        "test_samples": len(test_ds),
        "t_cal_start": t_cal_start,
        "t_cal_end": t_cal_end,
        "t_test_start": t_test_start,
        "t_test_end": t_test_end,
        "q_80": q_80,
        "q_90": q_90,
        "eval_80": eval_80,
        "eval_90": eval_90,
    }


def main():
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = PROJECT_ROOT / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")

    wind_res = run_wind_uncertainty(fig_dir, registry)
    demand_res = run_demand_uncertainty(fig_dir, registry)

    # Save structured summary to reports/tables/uncertainty_summary.json
    summary = {
        "wind": {
            "model_id": "EXP_WIND_DL_C2",
            "cal_samples": wind_res["cal_samples"],
            "test_samples": wind_res["test_samples"],
            "calibration_period": [wind_res["t_cal_start"], wind_res["t_cal_end"]],
            "test_period": [wind_res["t_test_start"], wind_res["t_test_end"]],
            "q_80_ms": wind_res["q_80"],
            "q_90_ms": wind_res["q_90"],
            "eval_80": {
                "nominal_coverage": 0.80,
                "empirical_coverage": wind_res["eval_80"]["clipped_metrics"]["coverage"],
                "mpiw_ms": wind_res["eval_80"]["clipped_metrics"]["mpiw"],
                "raw_mpiw_ms": wind_res["eval_80"]["raw_metrics"]["mpiw"],
                "width_reduction_ms": wind_res["eval_80"]["clipping_effect"]["width_reduction"],
                "winkler_score": wind_res["eval_80"]["clipped_metrics"]["winkler_score"],
                "undercoverage_rate": wind_res["eval_80"]["clipped_metrics"]["undercoverage_rate"],
                "overcoverage_rate": wind_res["eval_80"]["clipped_metrics"]["overcoverage_rate"],
            },
            "eval_90": {
                "nominal_coverage": 0.90,
                "empirical_coverage": wind_res["eval_90"]["clipped_metrics"]["coverage"],
                "mpiw_ms": wind_res["eval_90"]["clipped_metrics"]["mpiw"],
                "raw_mpiw_ms": wind_res["eval_90"]["raw_metrics"]["mpiw"],
                "width_reduction_ms": wind_res["eval_90"]["clipping_effect"]["width_reduction"],
                "winkler_score": wind_res["eval_90"]["clipped_metrics"]["winkler_score"],
                "undercoverage_rate": wind_res["eval_90"]["clipped_metrics"]["undercoverage_rate"],
                "overcoverage_rate": wind_res["eval_90"]["clipped_metrics"]["overcoverage_rate"],
            },
            "power_proxy": {
                "power_80": wind_res["power_80"],
                "power_90": wind_res["power_90"],
            },
        },
        "demand": {
            "model_id": "EXP_DEMAND_DL_D2",
            "cal_samples": demand_res["cal_samples"],
            "test_samples": demand_res["test_samples"],
            "calibration_period": [demand_res["t_cal_start"], demand_res["t_cal_end"]],
            "test_period": [demand_res["t_test_start"], demand_res["t_test_end"]],
            "q_80_kwh": demand_res["q_80"],
            "q_90_kwh": demand_res["q_90"],
            "eval_80": {
                "nominal_coverage": 0.80,
                "empirical_coverage": demand_res["eval_80"]["clipped_metrics"]["coverage"],
                "mpiw_kwh": demand_res["eval_80"]["clipped_metrics"]["mpiw"],
                "raw_mpiw_kwh": demand_res["eval_80"]["raw_metrics"]["mpiw"],
                "width_reduction_kwh": demand_res["eval_80"]["clipping_effect"]["width_reduction"],
                "winkler_score": demand_res["eval_80"]["clipped_metrics"]["winkler_score"],
                "undercoverage_rate": demand_res["eval_80"]["clipped_metrics"]["undercoverage_rate"],
                "overcoverage_rate": demand_res["eval_80"]["clipped_metrics"]["overcoverage_rate"],
            },
            "eval_90": {
                "nominal_coverage": 0.90,
                "empirical_coverage": demand_res["eval_90"]["clipped_metrics"]["coverage"],
                "mpiw_kwh": demand_res["eval_90"]["clipped_metrics"]["mpiw"],
                "raw_mpiw_kwh": demand_res["eval_90"]["raw_metrics"]["mpiw"],
                "width_reduction_kwh": demand_res["eval_90"]["clipping_effect"]["width_reduction"],
                "winkler_score": demand_res["eval_90"]["clipped_metrics"]["winkler_score"],
                "undercoverage_rate": demand_res["eval_90"]["clipped_metrics"]["undercoverage_rate"],
                "overcoverage_rate": demand_res["eval_90"]["clipped_metrics"]["overcoverage_rate"],
            },
        },
    }

    out_json = table_dir / "uncertainty_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStructured uncertainty summary written to {out_json}")
    print("\n" + "=" * 80)
    print("STAGE 4 CONFORMAL PREDICTION UNCERTAINTY ESTIMATION PIPELINE COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
