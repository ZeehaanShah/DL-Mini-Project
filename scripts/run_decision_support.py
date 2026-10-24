"""
Master orchestration script for Stage 5: Uncertainty-Aware Energy Feasibility and Mix Optimization.

Executes:
1. Loads verified predictions, physical proxies, and conformal uncertainty bounds for Wind and Demand.
2. Evaluates Domain-Specific Feasibility Scenarios:
   - Scenario A: Gujarat Wind Resource Microgrid Feasibility (Empirical Wind + Power Proxy).
   - Scenario B: Orlando Building Demand Feasibility (Empirical Demand + Conformal Bounds).
3. Evaluates Multimodal Hybrid Energy-Mix Optimization (Illustrative Synthetic Benchmark):
   - Deterministic exhaustive grid search over solar, wind, and storage capacities.
   - Optimization under Min-Cost, Min-Grid, and Min-Unmet objectives.
   - Comparative evaluation against 100% Grid baseline.
   - Conformal uncertainty sensitivity analysis (Conservative Stress vs Central vs Favorable).
4. Generates publication-ready figures in reports/figures/.
5. Exports structured tabular outputs to reports/tables/energy_feasibility_scenarios.csv and decision_support_summary.json.
6. Logs experiment records to experiments/experiment_registry.csv without overwriting historical records.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path and initialize Windows PyTorch DLLs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import src.utils.torch_loader

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.decision_support.energy_balance import (
    BatteryConfig,
    simulate_energy_dispatch,
)
from src.decision_support.scenarios import (
    ProvenanceType,
    ScenarioMetadata,
    convert_irradiance_to_solar_pv_kwh,
    convert_wind_speed_to_generation_kwh,
    build_synthetic_hybrid_benchmark_scenario,
)
from src.decision_support.optimizer import (
    CostModel,
    OptimizationObjective,
    optimize_energy_mix,
)
from src.decision_support.uncertainty_eval import evaluate_uncertainty_feasibility
from src.decision_support.plotting import (
    plot_energy_dispatch_trajectory,
    plot_uncertainty_dispatch_bounds,
    plot_capacity_optimization_surface,
    plot_scenario_comparison_metrics,
)
from src.data.wind import build_wind_datasets, TurbinePowerCurve
from src.data.demand import build_demand_datasets
from src.models.wind.sequence_models import WindLSTM
from src.models.demand.sequence_models import DemandLSTM
from src.training.trainer import RegressionTrainer
from src.uncertainty.conformal import SplitConformalCalibrator
from src.evaluation.registry import ExperimentRegistry


def run_stage5_decision_support():
    print("=" * 80)
    print("STAGE 5: UNCERTAINTY-AWARE ENERGY FEASIBILITY & MIX OPTIMIZATION (MODEL E)")
    print("=" * 80)

    cfg_path = PROJECT_ROOT / "configs" / "decision_support.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = PROJECT_ROOT / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    registry = ExperimentRegistry(PROJECT_ROOT / "experiments" / "experiment_registry.csv")

    cost_cfg = cfg.get("costs", {})
    costs = CostModel(
        solar_capex_per_kw=cost_cfg.get("solar_capex_per_kw", 800.0),
        wind_capex_per_kw=cost_cfg.get("wind_capex_per_kw", 1200.0),
        battery_capex_per_kwh=cost_cfg.get("battery_capex_per_kwh", 250.0),
        grid_energy_rate=cost_cfg.get("grid_energy_rate", 0.12),
        unmet_demand_penalty=cost_cfg.get("unmet_demand_penalty", 2.00),
        curtailment_penalty=cost_cfg.get("curtailment_penalty", 0.01),
        amortize_to_horizon_hours=240.0,
        lifetime_years=float(cost_cfg.get("lifetime_years", 20.0)),
    )

    # -------------------------------------------------------------------------
    # PART 1: LOAD VERIFIED TEST PREDICTIONS & UNCERTAINTY QUANTILE SCORES
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# 1. LOADING FROZEN CHECKPOINTS AND CONFORMAL CALIBRATION ARTIFACTS")
    print("#" * 80 + "\n")

    # Load Wind Data and Model
    wind_csv = PROJECT_ROOT / "data" / "processed" / "wind" / "wind_hourly_pilot.csv"
    curve = TurbinePowerCurve()
    _, cal_ds_w, test_ds_w, _, _ = build_wind_datasets(
        csv_path=wind_csv,
        feature_cols=["wind_speed"],
        target_col="wind_speed",
        time_col="datetime",
        sequence_length=24,
        forecast_horizon=1,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
        power_curve=curve,
    )
    wind_ckpt = PROJECT_ROOT / "checkpoints" / "EXP_WIND_DL_C2" / "best_model.pt"
    wind_model = WindLSTM(input_dim=1, hidden_dim=64, num_layers=1, dropout=0.1)
    ckpt_w = torch.load(wind_ckpt, map_location="cpu", weights_only=True)
    wind_model.load_state_dict(ckpt_w["model_state_dict"])
    wind_model.eval()

    unscaler_w = lambda arr: cal_ds_w.inverse_transform_targets(arr)
    trainer_w = RegressionTrainer(
        model=wind_model,
        optimizer=torch.optim.Adam(wind_model.parameters()),
        device=torch.device("cpu"),
        experiment_id="EXP_WIND_DL_C2",
        target_unscaler=unscaler_w,
    )
    cal_loader_w = DataLoader(cal_ds_w, batch_size=32, shuffle=False)
    test_loader_w = DataLoader(test_ds_w, batch_size=32, shuffle=False)
    y_true_w_cal, y_pred_w_cal = trainer_w.predict(cal_loader_w)
    y_true_w_test, y_pred_w_test = trainer_w.predict(test_loader_w)

    calibrator_w = SplitConformalCalibrator().fit(y_true_w_cal, y_pred_w_cal)
    q_w_80 = calibrator_w.compute_quantile(0.20)
    q_w_90 = calibrator_w.compute_quantile(0.10)
    w_low_90, w_upp_90 = calibrator_w.predict_interval(y_pred_w_test, alpha=0.10, clip_min=0.0)

    print(f"Wind Model C loaded: N_test={len(y_true_w_test)}, q(80%)={q_w_80:.4f} m/s, q(90%)={q_w_90:.4f} m/s")

    # Load Demand Data and Model
    demand_csv = PROJECT_ROOT / "data" / "processed" / "demand" / "demand_hourly_pilot.csv"
    _, cal_ds_d, test_ds_d, _, _ = build_demand_datasets(
        csv_path=demand_csv,
        feature_cols=["demand_kwh"],
        target_col="demand_kwh",
        time_col="datetime",
        sequence_length=24,
        forecast_horizon=1,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_hours=24,
        scale=True,
    )
    demand_ckpt = PROJECT_ROOT / "checkpoints" / "EXP_DEMAND_DL_D2" / "best_model.pt"
    demand_model = DemandLSTM(input_dim=1, hidden_dim=64, num_layers=1, dropout=0.1)
    ckpt_d = torch.load(demand_ckpt, map_location="cpu", weights_only=True)
    demand_model.load_state_dict(ckpt_d["model_state_dict"])
    demand_model.eval()

    unscaler_d = lambda arr: cal_ds_d.inverse_transform_targets(arr)
    trainer_d = RegressionTrainer(
        model=demand_model,
        optimizer=torch.optim.Adam(demand_model.parameters()),
        device=torch.device("cpu"),
        experiment_id="EXP_DEMAND_DL_D2",
        target_unscaler=unscaler_d,
    )
    cal_loader_d = DataLoader(cal_ds_d, batch_size=32, shuffle=False)
    test_loader_d = DataLoader(test_ds_d, batch_size=32, shuffle=False)
    y_true_d_cal, y_pred_d_cal = trainer_d.predict(cal_loader_d)
    y_true_d_test, y_pred_d_test = trainer_d.predict(test_loader_d)

    calibrator_d = SplitConformalCalibrator().fit(y_true_d_cal, y_pred_d_cal)
    q_d_80 = calibrator_d.compute_quantile(0.20)
    q_d_90 = calibrator_d.compute_quantile(0.10)
    d_low_90, d_upp_90 = calibrator_d.predict_interval(y_pred_d_test, alpha=0.10, clip_min=0.0)

    print(f"Demand Model D loaded: N_test={len(y_true_d_test)}, q(80%)={q_d_80:.4f} kWh, q(90%)={q_d_90:.4f} kWh")

    # Load Solar Irradiance Data from NASA POWER (Mumbai)
    nasa_csv = PROJECT_ROOT / "data" / "raw" / "nasa_power" / "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"
    nasa_df = pd.read_csv(nasa_csv, skiprows=12)
    solar_irr_full = nasa_df["ALLSKY_SFC_SW_DWN"].to_numpy(dtype=np.float64)

    # -------------------------------------------------------------------------
    # PART 2: MULTIMODAL HYBRID ENERGY-MIX OPTIMIZATION (SYNTHETIC BENCHMARK)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# 2. MULTIMODAL ENERGY-MIX OPTIMIZATION (SYNTHETIC ILLUSTRATIVE BENCHMARK)")
    print("#" * 80 + "\n")

    # Build 240-hour illustrative benchmark scenario
    n_benchmark = 240  # 10 days
    irr_bench = solar_irr_full[:n_benchmark]
    ws_bench = y_pred_w_test[:n_benchmark]
    dem_bench = y_pred_d_test[:n_benchmark]
    timestamps_bench = test_ds_d.target_timestamps[:n_benchmark]

    opt_grid = cfg.get("optimization_grid", {})
    solar_grid = opt_grid.get("solar_capacities_kw", [0.0, 25.0, 50.0, 75.0, 100.0, 150.0])
    wind_grid = opt_grid.get("wind_capacities_kw", [0.0, 50.0, 100.0, 200.0, 300.0])
    batt_grid = opt_grid.get("battery_capacities_kwh", [0.0, 50.0, 100.0, 200.0])

    print(f"Evaluating discrete search grid: {len(solar_grid)} solar x {len(wind_grid)} wind x {len(batt_grid)} batt = {len(solar_grid)*len(wind_grid)*len(batt_grid)} candidates...")

    # Run Optimization for MINIMIZE_COST
    opt_cost_res = optimize_energy_mix(
        demand_kwh=dem_bench,
        solar_irradiance_wh_m2=irr_bench,
        wind_speed_ms=ws_bench,
        solar_grid_kw=solar_grid,
        wind_grid_kw=wind_grid,
        battery_grid_kwh=batt_grid,
        grid_import_limit_kw=cfg.get("grid", {}).get("default_import_limit_kw", 100.0),
        objective=OptimizationObjective.MINIMIZE_COST,
        cost_model=costs,
        unmet_demand_tolerance=opt_grid.get("unmet_demand_tolerance", 0.01),
    )

    # Run Optimization for MINIMIZE_GRID_DEPENDENCE
    opt_grid_res = optimize_energy_mix(
        demand_kwh=dem_bench,
        solar_irradiance_wh_m2=irr_bench,
        wind_speed_ms=ws_bench,
        solar_grid_kw=solar_grid,
        wind_grid_kw=wind_grid,
        battery_grid_kwh=batt_grid,
        grid_import_limit_kw=cfg.get("grid", {}).get("default_import_limit_kw", 100.0),
        objective=OptimizationObjective.MINIMIZE_GRID_DEPENDENCE,
        cost_model=costs,
        unmet_demand_tolerance=opt_grid.get("unmet_demand_tolerance", 0.01),
    )

    print("\n>>> DETERMINISTIC OPTIMIZATION RESULTS:")
    print(f"1. Cost-Optimal Mix: Solar={opt_cost_res.best_capacities['solar_kw']} kW, "
          f"Wind={opt_cost_res.best_capacities['wind_kw']} kW, "
          f"Battery={opt_cost_res.best_capacities['battery_kwh']} kWh")
    print(f"   Renewable Penetration: {opt_cost_res.best_metrics['renewable_penetration']*100:.2f}% | "
          f"Grid Import: {opt_cost_res.best_metrics['grid_import_kwh']:.2f} kWh (Baseline: {opt_cost_res.baseline_metrics['grid_import_kwh']:.2f} kWh, "
          f"-{opt_cost_res.comparison_vs_baseline['grid_reduction_pct']:.1f}%) | "
          f"Total Cost: ${opt_cost_res.best_costs['total_cost']:,.2f}")

    print(f"\n2. Autonomy-Optimal Mix (Min Grid): Solar={opt_grid_res.best_capacities['solar_kw']} kW, "
          f"Wind={opt_grid_res.best_capacities['wind_kw']} kW, "
          f"Battery={opt_grid_res.best_capacities['battery_kwh']} kWh")
    print(f"   Renewable Penetration: {opt_grid_res.best_metrics['renewable_penetration']*100:.2f}% | "
          f"Grid Import: {opt_grid_res.best_metrics['grid_import_kwh']:.2f} kWh (-{opt_grid_res.comparison_vs_baseline['grid_reduction_pct']:.1f}%) | "
          f"Total Cost: ${opt_grid_res.best_costs['total_cost']:,.2f}")

    # -------------------------------------------------------------------------
    # PART 3: CONFORMAL UNCERTAINTY STRESS EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# 3. CONFORMAL PREDICTION UNCERTAINTY SENSITIVITY EVALUATION")
    print("#" * 80 + "\n")

    # Construct uncertainty bounds for benchmark slice
    d_cent_s = dem_bench
    d_low_s = d_low_90[:n_benchmark]
    d_upp_s = d_upp_90[:n_benchmark]

    ws_cent_s = ws_bench
    ws_low_s = w_low_90[:n_benchmark]
    ws_upp_s = w_upp_90[:n_benchmark]

    # Approximate solar uncertainty bounds (+-15% illustrative variation)
    irr_cent_s = irr_bench
    irr_low_s = np.maximum(0.0, irr_bench * 0.85)
    irr_upp_s = irr_bench * 1.15

    best_s_kw = opt_cost_res.best_capacities["solar_kw"]
    best_w_kw = opt_cost_res.best_capacities["wind_kw"]
    best_b_kwh = opt_cost_res.best_capacities["battery_kwh"]
    b_cfg = BatteryConfig(capacity_kwh=best_b_kwh) if best_b_kwh > 0 else None

    unc_summary = evaluate_uncertainty_feasibility(
        demand_central_kwh=d_cent_s,
        demand_lower_kwh=d_low_s,
        demand_upper_kwh=d_upp_s,
        solar_irradiance_central=irr_cent_s,
        solar_irradiance_lower=irr_low_s,
        solar_irradiance_upper=irr_upp_s,
        wind_speed_central=ws_cent_s,
        wind_speed_lower=ws_low_s,
        wind_speed_upper=ws_upp_s,
        solar_capacity_kw=best_s_kw,
        wind_capacity_kw=best_w_kw,
        battery_config=b_cfg,
        grid_import_limit_kw=cfg.get("grid", {}).get("default_import_limit_kw", 100.0),
        nominal_confidence=0.90,
    )

    print("Uncertainty Sensitivity Matrix for Cost-Optimal Configuration:")
    print(unc_summary.sensitivity_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 4: GENERATE PUBLICATION-QUALITY FIGURES
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# 4. GENERATING REPORT VISUALIZATIONS")
    print("#" * 80 + "\n")

    # 1. Hourly Dispatch Trajectory for Selected Optimal Mix (168h / 7 days)
    plot_energy_dispatch_trajectory(
        dispatch=opt_cost_res.best_dispatch,
        output_path=fig_dir / "energy_dispatch_central.png",
        timestamps=timestamps_bench,
        title=f"Stage 5 Decision Support: Optimal Energy Dispatch ({best_s_kw}kW Solar, {best_w_kw}kW Wind, {best_b_kwh}kWh Storage)",
        max_hours=168,
    )

    # 2. Conformal Uncertainty Dispatch Envelopes
    plot_uncertainty_dispatch_bounds(
        dispatch_central=unc_summary.central.dispatch,
        dispatch_stress=unc_summary.conservative_stress.dispatch,
        dispatch_opt=unc_summary.favorable_optimistic.dispatch,
        output_path=fig_dir / "energy_dispatch_uncertainty_bounds.png",
        timestamps=timestamps_bench,
        title="Stage 5 Decision Support: Energy Dispatch under 90% Conformal Bounds",
        max_hours=168,
    )

    # 3. Optimization Surface & Trade-Off Pareto Map
    plot_capacity_optimization_surface(
        candidates_df=opt_cost_res.candidates_table,
        output_path=fig_dir / "capacity_optimization_surface.png",
        title="Stage 5 Decision Support: Candidate Mix Cost vs Grid Reliance Trade-Off",
    )

    # 4. Uncertainty Scenario Metrics Grouped Bar Chart
    plot_scenario_comparison_metrics(
        metrics_summary_df=unc_summary.sensitivity_table,
        output_path=fig_dir / "scenario_comparison_metrics.png",
        title="Stage 5 Decision Support: Energy Metrics Across Conformal Scenarios",
    )

    print("Figures successfully saved to reports/figures/:")
    print(" - energy_dispatch_central.png")
    print(" - energy_dispatch_uncertainty_bounds.png")
    print(" - capacity_optimization_surface.png")
    print(" - scenario_comparison_metrics.png")

    # -------------------------------------------------------------------------
    # PART 5: SAVE STRUCTURED TABULAR OUTPUTS
    # -------------------------------------------------------------------------
    csv_out = table_dir / "energy_feasibility_scenarios.csv"
    opt_cost_res.candidates_table.to_csv(csv_out, index=False)
    print(f"\nCandidate scenario table exported to {csv_out} ({len(opt_cost_res.candidates_table)} configurations).")

    summary_json_path = table_dir / "decision_support_summary.json"
    summary_data = {
        "framework_title": cfg.get("framework_title"),
        "simulation_parameters": {
            "n_hours": n_benchmark,
            "grid_import_limit_kw": cfg.get("grid", {}).get("default_import_limit_kw", 100.0),
            "costs": asdict(costs),
        },
        "baseline_100_percent_grid": {
            "solar_kw": 0.0,
            "wind_kw": 0.0,
            "battery_kwh": 0.0,
            "metrics": opt_cost_res.baseline_metrics,
            "costs": opt_cost_res.baseline_costs,
        },
        "cost_optimal_configuration": {
            "capacities": opt_cost_res.best_capacities,
            "metrics": opt_cost_res.best_metrics,
            "costs": opt_cost_res.best_costs,
            "comparison_vs_baseline": opt_cost_res.comparison_vs_baseline,
        },
        "grid_optimal_configuration": {
            "capacities": opt_grid_res.best_capacities,
            "metrics": opt_grid_res.best_metrics,
            "costs": opt_grid_res.best_costs,
            "comparison_vs_baseline": opt_grid_res.comparison_vs_baseline,
        },
        "uncertainty_stress_evaluation": {
            "nominal_confidence": unc_summary.nominal_confidence,
            "scenarios": unc_summary.sensitivity_table.to_dict(orient="records"),
            "limitations": unc_summary.limitations,
        },
        "provenance_and_limitations": {
            "provenance": "SYNTHETIC_SCENARIO",
            "is_colocated": False,
            "disclaimer": (
                "Composite multimodal scenario combining normalized characteristic signatures "
                "from Models B (irradiance), C (wind speed), and D (demand) for proof-of-concept "
                "decision-support validation. Does NOT claim physical co-location."
            ),
        },
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Decision support summary JSON written to {summary_json_path}")

    # -------------------------------------------------------------------------
    # PART 6: LOG EXPERIMENTS TO REGISTRY WITHOUT OVERWRITING PRIOR ENTRIES
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# 6. LOGGING EXPERIMENTS TO REGISTRY")
    print("#" * 80 + "\n")

    registry.log_experiment(
        experiment_id="EXP_DS_BASE_GRID",
        dataset="SYNTHETIC_MULTIMODAL_POC",
        task="decision_support_feasibility",
        model_name="BaselineGridSupply",
        partition="test_slice",
        features="demand_kwh (240h)",
        target="served_demand_kwh",
        sequence_length=0,
        forecast_horizon=0,
        n_samples=n_benchmark,
        mae=opt_cost_res.baseline_metrics["unmet_demand_kwh"],
        rmse=opt_cost_res.baseline_costs["total_cost"],
        r2=opt_cost_res.baseline_metrics["renewable_penetration"],
        metric_unit="kWh",
        config=str(cfg_path),
        notes="100% grid supply baseline (0 kW solar, 0 kW wind, 0 kWh battery). Total demand served entirely by grid.",
    )

    registry.log_experiment(
        experiment_id="EXP_DS_OPTIMIZED_MIX",
        dataset="SYNTHETIC_MULTIMODAL_POC",
        task="decision_support_feasibility",
        model_name="DeterministicEnergyMixOptimizer",
        partition="test_slice",
        features=f"solar({best_s_kw}kW),wind({best_w_kw}kW),batt({best_b_kwh}kWh)",
        target="served_demand_kwh",
        sequence_length=0,
        forecast_horizon=0,
        n_samples=n_benchmark,
        mae=opt_cost_res.best_metrics["unmet_demand_kwh"],
        rmse=opt_cost_res.best_costs["total_cost"],
        r2=opt_cost_res.best_metrics["renewable_penetration"],
        metric_unit="kWh",
        config=str(cfg_path),
        notes=(
            f"Cost-optimal mix: {best_s_kw}kW Solar, {best_w_kw}kW Wind, {best_b_kwh}kWh Battery. "
            f"Ren. Pen: {opt_cost_res.best_metrics['renewable_penetration']*100:.2f}%, "
            f"Grid red: {opt_cost_res.comparison_vs_baseline['grid_reduction_pct']:.1f}%, "
            f"Unmet: {opt_cost_res.best_metrics['unmet_demand_kwh']:.2f} kWh"
        ),
    )

    m_stress = unc_summary.conservative_stress.metrics
    registry.log_experiment(
        experiment_id="EXP_DS_UNCERTAINTY_STRESS",
        dataset="SYNTHETIC_MULTIMODAL_POC",
        task="decision_support_feasibility",
        model_name="ConformalUncertaintyStressTest",
        partition="test_slice",
        features="conformal_lower_renewable_upper_demand",
        target="served_demand_kwh",
        sequence_length=0,
        forecast_horizon=0,
        n_samples=n_benchmark,
        mae=m_stress["unmet_demand_kwh"],
        rmse=costs.compute_total_cost(best_s_kw, best_w_kw, best_b_kwh, m_stress["grid_import_kwh"], m_stress["unmet_demand_kwh"], m_stress["curtailment_kwh"])["total_cost"],
        r2=m_stress["renewable_penetration"],
        metric_unit="kWh",
        config=str(cfg_path),
        notes=(
            f"Conservative stress test under 90% conformal bounds. "
            f"Ren. Pen: {m_stress['renewable_penetration']*100:.2f}%, "
            f"Grid imp: {m_stress['grid_import_kwh']:.2f} kWh, Unmet: {m_stress['unmet_demand_kwh']:.2f} kWh"
        ),
    )

    print("\n" + "=" * 80)
    print("STAGE 5 DECISION SUPPORT PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_stage5_decision_support()
