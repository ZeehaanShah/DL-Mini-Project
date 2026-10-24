"""
Model E: Uncertainty-Aware Decision Support and Energy-Mix Optimization Module.

Provides:
- Physical per-time-step energy balance simulation (simulate_energy_dispatch, BatteryConfig)
- Scenario generation and provenance tracking (EnergyScenario, ScenarioMetadata, ProvenanceType)
- Deterministic reproducible energy-mix optimization (optimize_energy_mix, OptimizationObjective, CostModel)
- Conformal uncertainty scenario evaluation (evaluate_uncertainty_feasibility)
- Dispatch and trade-off visualization utilities (plotting)
"""
from src.decision_support.energy_balance import (
    BatteryConfig,
    DispatchTimeSeries,
    simulate_energy_dispatch,
)
from src.decision_support.scenarios import (
    EnergyScenario,
    ProvenanceType,
    ScenarioMetadata,
    convert_irradiance_to_solar_pv_kwh,
    convert_wind_speed_to_generation_kwh,
    build_synthetic_hybrid_benchmark_scenario,
)
from src.decision_support.optimizer import (
    CostModel,
    OptimizationObjective,
    OptimizationResult,
    optimize_energy_mix,
)
from src.decision_support.uncertainty_eval import (
    ScenarioBoundResult,
    UncertaintyFeasibilitySummary,
    evaluate_uncertainty_feasibility,
)
from src.decision_support.plotting import (
    plot_energy_dispatch_trajectory,
    plot_uncertainty_dispatch_bounds,
    plot_capacity_optimization_surface,
    plot_scenario_comparison_metrics,
)

__all__ = [
    "BatteryConfig",
    "DispatchTimeSeries",
    "simulate_energy_dispatch",
    "EnergyScenario",
    "ProvenanceType",
    "ScenarioMetadata",
    "convert_irradiance_to_solar_pv_kwh",
    "convert_wind_speed_to_generation_kwh",
    "build_synthetic_hybrid_benchmark_scenario",
    "CostModel",
    "OptimizationObjective",
    "OptimizationResult",
    "optimize_energy_mix",
    "ScenarioBoundResult",
    "UncertaintyFeasibilitySummary",
    "evaluate_uncertainty_feasibility",
    "plot_energy_dispatch_trajectory",
    "plot_uncertainty_dispatch_bounds",
    "plot_capacity_optimization_surface",
    "plot_scenario_comparison_metrics",
]
