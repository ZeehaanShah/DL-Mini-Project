"""
Deterministic reproducible renewable energy-mix optimizer.

Evaluates candidate solar, wind, and storage capacities under physical constraints and
configurable objectives:
- MINIMIZE_UNMET_DEMAND: Maximizes reliability (loss of load minimization).
- MINIMIZE_GRID_DEPENDENCE: Maximizes renewable autonomy subject to reliability tolerance.
- MINIMIZE_COST: Minimizes total illustrative capital and operational costs.

Because evaluation is systematically conducted over a discrete candidate grid, this method
guarantees the global optimum over the specified search space with 100% reproducibility.
"""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.decision_support.energy_balance import (
    BatteryConfig,
    simulate_energy_dispatch,
    DispatchTimeSeries,
)
from src.decision_support.scenarios import (
    convert_irradiance_to_solar_pv_kwh,
    convert_wind_speed_to_generation_kwh,
)


class OptimizationObjective(Enum):
    MINIMIZE_UNMET_DEMAND = "MINIMIZE_UNMET_DEMAND"
    MINIMIZE_GRID_DEPENDENCE = "MINIMIZE_GRID_DEPENDENCE"
    MINIMIZE_COST = "MINIMIZE_COST"


@dataclass
class CostModel:
    """
    Illustrative engineering cost parameters.

    NOTE: Clearly labelled as illustrative benchmark proxies, NOT actual vendor bids or commercial quotes.
    """
    solar_capex_per_kw: float = 800.0       # Illustrative installed overnight capital proxy ($/kW)
    wind_capex_per_kw: float = 1200.0       # Illustrative installed overnight capital proxy ($/kW)
    battery_capex_per_kwh: float = 250.0    # Illustrative battery storage capital proxy ($/kWh)
    grid_energy_rate: float = 0.12          # Illustrative grid tariff ($/kWh)
    unmet_demand_penalty: float = 2.00      # Value of Lost Load (VoLL) penalty ($/kWh)
    curtailment_penalty: float = 0.01       # Minor over-generation penalty ($/kWh)
    amortize_to_horizon_hours: Optional[float] = None  # If set, prorates CAPEX to evaluation horizon
    lifetime_years: float = 20.0            # Assumed economic asset lifetime in years

    def compute_total_cost(
        self,
        solar_kw: float,
        wind_kw: float,
        battery_kwh: float,
        grid_import_kwh: float,
        unmet_demand_kwh: float,
        curtailment_kwh: float,
    ) -> Dict[str, float]:
        raw_capex = (
            self.solar_capex_per_kw * solar_kw
            + self.wind_capex_per_kw * wind_kw
            + self.battery_capex_per_kwh * battery_kwh
        )
        if self.amortize_to_horizon_hours is not None and self.amortize_to_horizon_hours > 0:
            capex_fraction = self.amortize_to_horizon_hours / (self.lifetime_years * 8760.0)
            capex = raw_capex * capex_fraction
        else:
            capex = raw_capex

        opex = (
            self.grid_energy_rate * grid_import_kwh
            + self.unmet_demand_penalty * unmet_demand_kwh
            + self.curtailment_penalty * curtailment_kwh
        )
        return {
            "capex": round(capex, 2),
            "opex": round(opex, 2),
            "total_cost": round(capex + opex, 2),
        }


@dataclass
class OptimizationResult:
    """Structured output of deterministic energy mix optimization."""
    objective: OptimizationObjective
    best_capacities: Dict[str, float]
    best_metrics: Dict[str, float]
    best_costs: Dict[str, float]
    best_dispatch: DispatchTimeSeries
    baseline_metrics: Dict[str, float]
    baseline_costs: Dict[str, float]
    comparison_vs_baseline: Dict[str, float]
    candidates_table: pd.DataFrame
    guarantees_global_optimum: bool = True


def optimize_energy_mix(
    demand_kwh: np.ndarray,
    solar_irradiance_wh_m2: np.ndarray,
    wind_speed_ms: np.ndarray,
    solar_grid_kw: List[float],
    wind_grid_kw: List[float],
    battery_grid_kwh: List[float],
    grid_import_limit_kw: float = 100.0,
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST,
    cost_model: Optional[CostModel] = None,
    unmet_demand_tolerance: float = 0.01,
    time_step_hours: float = 1.0,
) -> OptimizationResult:
    """
    Search discrete candidate capacity grid to find the optimal renewable energy mix.

    Args:
        demand_kwh: Hourly electricity demand series in kWh.
        solar_irradiance_wh_m2: Hourly solar irradiance in Wh/m².
        wind_speed_ms: Hourly wind speed in m/s.
        solar_grid_kw: List of candidate solar capacities in kW to evaluate.
        wind_grid_kw: List of candidate wind capacities in kW to evaluate.
        battery_grid_kwh: List of candidate battery capacities in kWh to evaluate.
        grid_import_limit_kw: Continuous grid import power limit in kW.
        objective: Selected OptimizationObjective.
        cost_model: CostModel object for economic evaluations.
        unmet_demand_tolerance: Maximum allowable loss of load fraction for constrained objectives.
        time_step_hours: Step duration in hours.

    Returns:
        OptimizationResult object.
    """
    costs = cost_model if cost_model is not None else CostModel()

    # 1. Evaluate Baseline: 100% Grid Supply (0 kW Solar, 0 kW Wind, 0 kWh Battery)
    zero_ren = np.zeros_like(demand_kwh, dtype=np.float64)
    base_ts, base_metrics = simulate_energy_dispatch(
        demand_kwh=demand_kwh,
        renewable_available_kwh=zero_ren,
        grid_import_limit_kw=grid_import_limit_kw,
        battery=BatteryConfig(capacity_kwh=0.0),
        time_step_hours=time_step_hours,
    )
    base_costs = costs.compute_total_cost(
        solar_kw=0.0,
        wind_kw=0.0,
        battery_kwh=0.0,
        grid_import_kwh=base_metrics["grid_import_kwh"],
        unmet_demand_kwh=base_metrics["unmet_demand_kwh"],
        curtailment_kwh=base_metrics["curtailment_kwh"],
    )

    # 2. Systematically Evaluate All Grid Candidates
    candidate_records = []
    best_candidate = None
    best_score = float("inf")
    best_ts = None
    best_metrics_dict = None
    best_costs_dict = None
    best_cap_dict = None

    for s_kw in solar_grid_kw:
        s_gen = convert_irradiance_to_solar_pv_kwh(solar_irradiance_wh_m2, capacity_kw=s_kw)
        for w_kw in wind_grid_kw:
            w_gen = convert_wind_speed_to_generation_kwh(wind_speed_ms, capacity_kw=w_kw)
            r_total = s_gen + w_gen

            for b_kwh in battery_grid_kwh:
                b_cfg = BatteryConfig(capacity_kwh=b_kwh) if b_kwh > 0 else BatteryConfig(capacity_kwh=0.0)

                ts, m = simulate_energy_dispatch(
                    demand_kwh=demand_kwh,
                    renewable_available_kwh=r_total,
                    grid_import_limit_kw=grid_import_limit_kw,
                    battery=b_cfg,
                    time_step_hours=time_step_hours,
                )

                c = costs.compute_total_cost(
                    solar_kw=s_kw,
                    wind_kw=w_kw,
                    battery_kwh=b_kwh,
                    grid_import_kwh=m["grid_import_kwh"],
                    unmet_demand_kwh=m["unmet_demand_kwh"],
                    curtailment_kwh=m["curtailment_kwh"],
                )

                # Determine objective score
                llf = m["loss_of_load_fraction"]
                is_feasible = llf <= unmet_demand_tolerance

                if objective == OptimizationObjective.MINIMIZE_UNMET_DEMAND:
                    # Lexicographic: minimize unmet demand first, then grid import, then cost
                    score = (
                        m["unmet_demand_kwh"] * 1e6
                        + m["grid_import_kwh"] * 1e2
                        + c["total_cost"]
                    )
                elif objective == OptimizationObjective.MINIMIZE_GRID_DEPENDENCE:
                    if is_feasible:
                        score = m["grid_import_kwh"] + (c["total_cost"] * 1e-4)
                    else:
                        # Heavy penalty if unmet demand constraint is violated
                        score = 1e9 + (m["unmet_demand_kwh"] * 1e4)
                elif objective == OptimizationObjective.MINIMIZE_COST:
                    if is_feasible:
                        score = c["total_cost"]
                    else:
                        score = 1e9 + (m["unmet_demand_kwh"] * 1e4) + c["total_cost"]
                else:
                    raise ValueError(f"Unknown optimization objective: {objective}")

                rec = {
                    "solar_kw": s_kw,
                    "wind_kw": w_kw,
                    "battery_kwh": b_kwh,
                    "renewable_penetration": m["renewable_penetration"],
                    "grid_import_kwh": m["grid_import_kwh"],
                    "unmet_demand_kwh": m["unmet_demand_kwh"],
                    "curtailment_kwh": m["curtailment_kwh"],
                    "loss_of_load_fraction": m["loss_of_load_fraction"],
                    "loss_of_load_hours": m["loss_of_load_hours"],
                    "capex": c["capex"],
                    "opex": c["opex"],
                    "total_cost": c["total_cost"],
                    "is_feasible": is_feasible,
                    "score": score,
                }
                candidate_records.append(rec)

                if score < best_score:
                    best_score = score
                    best_candidate = rec
                    best_ts = ts
                    best_metrics_dict = m
                    best_costs_dict = c
                    best_cap_dict = {
                        "solar_kw": s_kw,
                        "wind_kw": w_kw,
                        "battery_kwh": b_kwh,
                    }

    candidates_df = pd.DataFrame(candidate_records)

    # 3. Compute Delta Comparison vs Baseline
    grid_reduction_kwh = base_metrics["grid_import_kwh"] - best_metrics_dict["grid_import_kwh"]
    grid_reduction_pct = (grid_reduction_kwh / base_metrics["grid_import_kwh"] * 100.0) if base_metrics["grid_import_kwh"] > 0 else 0.0
    cost_diff = best_costs_dict["total_cost"] - base_costs["total_cost"]
    cost_savings_pct = (-cost_diff / base_costs["total_cost"] * 100.0) if base_costs["total_cost"] > 0 else 0.0

    comparison = {
        "baseline_grid_import_kwh": base_metrics["grid_import_kwh"],
        "optimized_grid_import_kwh": best_metrics_dict["grid_import_kwh"],
        "grid_reduction_kwh": round(grid_reduction_kwh, 4),
        "grid_reduction_pct": round(grid_reduction_pct, 2),
        "baseline_unmet_kwh": base_metrics["unmet_demand_kwh"],
        "optimized_unmet_kwh": best_metrics_dict["unmet_demand_kwh"],
        "baseline_total_cost": base_costs["total_cost"],
        "optimized_total_cost": best_costs_dict["total_cost"],
        "cost_diff": round(cost_diff, 2),
        "cost_savings_pct": round(cost_savings_pct, 2),
        "renewable_penetration_achieved": best_metrics_dict["renewable_penetration"],
    }

    return OptimizationResult(
        objective=objective,
        best_capacities=best_cap_dict,
        best_metrics=best_metrics_dict,
        best_costs=best_costs_dict,
        best_dispatch=best_ts,
        baseline_metrics=base_metrics,
        baseline_costs=base_costs,
        comparison_vs_baseline=comparison,
        candidates_table=candidates_df,
        guarantees_global_optimum=True,
    )
