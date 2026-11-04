"""
Uncertainty-aware energy feasibility evaluation using Split Conformal Prediction intervals.

Evaluates candidate system configurations across deterministic uncertainty scenarios:
- CENTRAL: Point forecasts (expected dispatch).
- CONSERVATIVE / STRESS: Lower renewable bound + Upper demand bound (system vulnerability test).
- FAVORABLE / OPTIMISTIC: Upper renewable bound + Lower demand bound (curtailment risk test).

Scientific Disclosure:
Combining marginal conformal intervals does NOT provide a calibrated joint probability
guarantee without copula modeling or Bonferroni adjustments. These scenarios represent
deterministic sensitivity bounds derived from the calibrated nonconformity quantiles.
"""
from dataclasses import dataclass
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


@dataclass
class ScenarioBoundResult:
    """Feasibility outcome for a specific uncertainty scenario bound."""
    scenario_name: str
    nominal_confidence: float
    demand_series: np.ndarray
    renewable_series: np.ndarray
    metrics: Dict[str, float]
    dispatch: DispatchTimeSeries


@dataclass
class UncertaintyFeasibilitySummary:
    """Summary of feasibility metrics across Central, Conservative, and Favorable scenarios."""
    nominal_confidence: float
    capacities: Dict[str, float]
    central: ScenarioBoundResult
    conservative_stress: ScenarioBoundResult
    favorable_optimistic: ScenarioBoundResult
    sensitivity_table: pd.DataFrame
    limitations: List[str]


def evaluate_uncertainty_feasibility(
    demand_central_kwh: np.ndarray,
    demand_lower_kwh: np.ndarray,
    demand_upper_kwh: np.ndarray,
    solar_irradiance_central: np.ndarray,
    solar_irradiance_lower: np.ndarray,
    solar_irradiance_upper: np.ndarray,
    wind_speed_central: np.ndarray,
    wind_speed_lower: np.ndarray,
    wind_speed_upper: np.ndarray,
    solar_capacity_kw: float,
    wind_capacity_kw: float,
    battery_config: Optional[BatteryConfig] = None,
    grid_import_limit_kw: float = 100.0,
    nominal_confidence: float = 0.90,
    time_step_hours: float = 1.0,
) -> UncertaintyFeasibilitySummary:
    """
    Evaluate system feasibility across uncertainty scenarios.

    Args:
        demand_central_kwh: Point forecast of demand.
        demand_lower_kwh: Conformal lower bound on demand (clipped >= 0).
        demand_upper_kwh: Conformal upper bound on demand.
        solar_irradiance_central: Central solar irradiance (Wh/m²).
        solar_irradiance_lower: Conservative solar irradiance.
        solar_irradiance_upper: Optimistic solar irradiance.
        wind_speed_central: Central wind speed (m/s).
        wind_speed_lower: Conformal lower bound on wind speed (clipped >= 0).
        wind_speed_upper: Conformal upper bound on wind speed.
        solar_capacity_kw: Installed solar PV capacity in kW.
        wind_capacity_kw: Installed wind capacity in kW.
        battery_config: Optional BatteryConfig object.
        grid_import_limit_kw: Continuous grid import capacity in kW.
        nominal_confidence: Nominal confidence level (e.g. 0.80 or 0.90).
        time_step_hours: Step duration in hours.

    Returns:
        UncertaintyFeasibilitySummary object.
    """
    b_cfg = battery_config if battery_config is not None else BatteryConfig(capacity_kwh=0.0)

    # 1. Central Scenario (Point forecasts)
    s_gen_cent = convert_irradiance_to_solar_pv_kwh(solar_irradiance_central, solar_capacity_kw, time_step_hours=time_step_hours)
    w_gen_cent = convert_wind_speed_to_generation_kwh(wind_speed_central, wind_capacity_kw, time_step_hours=time_step_hours)
    r_cent = s_gen_cent + w_gen_cent
    ts_cent, m_cent = simulate_energy_dispatch(
        demand_kwh=demand_central_kwh,
        renewable_available_kwh=r_cent,
        grid_import_limit_kw=grid_import_limit_kw,
        battery=b_cfg,
        time_step_hours=time_step_hours,
    )
    res_cent = ScenarioBoundResult(
        scenario_name="CENTRAL_POINT_FORECAST",
        nominal_confidence=nominal_confidence,
        demand_series=demand_central_kwh,
        renewable_series=r_cent,
        metrics=m_cent,
        dispatch=ts_cent,
    )

    # 2. Conservative / Stress Scenario (Lower renewables, Upper demand)
    s_gen_low = convert_irradiance_to_solar_pv_kwh(solar_irradiance_lower, solar_capacity_kw, time_step_hours=time_step_hours)
    w_gen_low = convert_wind_speed_to_generation_kwh(wind_speed_lower, wind_capacity_kw, time_step_hours=time_step_hours)
    r_low = s_gen_low + w_gen_low
    ts_stress, m_stress = simulate_energy_dispatch(
        demand_kwh=demand_upper_kwh,
        renewable_available_kwh=r_low,
        grid_import_limit_kw=grid_import_limit_kw,
        battery=b_cfg,
        time_step_hours=time_step_hours,
    )
    res_stress = ScenarioBoundResult(
        scenario_name="CONSERVATIVE_STRESS_TEST",
        nominal_confidence=nominal_confidence,
        demand_series=demand_upper_kwh,
        renewable_series=r_low,
        metrics=m_stress,
        dispatch=ts_stress,
    )

    # 3. Favorable / Optimistic Scenario (Upper renewables, Lower demand)
    s_gen_upp = convert_irradiance_to_solar_pv_kwh(solar_irradiance_upper, solar_capacity_kw, time_step_hours=time_step_hours)
    w_gen_upp = convert_wind_speed_to_generation_kwh(wind_speed_upper, wind_capacity_kw, time_step_hours=time_step_hours)
    r_upp = s_gen_upp + w_gen_upp
    ts_opt, m_opt = simulate_energy_dispatch(
        demand_kwh=demand_lower_kwh,
        renewable_available_kwh=r_upp,
        grid_import_limit_kw=grid_import_limit_kw,
        battery=b_cfg,
        time_step_hours=time_step_hours,
    )
    res_opt = ScenarioBoundResult(
        scenario_name="FAVORABLE_OPTIMISTIC",
        nominal_confidence=nominal_confidence,
        demand_series=demand_lower_kwh,
        renewable_series=r_upp,
        metrics=m_opt,
        dispatch=ts_opt,
    )

    # Build sensitivity summary DataFrame
    summary_data = [
        {
            "scenario": "Conservative (Stress)",
            "demand_kwh": m_stress["total_demand_kwh"],
            "renewable_kwh": m_stress["available_renewable_kwh"],
            "grid_import_kwh": m_stress["grid_import_kwh"],
            "unmet_demand_kwh": m_stress["unmet_demand_kwh"],
            "curtailment_kwh": m_stress["curtailment_kwh"],
            "renewable_penetration": m_stress["renewable_penetration"],
            "loss_of_load_fraction": m_stress["loss_of_load_fraction"],
            "loss_of_load_hours": m_stress["loss_of_load_hours"],
        },
        {
            "scenario": "Central (Expected)",
            "demand_kwh": m_cent["total_demand_kwh"],
            "renewable_kwh": m_cent["available_renewable_kwh"],
            "grid_import_kwh": m_cent["grid_import_kwh"],
            "unmet_demand_kwh": m_cent["unmet_demand_kwh"],
            "curtailment_kwh": m_cent["curtailment_kwh"],
            "renewable_penetration": m_cent["renewable_penetration"],
            "loss_of_load_fraction": m_cent["loss_of_load_fraction"],
            "loss_of_load_hours": m_cent["loss_of_load_hours"],
        },
        {
            "scenario": "Favorable (Optimistic)",
            "demand_kwh": m_opt["total_demand_kwh"],
            "renewable_kwh": m_opt["available_renewable_kwh"],
            "grid_import_kwh": m_opt["grid_import_kwh"],
            "unmet_demand_kwh": m_opt["unmet_demand_kwh"],
            "curtailment_kwh": m_opt["curtailment_kwh"],
            "renewable_penetration": m_opt["renewable_penetration"],
            "loss_of_load_fraction": m_opt["loss_of_load_fraction"],
            "loss_of_load_hours": m_opt["loss_of_load_hours"],
        },
    ]
    summary_df = pd.DataFrame(summary_data)

    limitations = [
        "Uncertainty bounds are constructed from marginal split conformal intervals.",
        "Joint coverage is not guaranteed without copula correlation modeling or Bonferroni scaling.",
        "Demand conformal intervals exhibit empirical test set undercoverage due to Florida seasonal distribution shift.",
        "Wind generation uses an illustrative generic 2.0 MW turbine power curve proxy.",
    ]

    return UncertaintyFeasibilitySummary(
        nominal_confidence=nominal_confidence,
        capacities={
            "solar_kw": solar_capacity_kw,
            "wind_kw": wind_capacity_kw,
            "battery_kwh": b_cfg.capacity_kwh,
        },
        central=res_cent,
        conservative_stress=res_stress,
        favorable_optimistic=res_opt,
        sensitivity_table=summary_df,
        limitations=limitations,
    )
