"""
Per-time-step energy balance and physical dispatch simulation.

Implements rigorous physical energy conservation:
    Renewable Energy Used + Permitted Grid Supply + Battery Discharge + Unmet Demand = Demand
    Available Renewable = Renewable Energy Used + Battery Charge + Curtailment

All quantities are computed in kilowatt-hours (kWh) over regular hourly time intervals (dt = 1.0 h).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np


def _to_1d_nonnegative_array(arr: Union[np.ndarray, list], name: str) -> np.ndarray:
    """Validate and convert array-like input to a flat 1D float64 non-negative array."""
    np_arr = np.asarray(arr, dtype=np.float64)
    if np_arr.ndim > 1 and np_arr.size != max(np_arr.shape):
        raise ValueError(f"'{name}' must be 1-dimensional; got shape {np_arr.shape}")
    flat = np_arr.ravel()
    if len(flat) == 0:
        raise ValueError(f"'{name}' cannot be empty")
    if not np.isfinite(flat).all():
        n_invalid = int(np.isnan(flat).sum() + np.isinf(flat).sum())
        raise ValueError(f"'{name}' contains {n_invalid} non-finite (NaN or Inf) values")
    if np.any(flat < -1e-9):
        min_val = float(np.min(flat))
        raise ValueError(f"'{name}' contains negative values (minimum: {min_val:.4f}); physical energy must be non-negative")
    return np.maximum(flat, 0.0)


@dataclass
class BatteryConfig:
    """
    Physical parameters for Battery Energy Storage System (BESS).

    Attributes:
        capacity_kwh: Nominal battery energy storage capacity in kWh.
        initial_soc_ratio: Initial State of Charge (SOC) as fraction of capacity (0 to 1).
        min_soc_ratio: Minimum allowable SOC reserve fraction (0 to 1).
        max_soc_ratio: Maximum allowable charging ceiling fraction (0 to 1).
        charge_efficiency: One-way charging efficiency in (0, 1].
        discharge_efficiency: One-way discharging efficiency in (0, 1].
        max_c_rate_charge: Maximum charging C-rate (power in kW = C-rate * capacity).
        max_c_rate_discharge: Maximum discharging C-rate (power in kW = C-rate * capacity).
    """
    capacity_kwh: float = 0.0
    initial_soc_ratio: float = 0.50
    min_soc_ratio: float = 0.10
    max_soc_ratio: float = 0.95
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    max_c_rate_charge: float = 0.50
    max_c_rate_discharge: float = 0.50

    def __post_init__(self):
        if self.capacity_kwh < 0.0:
            raise ValueError(f"Battery capacity must be non-negative; got {self.capacity_kwh}")
        if not (0.0 <= self.min_soc_ratio <= self.max_soc_ratio <= 1.0):
            raise ValueError(f"Invalid SOC ratios: 0 <= min ({self.min_soc_ratio}) <= max ({self.max_soc_ratio}) <= 1 required")
        if not (self.min_soc_ratio <= self.initial_soc_ratio <= self.max_soc_ratio):
            raise ValueError(f"initial_soc_ratio ({self.initial_soc_ratio}) must lie in [{self.min_soc_ratio}, {self.max_soc_ratio}]")
        if not (0.0 < self.charge_efficiency <= 1.0):
            raise ValueError(f"charge_efficiency must be in (0, 1]; got {self.charge_efficiency}")
        if not (0.0 < self.discharge_efficiency <= 1.0):
            raise ValueError(f"discharge_efficiency must be in (0, 1]; got {self.discharge_efficiency}")
        if self.max_c_rate_charge <= 0.0 or self.max_c_rate_discharge <= 0.0:
            raise ValueError("Max C-rates must be strictly positive")


@dataclass
class DispatchTimeSeries:
    """Detailed per-time-step energy dispatch records (all in kWh per hour)."""
    demand: np.ndarray
    renewable_available: np.ndarray
    renewable_used: np.ndarray
    curtailment: np.ndarray
    grid_import: np.ndarray
    unmet_demand: np.ndarray
    served_demand: np.ndarray
    battery_charge: np.ndarray
    battery_discharge: np.ndarray
    battery_soc: np.ndarray


def simulate_energy_dispatch(
    demand_kwh: Union[np.ndarray, list],
    renewable_available_kwh: Union[np.ndarray, list],
    grid_import_limit_kw: float = 1000.0,
    battery: Optional[BatteryConfig] = None,
    time_step_hours: float = 1.0,
) -> Tuple[DispatchTimeSeries, Dict[str, float]]:
    """
    Simulate per-time-step energy balance and dispatch under physical constraints.

    Governing Equations at each step t (dt = 1h):
    1. Direct Renewable Serving:
       R_used[t] = min(Demand[t], R_avail[t])
    2. Surplus Handling (R_avail[t] > Demand[t]):
       Surplus[t] = R_avail[t] - Demand[t]
       Battery_charge[t] = min(Surplus[t], P_ch_max, (SOC_max - SOC[t-1]) / eta_ch)
       Curtailment[t] = Surplus[t] - Battery_charge[t]
    3. Deficit Handling (R_avail[t] < Demand[t]):
       Deficit[t] = Demand[t] - R_used[t]
       Battery_discharge[t] = min(Deficit[t], P_dis_max, eta_dis * (SOC[t-1] - SOC_min))
       Remaining_deficit[t] = Deficit[t] - Battery_discharge[t]
       Grid_import[t] = min(Remaining_deficit[t], Grid_max)
       Unmet_demand[t] = Remaining_deficit[t] - Grid_import[t]
    4. Served Demand:
       Served_demand[t] = R_used[t] + Battery_discharge[t] + Grid_import[t]
    5. Conservation Laws:
       Served_demand[t] + Unmet_demand[t] == Demand[t]
       R_avail[t] == R_used[t] + Battery_charge[t] + Curtailment[t]

    Args:
        demand_kwh: Hourly electricity demand series in kWh.
        renewable_available_kwh: Hourly available renewable generation series in kWh.
        grid_import_limit_kw: Maximum continuous grid supply power limit in kW.
        battery: Optional BatteryConfig object. If None, battery capacity is 0.
        time_step_hours: Duration of time step in hours (default 1.0 h).

    Returns:
        Tuple of (DispatchTimeSeries, summary_metrics_dictionary).
    """
    d = _to_1d_nonnegative_array(demand_kwh, "demand_kwh")
    r = _to_1d_nonnegative_array(renewable_available_kwh, "renewable_available_kwh")

    if len(d) != len(r):
        raise ValueError(f"Shape mismatch: demand length {len(d)} != renewable length {len(r)}")
    if grid_import_limit_kw < 0.0:
        raise ValueError(f"grid_import_limit_kw must be non-negative; got {grid_import_limit_kw}")
    if time_step_hours <= 0.0:
        raise ValueError(f"time_step_hours must be strictly positive; got {time_step_hours}")

    n_steps = len(d)
    b_cfg = battery if battery is not None else BatteryConfig(capacity_kwh=0.0)

    # Initialize output arrays
    r_used = np.zeros(n_steps, dtype=np.float64)
    curtailment = np.zeros(n_steps, dtype=np.float64)
    grid_import = np.zeros(n_steps, dtype=np.float64)
    unmet_demand = np.zeros(n_steps, dtype=np.float64)
    b_charge = np.zeros(n_steps, dtype=np.float64)
    b_discharge = np.zeros(n_steps, dtype=np.float64)
    b_soc = np.zeros(n_steps, dtype=np.float64)

    # Battery limits
    has_battery = b_cfg.capacity_kwh > 1e-6
    soc_min = b_cfg.capacity_kwh * b_cfg.min_soc_ratio
    soc_max = b_cfg.capacity_kwh * b_cfg.max_soc_ratio
    p_ch_max = b_cfg.capacity_kwh * b_cfg.max_c_rate_charge * time_step_hours
    p_dis_max = b_cfg.capacity_kwh * b_cfg.max_c_rate_discharge * time_step_hours
    current_soc = b_cfg.capacity_kwh * b_cfg.initial_soc_ratio if has_battery else 0.0

    # Grid energy limit per step (Power * dt)
    grid_energy_max = grid_import_limit_kw * time_step_hours

    for t in range(n_steps):
        dt = d[t]
        rt = r[t]

        if rt >= dt:
            # Renewable surplus
            r_used[t] = dt
            surplus = rt - dt

            if has_battery:
                # Max energy that can be absorbed by battery
                headroom = max(0.0, soc_max - current_soc)
                max_charge_energy = min(surplus, p_ch_max, headroom / b_cfg.charge_efficiency)
                b_charge[t] = max_charge_energy
                current_soc += max_charge_energy * b_cfg.charge_efficiency
                curtailment[t] = surplus - max_charge_energy
            else:
                curtailment[t] = surplus

            b_discharge[t] = 0.0
            grid_import[t] = 0.0
            unmet_demand[t] = 0.0

        else:
            # Renewable deficit
            r_used[t] = rt
            curtailment[t] = 0.0
            deficit = dt - rt

            if has_battery:
                available_discharge_energy = max(0.0, (current_soc - soc_min) * b_cfg.discharge_efficiency)
                disch = min(deficit, p_dis_max, available_discharge_energy)
                b_discharge[t] = disch
                current_soc -= (disch / b_cfg.discharge_efficiency)
                rem_deficit = deficit - disch
            else:
                rem_deficit = deficit

            # Grid import up to grid capacity limit
            g_imp = min(rem_deficit, grid_energy_max)
            grid_import[t] = g_imp
            unmet_demand[t] = rem_deficit - g_imp

        b_soc[t] = current_soc

    served_demand = r_used + b_discharge + grid_import

    # Conservation checks (numerical tolerance: 1e-6 kWh)
    demand_diff = np.abs((served_demand + unmet_demand) - d)
    if np.any(demand_diff > 1e-6):
        max_viol = float(np.max(demand_diff))
        raise RuntimeError(f"Energy conservation violation in demand serving: max violation {max_viol:.2e} kWh")

    ren_diff = np.abs((r_used + b_charge + curtailment) - r)
    if np.any(ren_diff > 1e-6):
        max_viol = float(np.max(ren_diff))
        raise RuntimeError(f"Energy conservation violation in renewable balance: max violation {max_viol:.2e} kWh")

    ts = DispatchTimeSeries(
        demand=d,
        renewable_available=r,
        renewable_used=r_used,
        curtailment=curtailment,
        grid_import=grid_import,
        unmet_demand=unmet_demand,
        served_demand=served_demand,
        battery_charge=b_charge,
        battery_discharge=b_discharge,
        battery_soc=b_soc,
    )

    # Compute high-level summary metrics
    total_demand = float(np.sum(d))
    total_r_avail = float(np.sum(r))
    total_r_used = float(np.sum(r_used))
    total_curtailment = float(np.sum(curtailment))
    total_grid = float(np.sum(grid_import))
    total_unmet = float(np.sum(unmet_demand))
    total_served = float(np.sum(served_demand))
    total_b_discharge = float(np.sum(b_discharge))
    total_b_charge = float(np.sum(b_charge))

    # Safe zero-division metric computations
    ren_share_served = float((total_r_used + total_b_discharge) / total_served) if total_served > 1e-6 else 0.0
    direct_ren_share_served = float(total_r_used / total_served) if total_served > 1e-6 else 0.0
    grid_share_served = float(total_grid / total_served) if total_served > 1e-6 else 0.0
    loss_of_load_fraction = float(total_unmet / total_demand) if total_demand > 1e-6 else 0.0
    curtailment_fraction = float(total_curtailment / total_r_avail) if total_r_avail > 1e-6 else 0.0
    lolh = int(np.sum(unmet_demand > 1e-6))

    metrics = {
        "n_hours": int(n_steps),
        "total_demand_kwh": round(total_demand, 4),
        "available_renewable_kwh": round(total_r_avail, 4),
        "served_demand_kwh": round(total_served, 4),
        "renewable_used_kwh": round(total_r_used, 4),
        "battery_discharge_kwh": round(total_b_discharge, 4),
        "battery_charge_kwh": round(total_b_charge, 4),
        "grid_import_kwh": round(total_grid, 4),
        "unmet_demand_kwh": round(total_unmet, 4),
        "curtailment_kwh": round(total_curtailment, 4),
        "renewable_penetration": round(ren_share_served, 4),
        "direct_renewable_share": round(direct_ren_share_served, 4),
        "grid_dependence_fraction": round(grid_share_served, 4),
        "loss_of_load_fraction": round(loss_of_load_fraction, 4),
        "curtailment_fraction": round(curtailment_fraction, 4),
        "loss_of_load_hours": lolh,
        "peak_unmet_demand_kw": round(float(np.max(unmet_demand)), 4),
        "peak_grid_import_kw": round(float(np.max(grid_import)), 4),
    }

    return ts, metrics
