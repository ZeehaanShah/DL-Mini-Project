"""
Per-time-step energy balance and physical dispatch simulation.

Implements rigorous physical energy conservation:
    Renewable Energy Used + Permitted Grid Supply + Battery Discharge + Unmet Demand = Demand
    Available Renewable = Renewable Energy Used + Battery Charge + Curtailment

All quantities are computed in kilowatt-hours (kWh) over regular hourly time intervals (dt = 1.0 h).
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


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
    battery_renewable_soc: np.ndarray
    renewable_origin_battery_discharge: np.ndarray
    initial_soc_kwh: float


def _solve_cyclic_schedule(
    demand_kwh: np.ndarray,
    renewable_kwh: np.ndarray,
    grid_energy_max_kwh: float,
    battery: BatteryConfig,
    time_step_hours: float,
    initial_soc_kwh: float,
    target_soc_kwh: float,
) -> Tuple[np.ndarray, ...]:
    """Solve a finite-horizon dispatch MILP with a hard terminal-SOC target.

    The lexicographic objectives minimize (1) unmet demand, (2) grid imports,
    (3) renewable curtailment, then (4) battery throughput. A binary operating
    mode prevents charging and discharging in the same interval. Grid charging
    is excluded, matching the non-cyclic dispatch policy.
    """
    n_steps = len(demand_kwh)
    n_vars = 8 * n_steps
    idx_ren = 0
    idx_charge = n_steps
    idx_curtail = 2 * n_steps
    idx_grid = 3 * n_steps
    idx_unmet = 4 * n_steps
    idx_discharge = 5 * n_steps
    idx_soc = 6 * n_steps
    idx_mode = 7 * n_steps

    soc_min = battery.capacity_kwh * battery.min_soc_ratio
    soc_max = battery.capacity_kwh * battery.max_soc_ratio
    charge_limit = battery.capacity_kwh * battery.max_c_rate_charge * time_step_hours
    discharge_limit = battery.capacity_kwh * battery.max_c_rate_discharge * time_step_hours

    lower = np.zeros(n_vars, dtype=np.float64)
    upper = np.full(n_vars, np.inf, dtype=np.float64)
    upper[idx_ren:idx_ren + n_steps] = np.minimum(demand_kwh, renewable_kwh)
    upper[idx_charge:idx_charge + n_steps] = charge_limit
    upper[idx_curtail:idx_curtail + n_steps] = renewable_kwh
    upper[idx_grid:idx_grid + n_steps] = grid_energy_max_kwh
    upper[idx_unmet:idx_unmet + n_steps] = demand_kwh
    upper[idx_discharge:idx_discharge + n_steps] = discharge_limit
    lower[idx_soc:idx_soc + n_steps] = soc_min
    upper[idx_soc:idx_soc + n_steps] = soc_max
    lower[idx_mode:idx_mode + n_steps] = 0.0
    upper[idx_mode:idx_mode + n_steps] = 1.0
    lower[idx_soc + n_steps - 1] = target_soc_kwh
    upper[idx_soc + n_steps - 1] = target_soc_kwh

    integrality = np.zeros(n_vars, dtype=np.int32)
    integrality[idx_mode:idx_mode + n_steps] = 1

    rows = []
    cols = []
    values = []
    row_lower = np.full(5 * n_steps, -np.inf, dtype=np.float64)
    row_upper = np.full(5 * n_steps, np.inf, dtype=np.float64)

    def add(row: int, col: int, value: float) -> None:
        rows.append(row)
        cols.append(col)
        values.append(value)

    for t in range(n_steps):
        # R_used + grid + battery discharge + unmet = demand.
        add(t, idx_ren + t, 1.0)
        add(t, idx_grid + t, 1.0)
        add(t, idx_discharge + t, 1.0)
        add(t, idx_unmet + t, 1.0)
        row_lower[t] = row_upper[t] = demand_kwh[t]

        # R_used + battery charge + curtailment = renewable available.
        ren_row = n_steps + t
        add(ren_row, idx_ren + t, 1.0)
        add(ren_row, idx_charge + t, 1.0)
        add(ren_row, idx_curtail + t, 1.0)
        row_lower[ren_row] = row_upper[ren_row] = renewable_kwh[t]

        # SOC[t] = SOC[t-1] + eta_charge*charge - discharge/eta_discharge.
        soc_row = 2 * n_steps + t
        add(soc_row, idx_soc + t, 1.0)
        if t == 0:
            row_lower[soc_row] = row_upper[soc_row] = initial_soc_kwh
        else:
            add(soc_row, idx_soc + t - 1, -1.0)
            row_lower[soc_row] = row_upper[soc_row] = 0.0
        add(soc_row, idx_charge + t, -battery.charge_efficiency)
        add(soc_row, idx_discharge + t, 1.0 / battery.discharge_efficiency)

        # mode=1 means discharge; mode=0 means charge or idle.
        charge_row = 3 * n_steps + t
        add(charge_row, idx_charge + t, 1.0)
        add(charge_row, idx_mode + t, charge_limit)
        row_upper[charge_row] = charge_limit

        discharge_row = 4 * n_steps + t
        add(discharge_row, idx_discharge + t, 1.0)
        add(discharge_row, idx_mode + t, -discharge_limit)
        row_upper[discharge_row] = 0.0

    base_matrix = coo_matrix((values, (rows, cols)), shape=(5 * n_steps, n_vars)).tocsc()
    base_constraint = LinearConstraint(base_matrix, row_lower, row_upper)

    objectives = []
    for start, stop in (
        (idx_unmet, idx_unmet + n_steps),
        (idx_grid, idx_grid + n_steps),
        (idx_curtail, idx_curtail + n_steps),
    ):
        objective = np.zeros(n_vars, dtype=np.float64)
        objective[start:stop] = 1.0
        objectives.append(objective)
    throughput_objective = np.zeros(n_vars, dtype=np.float64)
    throughput_objective[idx_charge:idx_charge + n_steps] = 1.0
    throughput_objective[idx_discharge:idx_discharge + n_steps] = 1.0
    objectives.append(throughput_objective)

    lexicographic_constraints = []
    solution = None
    for objective in objectives:
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=[base_constraint, *lexicographic_constraints],
            options={"mip_rel_gap": 0.0},
        )
        if result.status == 2:
            raise ValueError(
                "Cyclic terminal SOC infeasible under the renewable-only charging, "
                f"SOC, and power constraints: {result.message}"
            )
        if not result.success or result.x is None:
            raise RuntimeError(f"Cyclic dispatch solver did not find a proven optimum: {result.message}")
        solution = result.x
        optimum = float(np.dot(objective, solution))
        objective_row = coo_matrix(objective.reshape(1, -1)).tocsc()
        lexicographic_constraints.append(
            LinearConstraint(objective_row, -np.inf, optimum + 1e-8)
        )

    assert solution is not None

    def extract(index: int) -> np.ndarray:
        return np.maximum(solution[index:index + n_steps], 0.0)

    r_used = extract(idx_ren)
    b_charge = extract(idx_charge)
    curtailment = extract(idx_curtail)
    grid_import = extract(idx_grid)
    unmet_demand = extract(idx_unmet)
    b_discharge = extract(idx_discharge)
    b_soc = extract(idx_soc)
    return r_used, b_charge, curtailment, grid_import, unmet_demand, b_discharge, b_soc


def simulate_energy_dispatch(
    demand_kwh: Union[np.ndarray, list],
    renewable_available_kwh: Union[np.ndarray, list],
    grid_import_limit_kw: float = 1000.0,
    battery: Optional[BatteryConfig] = None,
    time_step_hours: float = 1.0,
    terminal_soc_ratio: Optional[float] = None,
) -> Tuple[DispatchTimeSeries, Dict[str, Any]]:
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
        terminal_soc_ratio: If set, solve a cyclic mixed-integer dispatch schedule whose
            final SOC equals this fraction of capacity. The schedule lexicographically
            minimizes unmet demand, grid imports, renewable curtailment, then battery
            throughput. Infeasible schedules raise ValueError. None preserves the
            finite-horizon greedy policy and permits a different final SOC.

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
    if terminal_soc_ratio is not None:
        if not np.isfinite(terminal_soc_ratio) or not (b_cfg.min_soc_ratio <= terminal_soc_ratio <= b_cfg.max_soc_ratio):
            raise ValueError("terminal_soc_ratio must be finite and within the battery SOC limits")
        if b_cfg.capacity_kwh <= 1e-6:
            raise ValueError("A terminal SOC target requires a nonzero battery capacity")

    # Initialize output arrays
    r_used = np.zeros(n_steps, dtype=np.float64)
    curtailment = np.zeros(n_steps, dtype=np.float64)
    grid_import = np.zeros(n_steps, dtype=np.float64)
    unmet_demand = np.zeros(n_steps, dtype=np.float64)
    b_charge = np.zeros(n_steps, dtype=np.float64)
    b_discharge = np.zeros(n_steps, dtype=np.float64)
    b_soc = np.zeros(n_steps, dtype=np.float64)
    b_renewable_soc = np.zeros(n_steps, dtype=np.float64)
    renewable_origin_discharge = np.zeros(n_steps, dtype=np.float64)

    # Battery limits
    has_battery = b_cfg.capacity_kwh > 1e-6
    soc_min = b_cfg.capacity_kwh * b_cfg.min_soc_ratio
    soc_max = b_cfg.capacity_kwh * b_cfg.max_soc_ratio
    p_ch_max = b_cfg.capacity_kwh * b_cfg.max_c_rate_charge * time_step_hours
    p_dis_max = b_cfg.capacity_kwh * b_cfg.max_c_rate_discharge * time_step_hours
    current_soc = b_cfg.capacity_kwh * b_cfg.initial_soc_ratio if has_battery else 0.0
    initial_soc = current_soc
    renewable_soc = 0.0  # Stored renewable-origin energy; initial SOC is carry-in.

    # Grid energy limit per step (Power * dt)
    grid_energy_max = grid_import_limit_kw * time_step_hours

    if terminal_soc_ratio is not None:
        target_soc = b_cfg.capacity_kwh * terminal_soc_ratio
        (
            r_used,
            b_charge,
            curtailment,
            grid_import,
            unmet_demand,
            b_discharge,
            scheduled_soc,
        ) = _solve_cyclic_schedule(
            demand_kwh=d,
            renewable_kwh=r,
            grid_energy_max_kwh=grid_energy_max,
            battery=b_cfg,
            time_step_hours=time_step_hours,
            initial_soc_kwh=initial_soc,
            target_soc_kwh=target_soc,
        )

        # Track renewable-origin inventory separately from initial carry-in.
        for t in range(n_steps):
            if b_charge[t] > 0.0:
                stored_renewable = b_charge[t] * b_cfg.charge_efficiency
                renewable_soc += stored_renewable
                current_soc += stored_renewable
            elif b_discharge[t] > 0.0:
                withdrawn_soc = b_discharge[t] / b_cfg.discharge_efficiency
                renewable_fraction = renewable_soc / current_soc if current_soc > 1e-12 else 0.0
                renewable_withdrawal = min(renewable_soc, withdrawn_soc * renewable_fraction)
                renewable_soc -= renewable_withdrawal
                renewable_origin_discharge[t] = renewable_withdrawal * b_cfg.discharge_efficiency
                current_soc -= withdrawn_soc
            b_soc[t] = scheduled_soc[t]
            b_renewable_soc[t] = renewable_soc
            if abs(current_soc - scheduled_soc[t]) > 1e-6:
                raise RuntimeError("Cyclic schedule SOC does not match renewable-origin accounting state")
    else:
        for t in range(n_steps):
            dt = d[t]
            rt = r[t]

            if rt >= dt:
                # Renewable surplus
                r_used[t] = dt
                surplus = rt - dt

                if has_battery:
                    headroom = max(0.0, soc_max - current_soc)
                    max_charge_energy = min(surplus, p_ch_max, headroom / b_cfg.charge_efficiency)
                    b_charge[t] = max_charge_energy
                    stored_energy = max_charge_energy * b_cfg.charge_efficiency
                    current_soc += stored_energy
                    renewable_soc += stored_energy
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
                    withdrawn_soc = disch / b_cfg.discharge_efficiency
                    renewable_fraction = (renewable_soc / current_soc) if current_soc > 1e-12 else 0.0
                    renewable_withdrawal = min(renewable_soc, withdrawn_soc * renewable_fraction)
                    renewable_soc -= renewable_withdrawal
                    renewable_origin_discharge[t] = renewable_withdrawal * b_cfg.discharge_efficiency
                    current_soc -= withdrawn_soc
                    rem_deficit = deficit - disch
                else:
                    rem_deficit = deficit

                g_imp = min(rem_deficit, grid_energy_max)
                grid_import[t] = g_imp
                unmet_demand[t] = rem_deficit - g_imp

            b_soc[t] = current_soc
            b_renewable_soc[t] = renewable_soc

    served_demand = r_used + b_discharge + grid_import

    # Conservation checks (numerical tolerance: 1e-6 kWh). Battery carry-in is
    # reported separately; charging losses and discharging losses are explicit.
    demand_diff = np.abs((served_demand + unmet_demand) - d)
    if np.any(demand_diff > 1e-6):
        max_viol = float(np.max(demand_diff))
        raise RuntimeError(f"Energy conservation violation in demand serving: max violation {max_viol:.2e} kWh")

    ren_diff = np.abs((r_used + b_charge + curtailment) - r)
    if np.any(ren_diff > 1e-6):
        max_viol = float(np.max(ren_diff))
        raise RuntimeError(f"Energy conservation violation in renewable balance: max violation {max_viol:.2e} kWh")

    if terminal_soc_ratio is not None:
        target_soc = b_cfg.capacity_kwh * terminal_soc_ratio
        if abs(current_soc - target_soc) > 1e-6:
            raise RuntimeError(
                f"Cyclic dispatch violated its solved terminal SOC: final SOC {current_soc:.6f} kWh "
                f"does not match target {target_soc:.6f} kWh"
            )

    battery_soc_residual = (
        current_soc - initial_soc
        - float(np.sum(b_charge)) * b_cfg.charge_efficiency
        + float(np.sum(b_discharge)) / b_cfg.discharge_efficiency
    ) if has_battery else 0.0
    if abs(battery_soc_residual) > 1e-6:
        raise RuntimeError(
            f"Battery state-of-charge conservation violation: {battery_soc_residual:.2e} kWh"
        )
    origin_diff = np.abs(renewable_origin_discharge - np.minimum(b_discharge, renewable_origin_discharge))
    if np.any(origin_diff > 1e-9) or np.any(b_renewable_soc > b_soc + 1e-9):
        raise RuntimeError("Renewable-origin battery accounting exceeded available discharge or SOC")

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
        battery_renewable_soc=b_renewable_soc,
        renewable_origin_battery_discharge=renewable_origin_discharge,
        initial_soc_kwh=float(initial_soc),
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
    renewable_b_discharge = float(np.sum(renewable_origin_discharge))
    carry_in_discharge = max(0.0, total_b_discharge - renewable_b_discharge)
    final_soc = float(current_soc)

    # Renewable penetration counts direct renewable supply and only the share of
    # battery output charged from renewables during this simulation. Initial SOC
    # is carry-in energy and is never mislabeled as newly generated renewable.
    ren_share_served = float((total_r_used + renewable_b_discharge) / total_served) if total_served > 1e-6 else 0.0
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
        "renewable_origin_battery_discharge_kwh": round(renewable_b_discharge, 4),
        "initial_soc_carry_in_discharge_kwh": round(carry_in_discharge, 4),
        "battery_charge_kwh": round(total_b_charge, 4),
        "initial_soc_kwh": round(float(initial_soc), 4),
        "final_soc_kwh": round(final_soc, 4),
        "terminal_soc_delta_kwh": round(final_soc - float(initial_soc), 4),
        "battery_soc_balance_residual_kwh": round(float(battery_soc_residual), 10),
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
        "cyclic_terminal_soc_enforced": terminal_soc_ratio is not None,
        "terminal_soc_target_kwh": round(b_cfg.capacity_kwh * terminal_soc_ratio, 4) if terminal_soc_ratio is not None else None,
    }

    return ts, metrics
