"""
Unit tests for Stage 5: Decision Support, Energy Balance, and Energy-Mix Optimization.

Tests verify:
1. Physical energy conservation laws at every time-step and across the entire simulation horizon.
2. Robust error handling and rejection of negative values, non-finite inputs, and length mismatches.
3. Boundary conditions: zero demand, zero renewable, zero grid limit (islanded mode).
4. Battery storage constraints: SOC boundaries, power ratings, and round-trip efficiency losses.
5. Deterministic optimizer reproducibility and global optimality over discrete search spaces.
6. Baseline comparison calculations against 100% grid supply.
7. Uncertainty bound ordering and scenario stress testing.
8. Provenance metadata labeling separating empirical data from synthetic illustrative scenarios.
"""
import pytest
import numpy as np
import pandas as pd

from src.decision_support.energy_balance import (
    BatteryConfig,
    simulate_energy_dispatch,
    _to_1d_nonnegative_array,
)
from src.decision_support.scenarios import (
    ProvenanceType,
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


class TestEnergyBalanceConservation:
    """Test rigorous physical energy conservation laws."""

    def test_conservation_without_storage_surplus_and_deficit(self):
        """Verify served + unmet == demand and renewable == used + curtailment."""
        # 4-hour test: hour 0 surplus, hour 1 deficit, hour 2 exact match, hour 3 zero renewable
        demand = np.array([50.0, 100.0, 75.0, 60.0])
        renewable = np.array([80.0, 40.0, 75.0, 0.0])
        grid_limit = 50.0  # hour 1: deficit is 60, grid allows 50 -> unmet is 10

        ts, m = simulate_energy_dispatch(
            demand_kwh=demand,
            renewable_available_kwh=renewable,
            grid_import_limit_kw=grid_limit,
            battery=None,
        )

        # 1. Demand conservation at every step: served + unmet == demand
        np.testing.assert_allclose(ts.served_demand + ts.unmet_demand, demand, atol=1e-6)

        # 2. Renewable conservation at every step: used + curtailment == available
        np.testing.assert_allclose(ts.renewable_used + ts.curtailment, renewable, atol=1e-6)

        # 3. Specific step checks
        # Step 0: 80 available, 50 demand -> 50 used, 30 curtailed, 0 grid, 0 unmet
        assert ts.renewable_used[0] == 50.0
        assert ts.curtailment[0] == 30.0
        assert ts.grid_import[0] == 0.0
        assert ts.unmet_demand[0] == 0.0

        # Step 1: 40 available, 100 demand -> 40 used, 0 curtailed, 50 grid (limit), 10 unmet
        assert ts.renewable_used[1] == 40.0
        assert ts.curtailment[1] == 0.0
        assert ts.grid_import[1] == 50.0
        assert ts.unmet_demand[1] == 10.0

        # Step 3: 0 available, 60 demand -> 0 used, 50 grid, 10 unmet
        assert ts.renewable_used[3] == 0.0
        assert ts.grid_import[3] == 50.0
        assert ts.unmet_demand[3] == 10.0

    def test_conservation_with_battery_storage(self):
        """Verify battery charging, discharging, SOC conservation, and round-trip losses."""
        # Battery: 100 kWh capacity, initial SOC = 50 kWh (0.50 ratio), min=10, max=90
        # Charge eff = 0.90, Discharge eff = 0.90
        b_cfg = BatteryConfig(
            capacity_kwh=100.0,
            initial_soc_ratio=0.50,
            min_soc_ratio=0.10,
            max_soc_ratio=0.90,
            charge_efficiency=0.90,
            discharge_efficiency=0.90,
            max_c_rate_charge=1.0,
            max_c_rate_discharge=1.0,
        )

        # Hour 0: Surplus 20 kWh -> charges battery
        # Hour 1: Deficit 30 kWh -> discharges battery
        demand = np.array([30.0, 80.0])
        renewable = np.array([50.0, 50.0])

        ts, m = simulate_energy_dispatch(
            demand_kwh=demand,
            renewable_available_kwh=renewable,
            grid_import_limit_kw=100.0,
            battery=b_cfg,
        )

        # Step 0: Surplus = 20 kWh.
        # Absorbed into battery = 20 kWh * 0.90 = 18 kWh.
        # New SOC = 50 + 18 = 68 kWh.
        assert ts.battery_charge[0] == 20.0
        assert ts.curtailment[0] == 0.0
        assert ts.battery_soc[0] == pytest.approx(68.0, abs=1e-5)

        # Step 1: Deficit = 30 kWh.
        # Available discharge energy in battery above min_soc (10 kWh) = (68 - 10) * 0.90 = 52.2 kWh.
        # Required discharge = 30 kWh <= 52.2.
        # Discharged to load = 30 kWh.
        # Energy removed from storage = 30 / 0.90 = 33.3333 kWh.
        # New SOC = 68 - 33.3333 = 34.6667 kWh.
        assert ts.battery_discharge[1] == pytest.approx(30.0, abs=1e-5)
        assert ts.battery_soc[1] == pytest.approx(68.0 - (30.0 / 0.90), abs=1e-5)
        assert ts.grid_import[1] == 0.0
        assert ts.unmet_demand[1] == 0.0

        # Physical conservation: demand perfectly served
        np.testing.assert_allclose(ts.served_demand + ts.unmet_demand, demand, atol=1e-6)
        assert m["battery_soc_balance_residual_kwh"] == pytest.approx(0.0, abs=1e-6)

    def test_initial_soc_is_carry_in_not_renewable_generation(self):
        """Initial stored energy may serve load but must not inflate renewable share."""
        battery = BatteryConfig(capacity_kwh=100.0, initial_soc_ratio=0.50, min_soc_ratio=0.10)
        ts, metrics = simulate_energy_dispatch([10.0], [0.0], grid_import_limit_kw=0.0, battery=battery)

        assert ts.battery_discharge[0] == pytest.approx(10.0)
        assert metrics["renewable_origin_battery_discharge_kwh"] == pytest.approx(0.0)
        assert metrics["initial_soc_carry_in_discharge_kwh"] == pytest.approx(10.0)
        assert metrics["renewable_penetration"] == pytest.approx(0.0)
        assert metrics["terminal_soc_delta_kwh"] < 0.0
        assert metrics["battery_soc_balance_residual_kwh"] == pytest.approx(0.0, abs=1e-6)

    def test_renewable_origin_storage_is_counted_once(self):
        """Only the renewable-origin share of discharge contributes to renewable penetration."""
        battery = BatteryConfig(
            capacity_kwh=100.0, initial_soc_ratio=0.10, min_soc_ratio=0.10,
            charge_efficiency=0.90, discharge_efficiency=0.90,
            max_c_rate_charge=1.0, max_c_rate_discharge=1.0,
        )
        ts, metrics = simulate_energy_dispatch(
            [30.0, 40.0], [50.0, 30.0], grid_import_limit_kw=0.0, battery=battery
        )

        assert metrics["renewable_origin_battery_discharge_kwh"] == pytest.approx(6.4285714, abs=0.00005)
        assert metrics["initial_soc_carry_in_discharge_kwh"] == pytest.approx(3.5714286, abs=0.00005)
        assert metrics["renewable_used_kwh"] + metrics["renewable_origin_battery_discharge_kwh"] <= metrics["served_demand_kwh"] + 1e-6
        assert metrics["renewable_penetration"] == pytest.approx(66.4285714 / 70.0, abs=0.00005)
        assert np.all(ts.battery_renewable_soc <= ts.battery_soc + 1e-9)
        assert metrics["battery_soc_balance_residual_kwh"] == pytest.approx(0.0, abs=1e-6)


class TestInputValidationAndRejection:
    """Test validation errors for invalid, non-finite, and negative inputs."""

    def test_negative_values_raise(self):
        """Negative demand or renewable must raise ValueError."""
        with pytest.raises(ValueError, match="negative values"):
            _to_1d_nonnegative_array([10.0, -5.0, 20.0], "demand")

        with pytest.raises(ValueError, match="negative values"):
            simulate_energy_dispatch([-1.0], [10.0])

        with pytest.raises(ValueError, match="negative values"):
            simulate_energy_dispatch([10.0], [-1.0])

    def test_non_finite_values_raise(self):
        """NaN or Inf inputs must raise ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            _to_1d_nonnegative_array([10.0, np.nan, 20.0], "demand")

        with pytest.raises(ValueError, match="non-finite"):
            _to_1d_nonnegative_array([10.0, np.inf, 20.0], "demand")

    def test_shape_mismatch_raises(self):
        """Mismatched demand and renewable lengths must raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            simulate_energy_dispatch([10.0, 20.0], [10.0])

    def test_negative_grid_limit_raises(self):
        """Negative grid capacity limit must raise ValueError."""
        with pytest.raises(ValueError, match="grid_import_limit_kw must be non-negative"):
            simulate_energy_dispatch([10.0], [5.0], grid_import_limit_kw=-10.0)

    def test_invalid_battery_config_raises(self):
        """Invalid battery parameters must raise ValueError."""
        with pytest.raises(ValueError, match="Battery capacity must be non-negative"):
            BatteryConfig(capacity_kwh=-5.0)

        with pytest.raises(ValueError, match="Invalid SOC ratios"):
            BatteryConfig(min_soc_ratio=0.8, max_soc_ratio=0.4)

        with pytest.raises(ValueError, match="charge_efficiency must be in"):
            BatteryConfig(charge_efficiency=0.0)


class TestBoundaryConditions:
    """Test boundary edge cases: zero demand, zero renewable, zero grid limit."""

    def test_zero_demand(self):
        """When demand is 0, served demand and unmet demand are 0; all renewable is curtailed."""
        demand = np.array([0.0, 0.0])
        renewable = np.array([50.0, 30.0])
        ts, m = simulate_energy_dispatch(demand, renewable)
        assert m["total_demand_kwh"] == 0.0
        assert m["served_demand_kwh"] == 0.0
        assert m["unmet_demand_kwh"] == 0.0
        assert m["curtailment_kwh"] == 80.0
        assert m["loss_of_load_fraction"] == 0.0

    def test_zero_renewable(self):
        """When renewable is 0, demand is met entirely by grid (or unmet if over grid limit)."""
        demand = np.array([60.0, 120.0])
        renewable = np.array([0.0, 0.0])
        grid_limit = 100.0

        ts, m = simulate_energy_dispatch(demand, renewable, grid_import_limit_kw=grid_limit)
        assert m["renewable_used_kwh"] == 0.0
        assert m["grid_import_kwh"] == 160.0  # 60 + 100
        assert m["unmet_demand_kwh"] == 20.0  # 0 + 20
        assert m["loss_of_load_hours"] == 1

    def test_islanded_zero_grid_limit(self):
        """Zero grid limit (islanded microgrid): grid import must be strictly zero."""
        demand = np.array([50.0, 50.0])
        renewable = np.array([30.0, 70.0])
        ts, m = simulate_energy_dispatch(demand, renewable, grid_import_limit_kw=0.0)
        assert np.all(ts.grid_import == 0.0)
        assert m["grid_import_kwh"] == 0.0
        assert ts.unmet_demand[0] == 20.0
        assert ts.unmet_demand[1] == 0.0


class TestDeterministicOptimizer:
    """Test deterministic mix optimizer and reproducibility."""

    def test_reproducibility_and_global_optimum(self):
        """Running optimizer twice yields bitwise identical results matching global grid optimum."""
        np.random.seed(42)
        n = 48
        demand = np.random.uniform(30.0, 60.0, size=n)
        solar_irr = np.random.uniform(0.0, 800.0, size=n)
        wind_speed = np.random.uniform(2.0, 10.0, size=n)

        solar_grid = [0.0, 50.0, 100.0]
        wind_grid = [0.0, 50.0, 100.0]
        batt_grid = [0.0, 50.0]

        res1 = optimize_energy_mix(
            demand_kwh=demand,
            solar_irradiance_wh_m2=solar_irr,
            wind_speed_ms=wind_speed,
            solar_grid_kw=solar_grid,
            wind_grid_kw=wind_grid,
            battery_grid_kwh=batt_grid,
            objective=OptimizationObjective.MINIMIZE_COST,
        )

        res2 = optimize_energy_mix(
            demand_kwh=demand,
            solar_irradiance_wh_m2=solar_irr,
            wind_speed_ms=wind_speed,
            solar_grid_kw=solar_grid,
            wind_grid_kw=wind_grid,
            battery_grid_kwh=batt_grid,
            objective=OptimizationObjective.MINIMIZE_COST,
        )

        assert res1.best_capacities == res2.best_capacities
        assert res1.best_costs["total_cost"] == res2.best_costs["total_cost"]
        assert res1.guarantees_global_optimum is True

        # Verify best score is strictly the minimum score in candidate table
        min_table_score = res1.candidates_table["score"].min()
        best_score_found = res1.candidates_table.loc[
            (res1.candidates_table["solar_kw"] == res1.best_capacities["solar_kw"])
            & (res1.candidates_table["wind_kw"] == res1.best_capacities["wind_kw"])
            & (res1.candidates_table["battery_kwh"] == res1.best_capacities["battery_kwh"])
        ]["score"].values[0]
        assert best_score_found == pytest.approx(min_table_score)

    def test_baseline_comparison_validity(self):
        """Baseline metrics correspond to 100% grid supply (0 kW renewables)."""
        demand = np.array([40.0, 50.0, 60.0])
        irr = np.array([500.0, 600.0, 700.0])
        ws = np.array([6.0, 7.0, 8.0])

        res = optimize_energy_mix(
            demand_kwh=demand,
            solar_irradiance_wh_m2=irr,
            wind_speed_ms=ws,
            solar_grid_kw=[0.0, 50.0],
            wind_grid_kw=[0.0, 50.0],
            battery_grid_kwh=[0.0],
            grid_import_limit_kw=100.0,
            objective=OptimizationObjective.MINIMIZE_GRID_DEPENDENCE,
        )

        # Baseline must have zero renewable penetration and 100% grid
        assert res.baseline_metrics["renewable_used_kwh"] == 0.0
        assert res.baseline_metrics["grid_import_kwh"] == 150.0
        assert res.comparison_vs_baseline["renewable_penetration_achieved"] > 0.0


class TestUncertaintyFeasibility:
    """Test conformal uncertainty scenario stress testing."""

    def test_uncertainty_bound_ordering(self):
        """Stress scenario has higher grid reliance / unmet demand than favorable scenario."""
        n = 24
        dem_cent = np.full(n, 50.0)
        dem_low = np.full(n, 45.0)
        dem_upp = np.full(n, 55.0)

        irr_cent = np.full(n, 600.0)
        irr_low = np.full(n, 400.0)
        irr_upp = np.full(n, 800.0)

        ws_cent = np.full(n, 6.0)
        ws_low = np.full(n, 4.5)
        ws_upp = np.full(n, 7.5)

        summary = evaluate_uncertainty_feasibility(
            demand_central_kwh=dem_cent,
            demand_lower_kwh=dem_low,
            demand_upper_kwh=dem_upp,
            solar_irradiance_central=irr_cent,
            solar_irradiance_lower=irr_low,
            solar_irradiance_upper=irr_upp,
            wind_speed_central=ws_cent,
            wind_speed_lower=ws_low,
            wind_speed_upper=ws_upp,
            solar_capacity_kw=50.0,
            wind_capacity_kw=50.0,
            grid_import_limit_kw=50.0,
            nominal_confidence=0.90,
        )

        m_stress = summary.conservative_stress.metrics
        m_cent = summary.central.metrics
        m_opt = summary.favorable_optimistic.metrics

        # Available renewables: Low <= Central <= Upper
        assert m_stress["available_renewable_kwh"] < m_cent["available_renewable_kwh"] < m_opt["available_renewable_kwh"]
        # Demand: Upper >= Central >= Lower
        assert m_stress["total_demand_kwh"] > m_cent["total_demand_kwh"] > m_opt["total_demand_kwh"]
        # Grid import: Stress >= Central >= Optimistic
        assert m_stress["grid_import_kwh"] >= m_cent["grid_import_kwh"] >= m_opt["grid_import_kwh"]


class TestScenarioProvenanceLabeling:
    """Test scenario provenance tagging and physical conversion models."""

    def test_synthetic_scenario_metadata_integrity(self):
        """Synthetic multimodal benchmark scenario is explicitly tagged as SYNTHETIC_SCENARIO."""
        irr = np.full(50, 500.0)
        ws = np.full(50, 6.0)
        dem = np.full(50, 40.0)

        scen = build_synthetic_hybrid_benchmark_scenario(
            solar_irradiance_wh_m2=irr,
            wind_speed_ms=ws,
            demand_kwh=dem,
            solar_capacity_kw=50.0,
            wind_capacity_kw=100.0,
            n_hours=50,
        )

        assert scen.metadata.provenance == ProvenanceType.SYNTHETIC_SCENARIO
        assert scen.metadata.is_colocated is False
        assert len(scen.metadata.limitations) > 0
        assert scen.metadata.demand_unit == "kWh"

    def test_solar_and_wind_conversion_physics(self):
        """Verify solar and wind conversion output units and clipping physics."""
        # Solar: 1000 W/m² (STC) with 50 kW capacity and PR 0.78 -> 50 * 1.0 * 0.78 = 39.0 kWh
        sol_gen = convert_irradiance_to_solar_pv_kwh([1000.0], capacity_kw=50.0, performance_ratio=0.78)
        assert sol_gen[0] == pytest.approx(39.0)

        # Inverter clipping: 1500 W/m² (high reflection) should clip at 50 kW
        sol_clip = convert_irradiance_to_solar_pv_kwh([1500.0], capacity_kw=50.0, performance_ratio=0.78)
        assert sol_clip[0] <= 50.0

        # Wind: 12 m/s (rated) with 100 kW capacity -> 100.0 kWh
        w_gen = convert_wind_speed_to_generation_kwh([12.0], capacity_kw=100.0)
        assert w_gen[0] == pytest.approx(100.0)

        # Wind cut-in: 2.0 m/s (< 3.0 m/s cut-in) -> 0.0 kWh
        w_zero = convert_wind_speed_to_generation_kwh([2.0], capacity_kw=100.0)
        assert w_zero[0] == 0.0




def test_resource_conversions_scale_energy_with_interval_duration():
    from src.decision_support.scenarios import (
        convert_irradiance_to_solar_pv_kwh,
        convert_wind_speed_to_generation_kwh,
    )
    # Equal average irradiance over a 2h interval yields twice the energy.
    solar_1h = convert_irradiance_to_solar_pv_kwh([1000.0], 10.0, time_step_hours=1.0)
    solar_2h = convert_irradiance_to_solar_pv_kwh([2000.0], 10.0, time_step_hours=2.0)
    np.testing.assert_allclose(solar_2h, solar_1h * 2.0)
    wind_1h = convert_wind_speed_to_generation_kwh([12.0], 10.0, time_step_hours=1.0)
    wind_2h = convert_wind_speed_to_generation_kwh([12.0], 10.0, time_step_hours=2.0)
    np.testing.assert_allclose(wind_2h, wind_1h * 2.0)


def test_cost_model_prices_battery_carry_in_at_grid_rate():
    from src.decision_support.optimizer import CostModel
    model = CostModel(grid_energy_rate=0.12)
    costs = model.compute_total_cost(
        solar_kw=0.0, wind_kw=0.0, battery_kwh=0.0,
        grid_import_kwh=0.0, unmet_demand_kwh=0.0, curtailment_kwh=0.0,
        carry_in_discharge_kwh=10.0,
    )
    assert costs["carry_in_energy_cost"] == 1.2
    assert costs["total_cost"] == 1.2


def test_resource_conversions_reject_nonfinite_values():
    from src.decision_support.scenarios import convert_irradiance_to_solar_pv_kwh
    with pytest.raises(ValueError, match="finite"):
        convert_irradiance_to_solar_pv_kwh([float("nan")], 10.0)


def test_cyclic_terminal_soc_option_accepts_balanced_repeated_horizon():
    from src.decision_support.energy_balance import BatteryConfig, simulate_energy_dispatch
    battery = BatteryConfig(capacity_kwh=10.0, initial_soc_ratio=0.50)
    _, metrics = simulate_energy_dispatch(
        demand_kwh=[0.0, 4.14], renewable_available_kwh=[5.0, 0.0],
        battery=battery, terminal_soc_ratio=0.50,
    )
    assert metrics["final_soc_kwh"] == pytest.approx(metrics["initial_soc_kwh"], abs=1e-6)


def test_cyclic_dispatch_reschedules_to_reach_terminal_soc():
    from src.decision_support.energy_balance import BatteryConfig, simulate_energy_dispatch
    battery = BatteryConfig(capacity_kwh=10.0, initial_soc_ratio=0.50)
    ts, metrics = simulate_energy_dispatch(
        demand_kwh=[0.0, 4.14], renewable_available_kwh=[5.0, 0.0],
        battery=battery, terminal_soc_ratio=0.50,
    )
    assert metrics["final_soc_kwh"] == pytest.approx(metrics["initial_soc_kwh"], abs=1e-6)
    assert metrics["terminal_soc_delta_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["unmet_demand_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cyclic_terminal_soc_enforced"] is True
    assert metrics["renewable_origin_battery_discharge_kwh"] <= metrics["battery_discharge_kwh"]
    assert np.all(ts.battery_soc >= 1.0 - 1e-6)
    assert np.all(ts.battery_soc <= 9.5 + 1e-6)
    assert np.all((ts.battery_charge <= 1e-7) | (ts.battery_discharge <= 1e-7))
    assert np.all(ts.battery_charge <= 5.0 + 1e-6)
    assert np.all(ts.battery_discharge <= 5.0 + 1e-6)
    np.testing.assert_allclose(
        ts.battery_soc,
        np.r_[5.0, ts.battery_soc[:-1]]
        + ts.battery_charge * battery.charge_efficiency
        - ts.battery_discharge / battery.discharge_efficiency,
        atol=1e-6,
    )


def test_cyclic_dispatch_reports_true_infeasibility():
    from src.decision_support.energy_balance import BatteryConfig, simulate_energy_dispatch
    battery = BatteryConfig(capacity_kwh=10.0, initial_soc_ratio=0.50)
    with pytest.raises(ValueError, match="Cyclic terminal SOC infeasible"):
        simulate_energy_dispatch(
            demand_kwh=[0.0], renewable_available_kwh=[0.0],
            battery=battery, terminal_soc_ratio=0.75,
        )


def test_cyclic_dispatch_respects_grid_limit_and_reports_unmet_load():
    from src.decision_support.energy_balance import BatteryConfig, simulate_energy_dispatch
    battery = BatteryConfig(capacity_kwh=10.0, initial_soc_ratio=0.50)
    ts, metrics = simulate_energy_dispatch(
        demand_kwh=[5.0], renewable_available_kwh=[0.0], grid_import_limit_kw=2.0,
        battery=battery, terminal_soc_ratio=0.50,
    )
    assert ts.grid_import[0] == pytest.approx(2.0)
    assert ts.unmet_demand[0] == pytest.approx(3.0)
    assert metrics["final_soc_kwh"] == pytest.approx(5.0)


def test_battery_at_minimum_soc_cannot_discharge():
    from src.decision_support.energy_balance import BatteryConfig, simulate_energy_dispatch
    battery = BatteryConfig(capacity_kwh=10.0, initial_soc_ratio=0.10)
    ts, metrics = simulate_energy_dispatch([4.0], [0.0], battery=battery)
    assert ts.battery_discharge[0] == pytest.approx(0.0)
    assert ts.grid_import[0] == pytest.approx(4.0)
    assert metrics["final_soc_kwh"] == pytest.approx(1.0)


def test_battery_at_maximum_soc_curtails_unstorable_renewable():
    from src.decision_support.energy_balance import BatteryConfig, simulate_energy_dispatch
    battery = BatteryConfig(capacity_kwh=10.0, initial_soc_ratio=0.95)
    ts, metrics = simulate_energy_dispatch([0.0], [5.0], battery=battery)
    assert ts.battery_charge[0] == pytest.approx(0.0)
    assert ts.curtailment[0] == pytest.approx(5.0)
    assert metrics["final_soc_kwh"] == pytest.approx(9.5)


def test_noncyclic_dispatch_still_allows_intentional_initial_energy_use():
    from src.decision_support.energy_balance import BatteryConfig, simulate_energy_dispatch
    battery = BatteryConfig(capacity_kwh=10.0, initial_soc_ratio=0.50)
    _, metrics = simulate_energy_dispatch(
        demand_kwh=[4.0], renewable_available_kwh=[0.0], battery=battery
    )
    assert metrics["initial_soc_carry_in_discharge_kwh"] > 0.0
    assert metrics["final_soc_kwh"] < metrics["initial_soc_kwh"]
