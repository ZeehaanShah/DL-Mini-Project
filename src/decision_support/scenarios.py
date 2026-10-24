"""
Scenario definition and provenance tracking for energy feasibility analysis.

Enforces strict metadata tagging:
- REAL_OBSERVATION: Empirical measurements from verified sensors/meters.
- MODEL_PREDICTION: Direct point forecasts from trained deep-learning models.
- ENGINEERING_PROXY: Physical transfer models (e.g., turbine power curve, solar PV derate).
- SYNTHETIC_SCENARIO: Combined multi-source benchmark for PoC mix optimization.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ProvenanceType(Enum):
    REAL_OBSERVATION = "REAL_OBSERVATION"
    MODEL_PREDICTION = "MODEL_PREDICTION"
    ENGINEERING_PROXY = "ENGINEERING_PROXY"
    SYNTHETIC_SCENARIO = "SYNTHETIC_SCENARIO"


@dataclass
class ScenarioMetadata:
    """Metadata documenting input provenance, units, and limitations."""
    scenario_id: str
    provenance: ProvenanceType
    description: str
    location: str
    time_coverage: str
    n_hours: int
    demand_unit: str = "kWh"
    solar_unit: str = "kWh"
    wind_unit: str = "kWh"
    is_colocated: bool = False
    limitations: List[str] = field(default_factory=list)


@dataclass
class RenewableResourceProfiles:
    """Normalized or capacity-scaled resource profiles (all hourly kWh or normalized [0, 1])."""
    solar_normalized: np.ndarray      # Hourly generation per kW of installed solar capacity [kWh / kWp / h]
    wind_normalized: np.ndarray       # Hourly generation per kW of installed wind capacity [kWh / kW / h]
    timestamps: Optional[pd.DatetimeIndex] = None


@dataclass
class EnergyScenario:
    """Complete energy scenario inputs ready for dispatch simulation and optimization."""
    metadata: ScenarioMetadata
    demand_kwh: np.ndarray
    solar_capacity_kw: float
    wind_capacity_kw: float
    solar_generation_kwh: np.ndarray
    wind_generation_kwh: np.ndarray
    total_renewable_kwh: np.ndarray
    timestamps: Optional[pd.DatetimeIndex] = None

    def __post_init__(self):
        n = len(self.demand_kwh)
        if len(self.solar_generation_kwh) != n or len(self.wind_generation_kwh) != n:
            raise ValueError(
                f"Length mismatch: demand ({n}), solar ({len(self.solar_generation_kwh)}), "
                f"wind ({len(self.wind_generation_kwh)}) must be identical"
            )


def convert_irradiance_to_solar_pv_kwh(
    irradiance_wh_m2: Union[np.ndarray, list],
    capacity_kw: float,
    standard_test_irradiance: float = 1000.0,
    performance_ratio: float = 0.78,
) -> np.ndarray:
    """
    Convert surface solar irradiance (Wh/m²) to estimated solar PV electricity generation (kWh).

    Engineering Physical Conversion Model:
        E_PV[t] = Capacity_kW * (G[t] / G_STC) * PR

    Assumptions:
    1. Flat-plate crystalline silicon PV modules derated by a nominal performance ratio (PR = 0.78).
    2. Linear irradiance response normalized to Standard Test Conditions (STC: 1000 W/m² = 1000 Wh/m² per hour).
    3. Inverter clipping occurs at 1.0 * Capacity_kW.
    4. Temperature derating and soiling losses are encapsulated in PR.

    Args:
        irradiance_wh_m2: Hourly surface downward shortwave irradiance in Wh/m².
        capacity_kw: Installed PV array nameplate capacity in kW.
        standard_test_irradiance: STC reference irradiance in Wh/m² per hour (default: 1000.0).
        performance_ratio: Typical system derate factor (default: 0.78).

    Returns:
        1D float64 array of estimated solar electricity generation in kWh.
    """
    irr = np.asarray(irradiance_wh_m2, dtype=np.float64)
    if np.any(irr < -1e-6):
        raise ValueError("Irradiance cannot be negative")
    if capacity_kw < 0.0:
        raise ValueError("Solar capacity must be non-negative")
    if standard_test_irradiance <= 0.0 or performance_ratio <= 0.0:
        raise ValueError("STC irradiance and performance ratio must be strictly positive")

    # Hourly generation in kWh
    norm_irr = np.maximum(irr, 0.0) / standard_test_irradiance
    raw_gen = capacity_kw * norm_irr * performance_ratio

    # Inverter capacity clipping
    gen_clipped = np.minimum(raw_gen, capacity_kw)
    return gen_clipped.astype(np.float64)


def convert_wind_speed_to_generation_kwh(
    wind_speed_ms: Union[np.ndarray, list],
    capacity_kw: float,
    reference_rated_power_kw: float = 2000.0,
    cut_in_speed: float = 3.0,
    rated_speed: float = 12.0,
    cut_out_speed: float = 25.0,
) -> np.ndarray:
    """
    Convert wind speed (m/s) to estimated electrical generation (kWh) for a specified wind capacity.

    Uses an illustrative cubic turbine power curve scaled to the target installed capacity:
    - Below cut-in (3 m/s) or above cut-out (25 m/s): 0 kW
    - Between cut-in and rated (12 m/s): cubic progression P_rated * ((v^3 - v_in^3) / (v_rated^3 - v_in^3))
    - Between rated and cut-out: constant rated capacity

    Args:
        wind_speed_ms: Hub-height wind speed in m/s.
        capacity_kw: Installed wind plant capacity in kW.
        reference_rated_power_kw: Reference turbine rating (default: 2000 kW).
        cut_in_speed: Cut-in speed in m/s.
        rated_speed: Rated speed in m/s.
        cut_out_speed: Cut-out storm speed in m/s.

    Returns:
        1D float64 array of estimated wind electricity generation in kWh per hour.
    """
    ws = np.asarray(wind_speed_ms, dtype=np.float64)
    if np.any(ws < -1e-6):
        raise ValueError("Wind speed cannot be negative")
    if capacity_kw < 0.0:
        raise ValueError("Wind capacity must be non-negative")

    power_fraction = np.zeros_like(ws, dtype=np.float64)

    # Cubic aerodynamic regime
    cubic_mask = (ws >= cut_in_speed) & (ws < rated_speed)
    num = (ws[cubic_mask] ** 3) - (cut_in_speed ** 3)
    denom = (rated_speed ** 3) - (cut_in_speed ** 3)
    power_fraction[cubic_mask] = num / denom

    # Rated regime
    rated_mask = (ws >= rated_speed) & (ws < cut_out_speed)
    power_fraction[rated_mask] = 1.0

    # Scaled generation in kWh (Power * 1h)
    gen = capacity_kw * power_fraction
    return gen.astype(np.float64)


def build_synthetic_hybrid_benchmark_scenario(
    solar_irradiance_wh_m2: np.ndarray,
    wind_speed_ms: np.ndarray,
    demand_kwh: np.ndarray,
    solar_capacity_kw: float = 50.0,
    wind_capacity_kw: float = 100.0,
    n_hours: int = 240,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> EnergyScenario:
    """
    Construct a clearly labelled illustrative multimodal energy-mix benchmark.

    DISCLAIMER: Combines characteristic temporal signatures from Model B (Mumbai irradiance),
    Model C (Gujarat wind speed), and Model D (Orlando demand) into a synchronized 1-hour
    interval sequence for decision-support algorithm validation.
    Does NOT claim real-world co-location or synchronized sensor acquisition.
    """
    n = min(len(solar_irradiance_wh_m2), len(wind_speed_ms), len(demand_kwh), n_hours)

    sol_irr = solar_irradiance_wh_m2[:n]
    ws = wind_speed_ms[:n]
    dem = demand_kwh[:n]

    solar_gen = convert_irradiance_to_solar_pv_kwh(sol_irr, capacity_kw=solar_capacity_kw)
    wind_gen = convert_wind_speed_to_generation_kwh(ws, capacity_kw=wind_capacity_kw)
    total_ren = solar_gen + wind_gen

    meta = ScenarioMetadata(
        scenario_id="SCENARIO_MULTIMODAL_POC",
        provenance=ProvenanceType.SYNTHETIC_SCENARIO,
        description="Illustrative multimodal solar/wind/demand benchmark for decision-support PoC",
        location="Illustrative Hybrid Benchmark (Composite profiles)",
        time_coverage=f"{n} consecutive hourly simulation time-steps",
        n_hours=n,
        demand_unit="kWh",
        solar_unit="kWh",
        wind_unit="kWh",
        is_colocated=False,
        limitations=[
            "Composite scenario constructed from independent geographic datasets (NASA Mumbai, NREL Gujarat, BDG2 Florida).",
            "Demonstrates decision-support optimization algorithms only; does NOT represent physical installation measurements.",
            "Solar generation uses STC linear derate model (PR=0.78); wind generation uses generic 2.0 MW cubic power curve proxy.",
        ],
    )

    ts_slice = timestamps[:n] if timestamps is not None else None

    return EnergyScenario(
        metadata=meta,
        demand_kwh=dem,
        solar_capacity_kw=solar_capacity_kw,
        wind_capacity_kw=wind_capacity_kw,
        solar_generation_kwh=solar_gen,
        wind_generation_kwh=wind_gen,
        total_renewable_kwh=total_ren,
        timestamps=ts_slice,
    )
