"""Plain-language explanations derived from actual forecast and dispatch outputs."""
from typing import Any, Dict, Optional, Sequence
import math
import numpy as np


def explain_forecast(
    label: str,
    values: Sequence[float],
    unit: str,
    period: str,
    dataset: str,
    historical_mae: Optional[float] = None,
    mae_unit: Optional[str] = None,
    interval_coverage: Optional[float] = None,
    nominal_coverage: Optional[float] = None,
) -> Dict[str, str]:
    """Summarize a displayed historical forecast window without implying live prediction."""
    observed = np.asarray(values, dtype=float).reshape(-1)
    if observed.size == 0 or not np.isfinite(observed).all():
        return {
            "result": f"{label} is unavailable for this window.",
            "meaning": "No valid forecast values were returned.",
            "why": "A result cannot be interpreted without valid model output.",
            "reliability": "Historical performance is unavailable for this display.",
            "next": "Check that the required dataset and checkpoint are present.",
        }
    mean_value = float(np.mean(observed))
    peak_index = int(np.argmax(observed))
    peak_value = float(observed[peak_index])
    result = (
        f"Across {period}, the displayed {label.lower()} averages {mean_value:,.2f} {unit}; "
        f"the highest displayed value is {peak_value:,.2f} {unit}."
    )
    meaning = "These are one-step model estimates replayed on a held-out historical dataset, not a live reading or a future guarantee."
    why = f"The pattern in {label.lower()} helps show when energy supply or demand may be higher within this historical window."
    if historical_mae is None or not math.isfinite(float(historical_mae)):
        reliability = f"A verified held-out error metric for {dataset} is unavailable."
    else:
        reliability = (
            f"On the full held-out test partition from {dataset}, the model's mean absolute error was "
            f"{float(historical_mae):,.2f} {mae_unit or unit}. This is an average error, not a guarantee for each point."
        )
    if interval_coverage is not None and nominal_coverage is not None:
        reliability += (
            f" Historical test intervals contained {float(interval_coverage):.2%} of observations "
            f"against a {float(nominal_coverage):.0%} target."
        )
    next_step = (
        "Use the interval with the point estimate when comparing planning scenarios."
        if interval_coverage is not None
        else "Compare this historical pattern with the other forecast pages, keeping each dataset's location and period in view."
    )
    return {
        "result": result,
        "meaning": meaning,
        "why": why,
        "reliability": reliability,
        "next": next_step,
    }


def explain_coverage(
    nominal: float,
    observed: float,
    sample_count: int,
    dataset: str,
) -> Dict[str, Any]:
    """Explain empirical test coverage with an approximate count and target comparison."""
    if not (0.0 <= nominal <= 1.0 and 0.0 <= observed <= 1.0) or sample_count <= 0:
        raise ValueError("Coverage values must be fractions in [0, 1] and sample_count must be positive")
    inside_count = int(round(observed * sample_count))
    met_target = observed >= nominal
    status = "met or exceeded" if met_target else "fell below"
    return {
        "nominal": nominal,
        "observed": observed,
        "sample_count": sample_count,
        "inside_count": inside_count,
        "dataset": dataset,
        "met_target": met_target,
        "plain_text": (
            f"About {inside_count:,} of {sample_count:,} held-out observations from {dataset} "
            f"fell inside the historical range ({observed:.2%}). That {status} the intended "
            f"{nominal:.0%} level. This describes the evaluated test period; it does not guarantee future coverage."
        ),
    }


def explain_simulation(
    metrics: Dict[str, float],
    costs: Optional[Dict[str, float]],
    duration_hours: int,
    baseline_metrics: Optional[Dict[str, float]] = None,
    baseline_costs: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Explain modeled energy balance and compare it only with a supplied same-horizon baseline."""
    demand = float(metrics.get("total_demand_kwh", 0.0))
    grid = float(metrics.get("grid_import_kwh", 0.0))
    unmet = float(metrics.get("unmet_demand_kwh", 0.0))
    battery = float(metrics.get("battery_discharge_kwh", 0.0))
    curtailed = float(metrics.get("curtailment_kwh", 0.0))
    renewable_share = float(metrics.get("renewable_penetration", 0.0))
    if unmet > 1e-6:
        supply = f"The simulation leaves {unmet:,.2f} kWh of demand unmet over {duration_hours} hourly steps."
        next_step = "Review the hours with a shortfall and compare capacity or grid-limit changes in this same scenario."
    else:
        supply = f"The simulation supplies all {demand:,.2f} kWh of modeled demand over {duration_hours} hourly steps."
        next_step = "Compare another capacity mix over the same horizon before drawing a planning conclusion."
    if grid > 1e-6:
        supply += f" It imports {grid:,.2f} kWh from the grid."
    if battery > 1e-6:
        supply += f" The battery delivers {battery:,.2f} kWh during the run."
    supply += (
        f" Renewable energy supplied {renewable_share:.1%} of the electricity delivered "
        "(direct renewable use plus renewable-origin battery output); "
        f"{curtailed:,.2f} kWh of available generation is curtailed."
    )
    cost_text = "The cost estimate is unavailable because the cost model did not return a result."
    if costs and math.isfinite(float(costs.get("total_cost", float("nan")))):
        cost_text = (
            f"The modeled cost for this {duration_hours}-hour horizon is ${float(costs['total_cost']):,.2f}. "
            "It depends on the configured capital, grid-price, and penalty assumptions; it is not a quote."
        )
    comparison = "No same-horizon grid-only comparison is available."
    if baseline_metrics is not None:
        grid_delta = float(metrics.get("grid_import_kwh", 0.0)) - float(baseline_metrics.get("grid_import_kwh", 0.0))
        unmet_delta = float(metrics.get("unmet_demand_kwh", 0.0)) - float(baseline_metrics.get("unmet_demand_kwh", 0.0))
        comparison = (
            f"Compared with the grid-only run over the same {duration_hours} hours, grid imports changed by "
            f"{grid_delta:+,.2f} kWh and unmet demand changed by {unmet_delta:+,.2f} kWh."
        )
        if costs is not None and baseline_costs is not None:
            cost_delta = float(costs.get("total_cost", 0.0)) - float(baseline_costs.get("total_cost", 0.0))
            comparison += f" Modeled cost changed by ${cost_delta:+,.2f}."
    return {
        "supply": supply,
        "cost": cost_text,
        "comparison": comparison,
        "next": next_step,
        "reliability": "This is a deterministic simulation using synthetic, row-paired profiles from different places and periods; it is not a site forecast.",
    }
