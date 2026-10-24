"""
Visualization and plotting utilities for Stage 5: Decision Support & Energy-Mix Optimization.

Generates:
1. plot_energy_dispatch_trajectory: Hourly stacked dispatch chart showing renewable used, grid import, battery, unmet.
2. plot_uncertainty_dispatch_bounds: Comparative trajectory under conformal uncertainty bounds.
3. plot_capacity_optimization_surface: Optimization heatmaps and Pareto trade-off curves.
4. plot_scenario_comparison_metrics: Comparative grouped bar charts across uncertainty scenarios.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.decision_support.energy_balance import DispatchTimeSeries


def set_plot_style() -> None:
    """Set clean visual style for academic report figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    })


def plot_energy_dispatch_trajectory(
    dispatch: DispatchTimeSeries,
    output_path: Union[str, Path],
    timestamps: Optional[Union[pd.DatetimeIndex, np.ndarray, List]] = None,
    title: str = "Hourly Energy Dispatch and Supply Balance",
    max_hours: int = 168,  # Default 7 days (168 hours) for readability
) -> None:
    """
    Plot chronological stacked energy dispatch showing sources meeting demand.

    Args:
        dispatch: DispatchTimeSeries object.
        output_path: Destination image path.
        timestamps: Optional timestamp array.
        title: Plot title.
        max_hours: Maximum consecutive hours to display.
    """
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(dispatch.demand), max_hours)
    idx = np.arange(n)
    if timestamps is not None:
        x_axis = pd.to_datetime(pd.Series(timestamps[:n]))
    else:
        x_axis = idx

    d_slice = dispatch.demand[:n]
    r_used_slice = dispatch.renewable_used[:n]
    b_disch_slice = dispatch.battery_discharge[:n]
    g_slice = dispatch.grid_import[:n]
    u_slice = dispatch.unmet_demand[:n]
    c_slice = dispatch.curtailment[:n]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.0]})

    # Stacked dispatch serving demand
    # Stacking: Renewable Used -> Battery Discharge -> Grid Import -> Unmet Demand
    stack1 = r_used_slice
    stack2 = stack1 + b_disch_slice
    stack3 = stack2 + g_slice
    stack4 = stack3 + u_slice

    ax1.fill_between(x_axis, 0, stack1, label="Renewable Used Directly", color="#27ae60", alpha=0.85)
    if np.any(b_disch_slice > 1e-4):
        ax1.fill_between(x_axis, stack1, stack2, label="Battery Discharge", color="#2980b9", alpha=0.80)
    ax1.fill_between(x_axis, stack2, stack3, label="Grid Import", color="#e67e22", alpha=0.80)
    if np.any(u_slice > 1e-4):
        ax1.fill_between(x_axis, stack3, stack4, label="Unmet Demand (Loss of Load)", color="#c0392b", alpha=0.90)

    # Actual demand line overlay
    ax1.plot(x_axis, d_slice, color="#17202a", linewidth=1.8, linestyle="--", label="Total Demand (kWh)")

    ax1.set_ylabel("Power / Energy (kWh/h)")
    ax1.set_title(title, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, loc="upper right", ncol=2)

    # Subplot 2: Curtailment and Battery SOC
    ax2.plot(x_axis, c_slice, label="Renewable Curtailment (kWh)", color="#8e44ad", linewidth=1.5)
    if np.any(dispatch.battery_soc[:n] > 1e-4):
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_axis, dispatch.battery_soc[:n], label="Battery SOC (kWh)", color="#16a085", linewidth=1.5, linestyle=":")
        ax2_twin.set_ylabel("SOC (kWh)", color="#16a085")
        ax2_twin.grid(False)

    ax2.set_ylabel("Curtailment (kWh)")
    ax2.set_xlabel("Time (Hours)" if timestamps is None else "Timestamp")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, loc="upper left")

    if timestamps is not None:
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_uncertainty_dispatch_bounds(
    dispatch_central: DispatchTimeSeries,
    dispatch_stress: DispatchTimeSeries,
    dispatch_opt: DispatchTimeSeries,
    output_path: Union[str, Path],
    timestamps: Optional[Union[pd.DatetimeIndex, np.ndarray, List]] = None,
    title: str = "Energy Balance under Conformal Uncertainty Bounds",
    max_hours: int = 168,
) -> None:
    """Plot comparative dispatch envelopes across Central, Stress, and Favorable bounds."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(dispatch_central.demand), max_hours)
    x_axis = pd.to_datetime(pd.Series(timestamps[:n])) if timestamps is not None else np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    # Panel 1: Renewable Generation Uncertainty Envelope
    r_cent = dispatch_central.renewable_available[:n]
    r_stress = dispatch_stress.renewable_available[:n]
    r_opt = dispatch_opt.renewable_available[:n]

    ax1.fill_between(x_axis, r_stress, r_opt, color="#a9dfbf", alpha=0.55, label="Conformal Renewable Range [Lower, Upper]")
    ax1.plot(x_axis, r_cent, color="#1e8449", linewidth=2.0, label="Central Renewable Forecast (kWh)")
    ax1.plot(x_axis, dispatch_central.demand[:n], color="#2c3e50", linestyle="--", linewidth=1.5, label="Central Demand (kWh)")
    ax1.set_ylabel("Generation (kWh)")
    ax1.set_title(title, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, loc="upper right")

    # Panel 2: Grid Import & Unmet Demand Uncertainty Envelope
    g_cent = dispatch_central.grid_import[:n]
    g_stress = dispatch_stress.grid_import[:n]
    g_opt = dispatch_opt.grid_import[:n]

    ax2.fill_between(x_axis, g_opt, g_stress, color="#f8c471", alpha=0.50, label="Grid Import Range [Favorable, Stress]")
    ax2.plot(x_axis, g_cent, color="#d35400", linewidth=2.0, label="Central Grid Import (kWh)")

    # Show unmet demand in stress case if present
    u_stress = dispatch_stress.unmet_demand[:n]
    if np.any(u_stress > 1e-4):
        ax2.plot(x_axis, u_stress, color="#c0392b", linewidth=1.8, linestyle=":", label="Unmet Demand (Stress Case)")

    ax2.set_ylabel("Grid Import / Unmet (kWh)")
    ax2.set_xlabel("Time (Hours)" if timestamps is None else "Timestamp")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, loc="upper right")

    if timestamps is not None:
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_capacity_optimization_surface(
    candidates_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Renewable Energy Mix Optimization Surface",
) -> None:
    """Plot trade-off scatter plot of grid import vs total cost with renewable penetration color scale."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    df = candidates_df.copy()
    sc = ax.scatter(
        df["grid_import_kwh"],
        df["total_cost"],
        c=df["renewable_penetration"] * 100,
        cmap="viridis",
        s=60,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Renewable Penetration (%)", rotation=270, labelpad=15)

    # Highlight best candidate
    best_row = df.loc[df["score"].idxmin()]
    ax.scatter(
        best_row["grid_import_kwh"],
        best_row["total_cost"],
        color="#e74c3c",
        s=160,
        marker="*",
        edgecolor="black",
        label=f"Selected Mix: {best_row['solar_kw']}kW Solar, {best_row['wind_kw']}kW Wind, {best_row['battery_kwh']}kWh Batt",
        zorder=10,
    )

    ax.set_xlabel("Grid Import (kWh)")
    ax.set_ylabel("Illustrative Total Cost ($)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_scenario_comparison_metrics(
    metrics_summary_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Energy Balance Metrics Across Uncertainty Scenarios",
) -> None:
    """Grouped bar chart comparing key energy metrics across uncertainty scenarios."""
    set_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = metrics_summary_df.copy()
    categories = list(df["scenario"])
    x = np.arange(len(categories))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5.5))

    b1 = ax.bar(x - 1.5 * width, df["demand_kwh"], width, label="Demand", color="#34495e", alpha=0.85)
    b2 = ax.bar(x - 0.5 * width, df["renewable_kwh"], width, label="Renewable Gen", color="#27ae60", alpha=0.85)
    b3 = ax.bar(x + 0.5 * width, df["grid_import_kwh"], width, label="Grid Import", color="#e67e22", alpha=0.85)
    b4 = ax.bar(x + 1.5 * width, df["curtailment_kwh"], width, label="Curtailment", color="#8e44ad", alpha=0.85)

    ax.set_ylabel("Energy (kWh)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
