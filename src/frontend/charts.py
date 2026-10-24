"""
Interactive Plotly chart generators for the Streamlit decision-support dashboard.
Redesigned to match the clean light-theme renewable energy analytics prototype.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


FONT_FAMILY = "Plus Jakarta Sans, Inter, -apple-system, sans-serif"


def create_energy_mix_donut_chart(
    solar_val: float = 150.0,
    wind_val: float = 100.0,
    battery_val: float = 50.0,
    grid_val: float = 47.0,
    total_label: str = "347 kW",
    title: str = "System Snapshot – Illustrative Energy Mix",
    subtitle: str = "Optimal mix (example scenario)",
) -> go.Figure:
    """
    Create a clean, elegant Donut Chart matching the prototype's System Snapshot card.
    """
    labels = ["Solar", "Wind", "Battery Storage", "Grid Import"]
    values = [max(0.0, solar_val), max(0.0, wind_val), max(0.0, battery_val), max(0.0, grid_val)]
    colors = ["#F59E0B", "#0284C7", "#8B5CF6", "#64748B"]

    # Filter out near-zero categories
    filtered_labels = []
    filtered_values = []
    filtered_colors = []
    for lbl, val, col in zip(labels, values, colors):
        if val > 0:
            filtered_labels.append(lbl)
            filtered_values.append(val)
            filtered_colors.append(col)

    if not filtered_values:
        filtered_labels = ["Grid Import"]
        filtered_values = [100.0]
        filtered_colors = ["#64748B"]

    total_sum = sum(filtered_values)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=filtered_labels,
                values=filtered_values,
                hole=0.62,
                marker=dict(colors=filtered_colors, line=dict(color="#FFFFFF", width=3)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                hovertemplate="<b>%{label}</b><br>Capacity/Energy: %{value:.1f}<br>Share: %{percent}<extra></extra>",
                textfont=dict(size=12, color="#0F172A", family=FONT_FAMILY),
                showlegend=True,
            )
        ]
    )

    # Center annotation
    fig.add_annotation(
        text=f"<b>{total_label}</b><br><span style='font-size:11px; color:#64748B;'>TOTAL</span>",
        x=0.5,
        y=0.5,
        font=dict(size=16, color="#0F172A", family=FONT_FAMILY),
        showarrow=False,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px; color:#64748B;'>{subtitle}</span>",
            font=dict(size=15, color="#0F172A", family=FONT_FAMILY),
            x=0.02,
            y=0.96,
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11, color="#334155", family=FONT_FAMILY),
        ),
        margin=dict(l=20, r=90, t=55, b=20),
        template="plotly_white",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_overview_solar_pv_chart(
    timestamps: List[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Forecast vs Actual – Solar PV Output (Model A)",
    subtitle: str = "SKIPP'D Test Set Nowcasting (EXP_SOLAR_DL_S3)",
) -> go.Figure:
    """
    Create the high-contrast Forecast vs Actual line chart for the Overview page matching the prototype.
    Actual line: Dark navy (#0F172A)
    Predicted line: Amber/Orange dashed (#F59E0B)
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_true,
            mode="lines",
            name="Actual (kW)",
            line=dict(color="#0F172A", width=2.2),
            hovertemplate="Time: %{x}<br>Actual: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred,
            mode="lines",
            name="Predicted (kW)",
            line=dict(color="#F59E0B", width=2.0, dash="dash"),
            hovertemplate="Time: %{x}<br>Predicted: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px; color:#64748B;'>{subtitle}</span>",
            font=dict(size=15, color="#0F172A", family=FONT_FAMILY),
            x=0.02,
            y=0.96,
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            titlefont=dict(size=11, color="#64748B", family=FONT_FAMILY),
            tickfont=dict(size=10, color="#64748B", family=FONT_FAMILY),
        ),
        yaxis=dict(
            title="PV Output (kW)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            titlefont=dict(size=11, color="#64748B", family=FONT_FAMILY),
            tickfont=dict(size=10, color="#64748B", family=FONT_FAMILY),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#334155", family=FONT_FAMILY),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
        hovermode="x unified",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_mini_forecast_chart(
    timestamps: List[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_low: Optional[np.ndarray] = None,
    y_upp: Optional[np.ndarray] = None,
    title: str = "Forecast with Uncertainty",
    unit: str = "",
    theme: str = "green",
) -> go.Figure:
    """
    Create a compact preview forecast chart with uncertainty band for the Overview page (input_file_1.png).
    theme: 'green' for Wind, 'purple' for Demand, 'blue' for Solar
    """
    fig = go.Figure()

    theme_colors = {
        "green": {
            "forecast": "#10B981",
            "band": "rgba(16, 185, 129, 0.18)",
            "border": "#059669",
        },
        "purple": {
            "forecast": "#8B5CF6",
            "band": "rgba(139, 92, 246, 0.18)",
            "border": "#7C3AED",
        },
        "blue": {
            "forecast": "#0284C7",
            "band": "rgba(2, 132, 199, 0.18)",
            "border": "#0369A1",
        },
    }
    tc = theme_colors.get(theme, theme_colors["green"])

    # Uncertainty Ribbon
    if y_low is not None and y_upp is not None:
        fig.add_trace(
            go.Scatter(
                x=list(timestamps) + list(timestamps)[::-1],
                y=list(y_upp) + list(y_low)[::-1],
                fill="toself",
                fillcolor=tc["band"],
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="90% Prediction Interval",
            )
        )

    # Actual Observation Line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_true,
            mode="lines",
            name="Actual",
            line=dict(color="#0F172A", width=1.8),
            hovertemplate=f"Actual: <b>%{{y:.2f}} {unit}</b><extra></extra>",
        )
    )

    # Forecast Line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred,
            mode="lines",
            name="Forecast (Mean)",
            line=dict(color=tc["forecast"], width=1.8, dash="dash"),
            hovertemplate=f"Forecast: <b>%{{y:.2f}} {unit}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#0F172A", family=FONT_FAMILY),
            x=0.02,
            y=0.96,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(size=9, color="#64748B", family=FONT_FAMILY),
        ),
        yaxis=dict(
            title=unit,
            showgrid=True,
            gridcolor="#F1F5F9",
            titlefont=dict(size=10, color="#64748B", family=FONT_FAMILY),
            tickfont=dict(size=9, color="#64748B", family=FONT_FAMILY),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color="#334155", family=FONT_FAMILY),
        ),
        margin=dict(l=35, r=15, t=50, b=30),
        template="plotly_white",
        hovermode="x unified",
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_actual_vs_predicted_chart(
    timestamps: List[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    unit: str = "",
    y_low: Optional[np.ndarray] = None,
    y_upp: Optional[np.ndarray] = None,
    nominal_label: str = "90% Conformal Interval",
) -> go.Figure:
    """Create an interactive Actual vs Predicted line chart with clean light styling."""
    fig = go.Figure()

    # Optional Uncertainty Envelope
    if y_low is not None and y_upp is not None:
        fig.add_trace(
            go.Scatter(
                x=list(timestamps) + list(timestamps)[::-1],
                y=list(y_upp) + list(y_low)[::-1],
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.16)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=nominal_label,
            )
        )

    # Actual Target Line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_true,
            mode="lines",
            name="Actual Observation",
            line=dict(color="#0F172A", width=2.2),
            hovertemplate="Time: %{x}<br>Actual: <b>%{y:.2f} " + unit + "</b><extra></extra>",
        )
    )

    # Predicted Forecast Line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred,
            mode="lines",
            name="Forecast (Deep Learning)",
            line=dict(color="#0284C7", width=2.2, dash="solid"),
            hovertemplate="Time: %{x}<br>Forecast: <b>%{y:.2f} " + unit + "</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="#0F172A", family=FONT_FAMILY)),
        xaxis=dict(
            title="Timestamp",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            titlefont=dict(size=11, color="#64748B", family=FONT_FAMILY),
            tickfont=dict(size=10, color="#64748B", family=FONT_FAMILY),
        ),
        yaxis=dict(
            title=f"Value ({unit})",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            titlefont=dict(size=11, color="#64748B", family=FONT_FAMILY),
            tickfont=dict(size=10, color="#64748B", family=FONT_FAMILY),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#334155", family=FONT_FAMILY),
        ),
        margin=dict(l=40, r=30, t=65, b=40),
        template="plotly_white",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_dispatch_dashboard_chart(
    timestamps: List[Any],
    demand: np.ndarray,
    renewable_used: np.ndarray,
    battery_discharge: np.ndarray,
    grid_import: np.ndarray,
    curtailment: np.ndarray,
    battery_soc: Optional[np.ndarray] = None,
    title: str = "Hourly Energy Dispatch & Supply Mix",
) -> go.Figure:
    """Create stacked dispatch area chart showing how demand is served hour by hour."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Energy Supply & Demand Tracking (kWh)", "Renewable Curtailment & Battery SOC (kWh)"),
        row_heights=[0.65, 0.35],
    )

    # Subplot 1: Stacked Supply Areas
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=renewable_used,
            mode="lines",
            name="Renewable Direct Used",
            stackgroup="supply",
            line=dict(color="#10B981", width=0.5),
            fillcolor="rgba(16, 185, 129, 0.75)",
            hovertemplate="Renewable Used: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=battery_discharge,
            mode="lines",
            name="Battery Discharge",
            stackgroup="supply",
            line=dict(color="#8B5CF6", width=0.5),
            fillcolor="rgba(139, 92, 246, 0.75)",
            hovertemplate="Battery Discharge: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=grid_import,
            mode="lines",
            name="Grid Import",
            stackgroup="supply",
            line=dict(color="#F59E0B", width=0.5),
            fillcolor="rgba(245, 158, 11, 0.75)",
            hovertemplate="Grid Import: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=demand,
            mode="lines",
            name="Total Demand",
            line=dict(color="#0F172A", width=2.5, dash="dash"),
            hovertemplate="Total Demand: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Curtailment & SOC
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=curtailment,
            name="Curtailed Renewable",
            marker=dict(color="rgba(239, 68, 68, 0.75)"),
            hovertemplate="Curtailment: %{y:.2f} kWh<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if battery_soc is not None:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=battery_soc,
                mode="lines",
                name="Battery Stored Energy (kWh)",
                line=dict(color="#0284C7", width=2),
                hovertemplate="Battery SOC: %{y:.2f} kWh<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="#0F172A", family=FONT_FAMILY)),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(family=FONT_FAMILY)),
        margin=dict(l=40, r=30, t=70, b=40),
        template="plotly_white",
        hovermode="x unified",
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Energy (kWh)", showgrid=True, gridcolor="#F1F5F9", row=1, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", showgrid=True, gridcolor="#F1F5F9", row=2, col=1)
    fig.update_xaxes(title_text="Timestamp", showgrid=True, gridcolor="#F1F5F9", row=2, col=1)

    return fig


def create_scenario_comparison_bar_chart(scenario_records: List[Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing metrics across conformal uncertainty scenarios."""
    df = pd.DataFrame(scenario_records)
    if df.empty or "scenario" not in df.columns:
        return go.Figure()

    scenarios = df["scenario"].tolist()
    demand_vals = df["demand_kwh"].tolist() if "demand_kwh" in df.columns else []
    renewable_vals = df["renewable_kwh"].tolist() if "renewable_kwh" in df.columns else []
    grid_vals = df["grid_import_kwh"].tolist() if "grid_import_kwh" in df.columns else []
    curtail_vals = df["curtailment_kwh"].tolist() if "curtailment_kwh" in df.columns else []

    fig = go.Figure()

    fig.add_trace(go.Bar(name="Total Demand (kWh)", x=scenarios, y=demand_vals, marker_color="#334155"))
    fig.add_trace(go.Bar(name="Available Renewable (kWh)", x=scenarios, y=renewable_vals, marker_color="#10B981"))
    fig.add_trace(go.Bar(name="Grid Import (kWh)", x=scenarios, y=grid_vals, marker_color="#F59E0B"))
    fig.add_trace(go.Bar(name="Curtailment (kWh)", x=scenarios, y=curtail_vals, marker_color="#EF4444"))

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Energy Quantities Across Conformal Uncertainty Scenarios</b>", font=dict(size=16, color="#0F172A", family=FONT_FAMILY)),
        xaxis=dict(title="Scenario", linecolor="#E2E8F0"),
        yaxis=dict(title="Total Energy (kWh)", showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family=FONT_FAMILY)),
        margin=dict(l=40, r=30, t=60, b=40),
        template="plotly_white",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_tradeoff_pareto_chart(candidates_df: pd.DataFrame) -> go.Figure:
    """Create interactive Pareto trade-off chart between Total Cost and Grid Dependence."""
    if candidates_df.empty:
        return go.Figure()

    fig = go.Figure()

    # All candidate points
    fig.add_trace(
        go.Scatter(
            x=candidates_df["renewable_penetration"] * 100,
            y=candidates_df["total_cost"],
            mode="markers",
            name="Candidate Mixes (120)",
            marker=dict(
                size=candidates_df["battery_kwh"] / 15 + 8,
                color=candidates_df["grid_import_kwh"],
                colorscale="Viridis",
                colorbar=dict(title="Grid Import (kWh)"),
                showscale=True,
                opacity=0.75,
                line=dict(color="#1E293B", width=0.5),
            ),
            hovertemplate=(
                "Solar: %{customdata[0]} kW<br>"
                "Wind: %{customdata[1]} kW<br>"
                "Battery: %{customdata[2]} kWh<br>"
                "Ren. Penetration: %{x:.1f}%<br>"
                "Total Cost: $%{y:,.2f}<br>"
                "Grid Import: %{customdata[3]:,.1f} kWh<extra></extra>"
            ),
            customdata=candidates_df[["solar_kw", "wind_kw", "battery_kwh", "grid_import_kwh"]].values,
        )
    )

    # Highlight 100% Grid Baseline
    base_row = candidates_df[(candidates_df["solar_kw"] == 0) & (candidates_df["wind_kw"] == 0) & (candidates_df["battery_kwh"] == 0)]
    if not base_row.empty:
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[base_row["total_cost"].values[0]],
                mode="markers+text",
                name="100% Grid Baseline",
                text=["Baseline"],
                textposition="top right",
                marker=dict(symbol="diamond", size=14, color="#EF4444", line=dict(color="#7F1D1D", width=2)),
            )
        )

    # Highlight Cost-Optimal
    min_cost_idx = candidates_df["total_cost"].idxmin()
    best_cost_row = candidates_df.loc[min_cost_idx]
    fig.add_trace(
        go.Scatter(
            x=[best_cost_row["renewable_penetration"] * 100],
            y=[best_cost_row["total_cost"]],
            mode="markers+text",
            name="Cost-Optimal Mix",
            text=["Cost-Optimal"],
            textposition="top left",
            marker=dict(symbol="star", size=16, color="#10B981", line=dict(color="#064E3B", width=2)),
        )
    )

    fig.update_layout(
        title=dict(text="<b>Candidate Renewable Energy Mixes: Cost vs Autonomy Trade-Off</b>", font=dict(size=16, color="#0F172A", family=FONT_FAMILY)),
        xaxis=dict(title="Renewable Penetration (% of Total Demand)", showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(title="Total Horizon Amortized Cost ($)", showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family=FONT_FAMILY)),
        margin=dict(l=40, r=30, t=60, b=40),
        template="plotly_white",
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
