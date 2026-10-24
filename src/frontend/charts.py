"""
Interactive Plotly chart generators for the Streamlit decision-support dashboard.
Redesigned with high-contrast, WCAG AA compliant light-theme tokens,
clear axis units, sensible timestamp formatting, and readable legends.
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
    Clearly distinguishes Solar, Wind, Battery Storage, and Grid Import with units.
    """
    labels = ["Solar PV", "Wind Turbine", "Battery Storage", "Grid Import"]
    values = [max(0.0, solar_val), max(0.0, wind_val), max(0.0, battery_val), max(0.0, grid_val)]
    colors = ["#F59E0B", "#0369A1", "#7C3AED", "#64748B"]

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

    fig = go.Figure(
        data=[
            go.Pie(
                labels=filtered_labels,
                values=filtered_values,
                hole=0.62,
                marker=dict(colors=filtered_colors, line=dict(color="#FFFFFF", width=3)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                hovertemplate="<b>%{label}</b><br>Configured Capacity/Power: %{value:.1f} kW<br>Share: %{percent}<extra></extra>",
                textfont=dict(size=12, color="#172033", family=FONT_FAMILY),
                showlegend=True,
            )
        ]
    )

    # Center annotation
    fig.add_annotation(
        text=f"<b>{total_label}</b><br><span style='font-size:11px; color:#475569;'>TOTAL MIX</span>",
        x=0.5,
        y=0.5,
        font=dict(size=16, color="#172033", family=FONT_FAMILY),
        showarrow=False,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px; color:#475569;'>{subtitle}</span>",
            font=dict(size=15, color="#172033", family=FONT_FAMILY),
            x=0.02,
            y=0.96,
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11, color="#172033", family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        margin=dict(l=20, r=100, t=55, b=20),
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
    Create the high-contrast Forecast vs Actual line chart for the Overview page.
    Actual line: Dark high-contrast slate/navy (#172033)
    Predicted line: Amber/Orange dashed (#F59E0B)
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_true,
            mode="lines",
            name="Actual Observation (kW)",
            line=dict(color="#172033", width=2.4),
            hovertemplate="Time: %{x}<br>Actual PV: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred,
            mode="lines",
            name="Predicted Nowcast (kW)",
            line=dict(color="#F59E0B", width=2.2, dash="dash"),
            hovertemplate="Time: %{x}<br>Predicted PV: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px; color:#475569;'>{subtitle}</span>",
            font=dict(size=15, color="#172033", family=FONT_FAMILY),
            x=0.02,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text="Time of Day (PST)", font=dict(size=11, color="#172033", family=FONT_FAMILY)),
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=10, color="#475569", family=FONT_FAMILY),
            nticks=12,
            exponentformat="none",
            showexponent="none",
        ),
        yaxis=dict(
            title=dict(text="PV Output Power (kW)", font=dict(size=11, color="#172033", family=FONT_FAMILY)),
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=10, color="#475569", family=FONT_FAMILY),
            rangemode="tozero",
            exponentformat="none",
            showexponent="none",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#172033", family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        margin=dict(l=45, r=25, t=65, b=45),
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
    Create a compact preview forecast chart with uncertainty band for the Overview page.
    theme: 'green' for Wind, 'purple' for Demand, 'blue' for Solar
    """
    fig = go.Figure()

    theme_colors = {
        "green": {
            "forecast": "#047857",
            "band": "rgba(4, 120, 87, 0.18)",
            "border": "#065F46",
        },
        "purple": {
            "forecast": "#7C3AED",
            "band": "rgba(124, 58, 237, 0.18)",
            "border": "#6D28D9",
        },
        "blue": {
            "forecast": "#0369A1",
            "band": "rgba(3, 105, 161, 0.18)",
            "border": "#075985",
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
            name="Actual Observation",
            line=dict(color="#172033", width=2.0),
            hovertemplate=f"Actual: <b>%{{y:.2f}} {unit}</b><extra></extra>",
        )
    )

    # Forecast Line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred,
            mode="lines",
            name="Model Forecast",
            line=dict(color=tc["forecast"], width=2.0, dash="dash"),
            hovertemplate=f"Forecast: <b>%{{y:.2f}} {unit}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#172033", family=FONT_FAMILY),
            x=0.02,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text="Continuous Timeline (Hours)", font=dict(size=10, color="#172033", family=FONT_FAMILY)),
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=9, color="#475569", family=FONT_FAMILY),
            nticks=8,
            exponentformat="none",
            showexponent="none",
        ),
        yaxis=dict(
            title=dict(text=f"Value ({unit})", font=dict(size=10, color="#172033", family=FONT_FAMILY)),
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=9, color="#475569", family=FONT_FAMILY),
            exponentformat="none",
            showexponent="none",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color="#172033", family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        margin=dict(l=40, r=20, t=55, b=35),
        template="plotly_white",
        hovermode="x unified",
        height=265,
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
    nominal_label: str = "90% Conformal Prediction Interval",
) -> go.Figure:
    """
    Create an interactive Actual vs Predicted line chart with clean light styling.
    Ensures high contrast between actual observations, deep learning forecasts, and uncertainty intervals.
    """
    fig = go.Figure()

    # Optional Uncertainty Envelope
    if y_low is not None and y_upp is not None:
        fig.add_trace(
            go.Scatter(
                x=list(timestamps) + list(timestamps)[::-1],
                y=list(y_upp) + list(y_low)[::-1],
                fill="toself",
                fillcolor="rgba(4, 120, 87, 0.16)",
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
            name=f"Actual Observation ({unit})",
            line=dict(color="#172033", width=2.4),
            hovertemplate="Time: %{x}<br>Actual: <b>%{y:.2f} " + unit + "</b><extra></extra>",
        )
    )

    # Predicted Forecast Line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred,
            mode="lines",
            name=f"Forecast (Deep Learning) ({unit})",
            line=dict(color="#0369A1", width=2.2, dash="solid"),
            hovertemplate="Time: %{x}<br>Forecast: <b>%{y:.2f} " + unit + "</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="#172033", family=FONT_FAMILY)),
        xaxis=dict(
            title=dict(text="Continuous Timeline (Date / Hour)", font=dict(size=11, color="#172033", family=FONT_FAMILY)),
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=10, color="#475569", family=FONT_FAMILY),
            nticks=10,
            exponentformat="none",
            showexponent="none",
        ),
        yaxis=dict(
            title=dict(text=f"Physical Quantity ({unit})", font=dict(size=11, color="#172033", family=FONT_FAMILY)),
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=10, color="#475569", family=FONT_FAMILY),
            exponentformat="none",
            showexponent="none",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#172033", family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        margin=dict(l=45, r=30, t=65, b=45),
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
    """
    Create stacked dispatch area chart showing how demand is served hour by hour.
    Individual energy carriers are clearly distinguishable with distinct high-contrast colors.
    """
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
            name="Direct Renewable Used",
            stackgroup="supply",
            line=dict(color="#047857", width=0.5),
            fillcolor="rgba(4, 120, 87, 0.75)",
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
            line=dict(color="#7C3AED", width=0.5),
            fillcolor="rgba(124, 58, 237, 0.75)",
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
            name="Total Demand Target",
            line=dict(color="#172033", width=2.6, dash="dash"),
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
            marker=dict(color="rgba(239, 68, 68, 0.80)"),
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
                line=dict(color="#0369A1", width=2.2),
                hovertemplate="Battery SOC: %{y:.2f} kWh<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="#172033", family=FONT_FAMILY)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(family=FONT_FAMILY, size=11, color="#172033"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        margin=dict(l=45, r=30, t=70, b=45),
        template="plotly_white",
        hovermode="x unified",
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Energy (kWh)", title_font=dict(color="#172033"), showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", tickfont=dict(color="#475569"), row=1, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", title_font=dict(color="#172033"), showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", tickfont=dict(color="#475569"), row=2, col=1)
    fig.update_xaxes(title_text="Hourly Simulation Timeline", title_font=dict(color="#172033"), showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", tickfont=dict(color="#475569"), nticks=10, row=2, col=1)

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
    fig.add_trace(go.Bar(name="Available Renewable (kWh)", x=scenarios, y=renewable_vals, marker_color="#047857"))
    fig.add_trace(go.Bar(name="Grid Import (kWh)", x=scenarios, y=grid_vals, marker_color="#F59E0B"))
    fig.add_trace(go.Bar(name="Curtailment (kWh)", x=scenarios, y=curtail_vals, marker_color="#EF4444"))

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Energy Quantities Across Conformal Uncertainty Scenarios</b>", font=dict(size=16, color="#172033", family=FONT_FAMILY)),
        xaxis=dict(title="Uncertainty Scenario", linecolor="#CBD5E1", tickfont=dict(size=11, color="#172033", family=FONT_FAMILY)),
        yaxis=dict(title="Total Energy (kWh)", showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", tickfont=dict(size=10, color="#475569", family=FONT_FAMILY)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family=FONT_FAMILY, size=11, color="#172033"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        margin=dict(l=45, r=30, t=65, b=45),
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
                colorbar=dict(
                    title=dict(text="Grid Import (kWh)", font=dict(color="#172033", size=11)),
                    tickfont=dict(color="#475569"),
                ),
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
            marker=dict(symbol="star", size=16, color="#047857", line=dict(color="#064E3B", width=2)),
        )
    )

    fig.update_layout(
        title=dict(text="<b>Candidate Renewable Energy Mixes: Cost vs Autonomy Trade-Off</b>", font=dict(size=16, color="#172033", family=FONT_FAMILY)),
        xaxis=dict(
            title=dict(text="Renewable Penetration (% of Total Demand)", font=dict(size=11, color="#172033")),
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=10, color="#475569"),
        ),
        yaxis=dict(
            title=dict(text="Total Horizon Amortized Cost ($)", font=dict(size=11, color="#172033")),
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=10, color="#475569"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family=FONT_FAMILY, size=11, color="#172033"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        margin=dict(l=45, r=30, t=65, b=45),
        template="plotly_white",
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
