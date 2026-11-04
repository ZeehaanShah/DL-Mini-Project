"""
Master Streamlit Web Application for:
“A Multimodal, Uncertainty-Aware Decision-Support Framework for Location-Specific Renewable Energy Feasibility and Energy-Mix Optimization”

Redesigned with high-contrast, WCAG AA compliant light-theme styling, accessible sidebar navigation,
zero-indent HTML rendering (eliminating code-block </div> artefacts), informative slider guides,
and comprehensive scientific methodology disclosures.

Authors: DL MiniProject Research Team
Framework: PyTorch, Split Conformal Prediction, Deterministic Dispatch Simulation, Streamlit, Plotly
"""
from pathlib import Path
import sys
from typing import Dict, Any, List

# Ensure project root is in sys.path and Windows DLLs are initialized
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils.torch_loader  # noqa: F401
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Import frontend loaders, components, and charts
from src.frontend.model_loaders import (
    load_registry_data,
    get_verified_metric,
    load_uncertainty_summary,
    load_decision_support_summary,
    load_feasibility_scenarios_table,
    predict_solar_pv_from_image,
    get_solar_nowcasting_test_evaluation,
    get_weather_test_evaluation,
    get_wind_test_evaluation,
    get_demand_test_evaluation,
    simulate_custom_system,
    get_benchmark_time_series,
)
from src.frontend.components import (
    apply_custom_styles,
    render_top_navbar,
    render_sidebar_scope,
    render_header,
    render_model_card,
    render_quick_action_card,
    render_control_context_card,
    render_disclaimer,
    render_provenance_badge,
)
from src.frontend.charts import (
    create_energy_mix_donut_chart,
    create_overview_solar_pv_chart,
    create_mini_forecast_chart,
    create_actual_vs_predicted_chart,
    create_dispatch_dashboard_chart,
    create_scenario_comparison_bar_chart,
    create_tradeoff_pareto_chart,
)
from src.decision_support.optimizer import (
    CostModel,
    OptimizationObjective,
    optimize_energy_mix,
)
from src.decision_support.uncertainty_eval import evaluate_uncertainty_feasibility
from src.decision_support.energy_balance import BatteryConfig
from src.frontend.explanations import explain_coverage, explain_forecast, explain_simulation


# -----------------------------------------------------------------------------
# APP CONFIGURATION & THEMING
# -----------------------------------------------------------------------------
def _metric_text(experiment_id: str, metric: str, decimals: int, unit: str = "") -> str:
    """Format a verified test metric from the experiment registry."""
    value = get_verified_metric(experiment_id, metric)
    if value is None:
        return "Unavailable"
    formatted = f"{value:,.{decimals}f}"
    return f"{formatted} {unit}".strip()


def _metric_count(experiment_id: str) -> str:
    """Format the verified test sample count from the experiment registry."""
    value = get_verified_metric(experiment_id, "n_samples")
    return f"{int(value):,} samples" if value is not None else "Unavailable"


def _uncertainty_card_html(title: str, summary: dict, unit: str, metric_suffix: str) -> str:
    """Build a calibration/test coverage card from the generated uncertainty artifact."""
    eval_80 = summary.get("eval_80", {})
    eval_90 = summary.get("eval_90", {})
    cal_n = summary.get("cal_samples")
    test_n = summary.get("test_samples")
    if not eval_80 or not eval_90:
        return f'<div class="chart-card"><b>{title}</b><p>Generated coverage results are unavailable.</p></div>'
    return f"""<div class="chart-card">
      <div style="font-weight:700; color:var(--text-primary); font-size:1.1rem; margin-bottom:.5rem;">{title}</div>
      <div style="font-size:.85rem; color:var(--text-secondary); margin-bottom:.85rem;">Calibration samples: {cal_n:,} | Test samples: {test_n:,}</div>
      <table class="styled-table"><thead><tr><th>Metric</th><th>80% nominal</th><th>90% nominal</th></tr></thead>
      <tbody>
      <tr><td>Empirical test coverage</td><td>{eval_80.get('empirical_coverage', float('nan')):.2%}</td><td>{eval_90.get('empirical_coverage', float('nan')):.2%}</td></tr>
      <tr><td>Mean interval width</td><td>{eval_80.get('mpiw_'+metric_suffix, float('nan')):.2f} {unit}</td><td>{eval_90.get('mpiw_'+metric_suffix, float('nan')):.2f} {unit}</td></tr>
      <tr><td>Winkler interval score</td><td>{eval_80.get('winkler_score', float('nan')):.2f}</td><td>{eval_90.get('winkler_score', float('nan')):.2f}</td></tr>
      </tbody></table></div>"""


def _render_explanation(title: str, explanation: Dict[str, str]) -> None:
    """Show a result interpretation before its technical detail."""
    st.subheader(title)
    for key, label in (("result", "Result"), ("meaning", "What it means"), ("why", "Why it matters"),
                       ("reliability", "How reliable it is"), ("next", "What to consider next")):
        value = explanation.get(key)
        if value:
            st.markdown(f"**{label}:** {value}")


st.set_page_config(
    page_title="Renewable Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom high-contrast light-theme styling
apply_custom_styles()


# -----------------------------------------------------------------------------
# STATEFUL SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
PAGE_OPTIONS = [
    "1. Energy overview",
    "2. Solar image estimate",
    "3. Solar resource forecast",
    "4. Wind forecast",
    "5. Electricity demand",
    "6. How reliable are these estimates?",
    "7. Build and compare an energy mix",
    "8. Methodology and limitations",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGE_OPTIONS[0]

def set_page(page_name: str) -> None:
    st.session_state["nav_page"] = page_name


# Sidebar brand header
st.sidebar.markdown(
    '<div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 1.25rem; padding-bottom: 0.85rem; border-bottom: 1px solid var(--border);">'
    '<div style="width: 36px; height: 36px; background: linear-gradient(135deg, var(--brand) 0%, var(--brand-strong) 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: var(--text-on-brand); font-weight: bold; font-size: 1.2rem;">⚡</div>'
    '<div>'
    '<div style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary); line-height: 1.2;">Energy Intelligence</div>'
    '<div style="font-size: 0.75rem; color: var(--text-secondary); font-weight: 500;">Decision-Support System</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

current_idx = PAGE_OPTIONS.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in PAGE_OPTIONS else 0

selected_page = st.sidebar.radio(
    "Navigation Menu",
    PAGE_OPTIONS,
    index=current_idx,
    label_visibility="collapsed",
)
st.session_state["nav_page"] = selected_page

# Bottom project scope checklist in sidebar
render_sidebar_scope()


# -----------------------------------------------------------------------------
# TOP APPLICATION BANNER
# -----------------------------------------------------------------------------
col_nav_left, col_nav_right = st.columns([0.72, 0.28])
with col_nav_left:
    st.markdown(
        '<div style="display: flex; align-items: center; gap: 0.85rem; padding: 0.25rem 0;">'
        '<div class="brand-logo">⚡</div>'
        '<div>'
        '<div class="brand-title">Renewable Energy Intelligence</div>'
        '<div class="brand-subtitle">A Multimodal, Uncertainty-Aware Decision-Support Framework</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with col_nav_right:
    c_status, c_act = st.columns([0.48, 0.52])
    with c_status:
        st.markdown(
            '<div style="margin-top: 6px;">'
            '<span class="navbar-status-badge">'
            '<span class="status-dot"></span>'
            'Historical model results'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c_act:
        if st.button("▶ Run Analysis", key="btn_top_run_analysis", type="primary", use_container_width=True):
            set_page("7. Build and compare an energy mix")
            st.rerun()

st.markdown("<div style='border-bottom: 1px solid var(--border); margin-bottom: 1.25rem; margin-top: 0.5rem;'></div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: ENERGY OVERVIEW
if st.session_state["nav_page"] == "1. Energy overview":
    render_header(
        "Energy overview",
        "A historical synthetic example showing how renewable supply, storage, and grid electricity balance modeled demand.",
    )
    with st.expander("What does this dashboard predict?"):
        st.write(
            "The forecasting pages replay frozen models on held-out historical data. "
            "The energy overview combines separate historical datasets by row position to demonstrate a simulation workflow. "
            "It is not a live forecast or a real-site feasibility study."
        )

    summary = load_decision_support_summary()
    try:
        settings = summary.get("simulation_parameters", {})
        saved = summary.get("cost_optimal_configuration", {})
        capacities = saved.get("capacities", {})
        solar_kw = float(capacities["solar_kw"])
        wind_kw = float(capacities["wind_kw"])
        battery_kwh = float(capacities["battery_kwh"])
        duration = int(settings.get("n_hours", 240))
        grid_limit = float(settings.get("grid_import_limit_kw", 100.0))
        dispatch, metrics, costs = simulate_custom_system(
            solar_kw, wind_kw, battery_kwh,
            grid_import_limit_kw=grid_limit,
            duration_hours=duration,
        )
        _, baseline_metrics, baseline_costs = simulate_custom_system(
            0.0, 0.0, 0.0,
            grid_import_limit_kw=grid_limit,
            duration_hours=duration,
        )
    except Exception as exc:
        st.warning("The overview cannot be calculated because one or more required datasets or model checkpoints are unavailable.")
        with st.expander("Technical error details"):
            st.code(str(exc))
    else:
        st.caption(
            f"Saved cost-focused example: {solar_kw:g} kW solar, {wind_kw:g} kW wind, "
            f"{battery_kwh:g} kWh battery, {grid_limit:g} kW grid limit, over {duration} hourly steps."
        )
        explanation = explain_simulation(
            metrics, costs, duration, baseline_metrics=baseline_metrics, baseline_costs=baseline_costs
        )
        st.info(explanation["supply"])
        st.markdown(f"**Compared with grid-only supply:** {explanation['comparison']}")
        st.markdown(f"**Why it matters:** Grid imports, battery output, and shortfalls show which simulated sources are needed to serve demand.")
        st.caption(explanation["reliability"])
        st.caption(f"**What to consider next:** {explanation['next']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Solar energy available ({duration} h)", f"{metrics['solar_generation_available_kwh']:,.1f} kWh")
        m2.metric(f"Wind energy available ({duration} h)", f"{metrics['wind_generation_available_kwh']:,.1f} kWh")
        m3.metric(f"Electricity needed ({duration} h)", f"{metrics['total_demand_kwh']:,.1f} kWh")
        m4.metric("Electricity supplied", f"{metrics['served_demand_kwh']:,.1f} kWh")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Grid electricity required", f"{metrics['grid_import_kwh']:,.1f} kWh")
        m6.metric("Possible supply shortfall", f"{metrics['unmet_demand_kwh']:,.2f} kWh")
        m7.metric("Battery contribution", f"{metrics['battery_discharge_kwh']:,.1f} kWh")
        m8.metric("Estimated scenario cost", "$" + f"{costs['total_cost']:,.2f}")

        st.subheader("How the simulated demand is supplied")
        n_show = min(48, len(dispatch.demand))
        fig = create_dispatch_dashboard_chart(
            timestamps=[f"Hour {i + 1}" for i in range(n_show)],
            demand=dispatch.demand[:n_show],
            renewable_used=dispatch.renewable_used[:n_show],
            battery_discharge=dispatch.battery_discharge[:n_show],
            grid_import=dispatch.grid_import[:n_show],
            unmet_demand=dispatch.unmet_demand[:n_show],
            curtailment=dispatch.curtailment[:n_show],
            battery_soc=dispatch.battery_soc[:n_show],
            title="First 48 hours of the simulated energy balance",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Estimated cost depends on the project’s illustrative capital, grid-price, curtailment, and unmet-demand assumptions; "
            "it is not an installation quote."
        )
        with st.expander("Detailed results and assumptions"):
            st.json({"simulation_metrics": metrics, "costs": costs, "grid_only_baseline_metrics": baseline_metrics})
        render_disclaimer(
            "This benchmark pairs Stanford solar-image, Mumbai irradiance, Gujarat wind, and Orlando demand data by array position. "
            "Their locations and timestamps are incompatible, so the results demonstrate software behavior only.",
            warning=True,
        )
# PAGE 2: SOLAR PV NOWCASTING (MODEL A)
# =============================================================================
elif st.session_state["nav_page"] == "2. Solar image estimate":
    render_header(
        "Estimate solar power from a sky image",
        "Choose a built-in example or upload a sky image to estimate PV output in kilowatts.",
    )
    st.info(
        "The image model estimates power associated with the training benchmark. It does not read a meter or measure a real installation. "
        "Lighting, camera position, cloud conditions, and differences from the training setup can affect the result."
    )

    # Model Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_model_card("Model A", "Test-Set MAE", _metric_text("EXP_SOLAR_DL_S3_TEST", "mae", 4, "kW"), "ResNet-18 (Fine-Tuned)", color_theme="green", icon="☀️", badge="TEST SET")
    with m2:
        render_model_card("Model A", "Test-Set RMSE", _metric_text("EXP_SOLAR_DL_S3_TEST", "rmse", 4, "kW"), "ResNet-18 (Fine-Tuned)", color_theme="blue", icon="📐", badge="TEST SET")
    with m3:
        render_model_card("Model A", "Test Samples", _metric_count("EXP_SOLAR_DL_S3_TEST"), "Official SKIPP'D test parquet", color_theme="teal", icon="🧪", badge="TEST SET")
    with m4:
        render_model_card("Model A", "Test-Set R²", _metric_text("EXP_SOLAR_DL_S3_TEST", "r2", 4), "ResNet-18 (Fine-Tuned)", color_theme="purple", icon="📈", badge="TEST SET")

    render_provenance_badge(
        "Stanford SKIPP'D Ground Fisheye Camera",
        "Vision Benchmark",
        "Stanford, CA, USA (37.43°N, 122.17°W)",
        "2017–2019 (14,003 test images)",
        "kW (Kilowatts)",
    )

    st.markdown('<div class="section-heading"><span>📷</span><span>Interactive Sky Image Inference</span></div>', unsafe_allow_html=True)
    st.markdown("Test the frozen ResNet-18 nowcaster using either verified test sky images or by uploading an image.")

    tab_sample, tab_upload = st.tabs(["🖼️ Built-in held-out examples", "⬆️ Upload a sky image"])

    sample_dir = PROJECT_ROOT / "data" / "sample_images"
    sample_files = sorted(list(sample_dir.glob("*.png"))) if sample_dir.exists() else []

    with tab_sample:
        if sample_files:
            sample_options = {
                f"Example {index + 1}: held-out PV value {sample.stem.rsplit('_pv_', 1)[-1].removesuffix('kW')} kW": sample
                for index, sample in enumerate(sample_files)
            }
            chosen_label = st.radio("Select a test sky image sample:", list(sample_options.keys()))
            chosen_file = sample_options[chosen_label]

            col_img, col_pred = st.columns([1, 1.2])
            with col_img:
                st.image(str(chosen_file), caption=f"Selected Input: {chosen_label}", width=260)
            with col_pred:
                if st.button("⚡ Run Inference on Selected Sample", type="primary", use_container_width=True):
                    with st.spinner("Executing ResNet-18 forward pass on CPU..."):
                        pred_kw = predict_solar_pv_from_image(chosen_file)
                        st.markdown(
                            '<div class="chart-card" style="border-top: 4px solid var(--brand);">'
                            '<div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">PREDICTED ROOFTOP PV GENERATION</div>'
                            f'<div style="font-size: 2.2rem; font-weight: 800; color: var(--text-primary); margin: 0.35rem 0;">{pred_kw:.2f} kW</div>'
                            '<div style="font-size: 0.82rem; color: var(--brand); font-weight: 700;">Model: ResNet-18 (EXP_SOLAR_DL_S3)</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.warning("No sample images found in data/sample_images/.")

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a sky image (PNG or JPG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            col_up_img, col_up_pred = st.columns([1, 1.2])
            with col_up_img:
                st.image(uploaded_file, caption="Uploaded Sky Image", width=260)
            with col_up_pred:
                if st.button("📷 Upload Sky Image & Predict", type="primary", use_container_width=True):
                    with st.spinner("Processing image and executing model..."):
                        try:
                            img = Image.open(uploaded_file).convert("RGB")
                            pred_kw = predict_solar_pv_from_image(img)
                            st.markdown(
                                '<div class="chart-card" style="border-top: 4px solid var(--blue);">'
                                '<div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">PREDICTED PV POWER</div>'
                                f'<div style="font-size: 2.2rem; font-weight: 800; color: var(--text-primary); margin: 0.35rem 0;">{pred_kw:.2f} kW</div>'
                                '<div style="font-size: 0.82rem; color: var(--text-secondary);">Resized to 64×64 and ImageNet normalized.</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            st.error("This file could not be read as a supported sky image. Please choose a valid PNG or JPEG image.")
                            with st.expander("Technical error details"):
                                st.code(str(e))

    # Historical Test Evaluation Chart
    st.markdown('<div class="section-heading"><span>📈</span><span>Historical Test-Set Evaluation Trajectory</span></div>', unsafe_allow_html=True)
    solar_eval = get_solar_nowcasting_test_evaluation(max_samples=260)
    if solar_eval and "timestamps" in solar_eval:
        fig_solar = create_overview_solar_pv_chart(
            timestamps=solar_eval["timestamps"],
            y_true=solar_eval["y_true"],
            y_pred=solar_eval["y_pred"],
            title="Model A: Actual vs ResNet-18 Predictions (SKIPP'D Test Morning)",
            subtitle="Verified test sequence from 07:38 to 12:03 PST (1-minute resolution)",
        )
        st.plotly_chart(fig_solar, use_container_width=True)


# =============================================================================
# PAGE 3: SOLAR IRRADIANCE FORECASTING (MODEL B)
# =============================================================================
elif st.session_state["nav_page"] == "3. Solar resource forecast":
    render_header(
        "Solar resource forecast",
        "Review one-hour-ahead estimates of incoming solar energy on the held-out Mumbai weather record.",
    )
    st.info(
        "Irradiance describes incoming solar energy per area. It is not the same as electricity a solar panel can generate; "
        "panel area, orientation, efficiency, shading, and other conditions also matter."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_model_card("Model B", "Test-Set MAE", _metric_text("EXP_NASA_DL_W1_TEST", "mae", 4, "Wh/m²"), "WeatherLSTM W1 (h=32, L=1)", color_theme="blue", icon="▦", badge="TEST SET")
    with m2:
        render_model_card("Model B", "Test-Set RMSE", _metric_text("EXP_NASA_DL_W1_TEST", "rmse", 4, "Wh/m²"), "WeatherLSTM W1", color_theme="green", icon="📐", badge="TEST SET")
    with m3:
        render_model_card("Model B", "Test-Set R²", _metric_text("EXP_NASA_DL_W1_TEST", "r2", 4), "WeatherLSTM W1", color_theme="teal", icon="⭐", badge="TEST SET")
    with m4:
        render_model_card("Model B", "Test Samples", _metric_count("EXP_NASA_DL_W1_TEST"), "Current 24-hour embargo split", color_theme="purple", icon="🧪", badge="TEST SET")

    render_provenance_badge(
        "NASA POWER All-Sky Surface Downward Solar",
        "Numerical Reanalysis",
        "Mumbai, India (19.076°N, 72.878°E)",
        "2023 (8,760 continuous hourly records)",
        "Wh/m² (Watt-hours per square meter)",
    )

    st.markdown('<div class="section-heading"><span>🎛️</span><span>Visualization Controls</span></div>', unsafe_allow_html=True)
    
    # Context Guide for Slider
    render_control_context_card(
        param_name="Displayed Test-Set Window (Hours)",
        description="Select the continuous sequence length of hourly observations from the held-out test partition (2023) to visualize. Changing this control alters the temporal window to inspect diurnal solar cycles and model tracking fidelity across multiple consecutive days.",
        unit="Hours (h)",
        min_val=24,
        max_val=336,
        current_val="User Selected (Step: 24h)",
        impact="Expands or contracts the displayed time-series window. The model's forecast horizon remains fixed at 1-hour ahead (t+1).",
        icon="⏱️",
    )

    c_slider1, c_slider2 = st.columns([1, 2])
    with c_slider1:
        st.markdown(
            '<div style="background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:0.75rem; margin-top:0.35rem;">'
            '<div style="font-size:0.82rem; color:var(--text-secondary); font-weight:600;">FORECAST HORIZON</div>'
            '<div style="font-size:1.15rem; font-weight:800; color:var(--text-primary);">1-Hour Ahead (t+1)</div>'
            '<div style="font-size:0.78rem; color:var(--brand);">Fixed recurrent single-step inference</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c_slider2:
        n_samples = st.slider("Displayed Time Window (Hours):", min_value=24, max_value=336, value=96, step=24)
        st.markdown(f"<div style='font-size:0.82rem; color:var(--text-secondary); font-weight:600;'>Active window: <b>{n_samples} hours</b> ({n_samples/24:.1f} days of continuous test series)</div>", unsafe_allow_html=True)

    full_eval_b = get_weather_test_evaluation(max_samples=10_000)
    if full_eval_b and "timestamps" in full_eval_b:
        eval_b = dict(full_eval_b)
        for array_key in ("timestamps", "y_true", "y_pred", "raw_y_pred"):
            eval_b[array_key] = full_eval_b[array_key][:n_samples]
        explanation = explain_forecast(
            "solar irradiance", eval_b["y_pred"], "Wh/m²", f"{len(eval_b['y_pred'])} displayed hours",
            "NASA POWER Mumbai 2023 test partition",
            historical_mae=get_verified_metric("EXP_NASA_DL_W1_TEST", "mae"), mae_unit="Wh/m²",
        )
        _render_explanation("How to read this forecast", explanation)
        fig_b = create_actual_vs_predicted_chart(
            timestamps=eval_b["timestamps"],
            y_true=eval_b["y_true"],
            y_pred=eval_b["y_pred"],
            title=f"Hourly Solar Irradiance: Actual vs Predicted (Showing {n_samples} Hours)",
            unit="Wh/m²",
        )
        st.plotly_chart(fig_b, use_container_width=True)

        changed_n = int(full_eval_b["changed_prediction_count"])
        test_n = len(full_eval_b["timestamps"])
        st.caption(
            f"The plot shows nonnegative projected outputs for {n_samples} selected hours. "
            f"Across the full held-out test set, projection changed {changed_n:,}/{test_n:,} predictions "
            f"({full_eval_b['invalid_prediction_fraction']:.2%}); raw and projected scores for the same "
            "test rows are listed below."
        )
        with st.expander("Technical results: raw model vs nonnegative output"):
            raw_metrics = {
                "MAE": get_verified_metric("EXP_NASA_DL_W1_TEST", "mae"),
                "RMSE": get_verified_metric("EXP_NASA_DL_W1_TEST", "rmse"),
                "R²": get_verified_metric("EXP_NASA_DL_W1_TEST", "r2"),
            }
            comparison = pd.DataFrame(
                [
                    {
                        "Prediction series": "Raw model output",
                        "MAE (Wh/m²)": raw_metrics["MAE"],
                        "RMSE (Wh/m²)": raw_metrics["RMSE"],
                        "R²": raw_metrics["R²"],
                        "Changed by projection": 0,
                        "Negative predictions": full_eval_b["invalid_prediction_count"],
                    },
                    {
                        "Prediction series": "Nonnegative projected output",
                        "MAE (Wh/m²)": full_eval_b["projected_mae"],
                        "RMSE (Wh/m²)": full_eval_b["projected_rmse"],
                        "R²": full_eval_b["projected_r2"],
                        "Changed by projection": changed_n,
                        "Negative predictions": 0,
                    },
                ]
            )
            st.dataframe(comparison, hide_index=True, use_container_width=True)
            st.caption(
                f"Both rows use the same {test_n:,} held-out NASA POWER observations "
                f"({pd.Timestamp(full_eval_b['timestamps'][0])} to {pd.Timestamp(full_eval_b['timestamps'][-1])}). "
                "The raw row is the frozen checkpoint benchmark; projection is a physical output constraint, not retraining."
            )

        residuals = eval_b["y_true"] - eval_b["y_pred"]
        st.markdown(
            f'<div style="background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:0.65rem 1rem; font-size:0.88rem; color:var(--text-primary);">'
            f'<b>Projected-output residuals</b>: Mean Residual: <code>{np.mean(residuals):+.2f} Wh/m²</code> &nbsp;|&nbsp; '
            f'Max Absolute Error: <code>{np.max(np.abs(residuals)):.2f} Wh/m²</code> &nbsp;|&nbsp; '
            f'Standard Deviation of Error: <code>{np.std(residuals):.2f} Wh/m²</code>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 4: WIND FORECASTING (MODEL C)
# =============================================================================
elif st.session_state["nav_page"] == "4. Wind forecast":
    render_header(
        "Wind forecast",
        "Review historical one-hour-ahead wind-speed estimates and an optional illustrative turbine-output conversion.",
    )
    wind_uncertainty = load_uncertainty_summary().get("wind", {})
    wind_coverage_90 = float(wind_uncertainty.get("eval_90", {}).get("empirical_coverage", float("nan")))
    wind_test_n = int(wind_uncertainty.get("test_samples", 0) or 0)
    wind_full_eval = get_wind_test_evaluation(max_samples=wind_test_n or 1242)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_model_card("Model C", "Test Wind MAE", _metric_text("EXP_WIND_DL_C2_TEST", "mae", 4, "m/s"), "WindLSTM C2 (h=64, L=1)", color_theme="teal", icon="🌀", badge="TEST SET")
    with m2:
        render_model_card("Model C", "Test Wind RMSE", _metric_text("EXP_WIND_DL_C2_TEST", "rmse", 4, "m/s"), "WindLSTM C2", color_theme="blue", icon="📐", badge="TEST SET")
    with m3:
        render_model_card("Model C", "Observed interval coverage", f"{wind_coverage_90:.2%}" if np.isfinite(wind_coverage_90) else "Unavailable", f"90% target; {wind_test_n:,} test observations", color_theme="green", icon="🛡️", badge="TEST RESULTS")
    with m4:
        power_mae = wind_full_eval.get("power_mae_kw")
        render_model_card("Model C", "Test power-proxy MAE", f"{power_mae:,.2f} kW" if power_mae is not None else "Unavailable", f"{wind_test_n:,} held-out points; illustrative curve", color_theme="purple", icon="⚡", badge="PROXY")

    render_provenance_badge(
        "NREL India Wind Toolkit 100m AGL",
        "Meso-scale WRF Simulation",
        "Gujarat Offshore/Coastal (23.24°N, 69.66°E)",
        "2014 (8,760 hours)",
        "m/s (Meters per second)",
    )

    st.markdown('<div class="section-heading"><span>🎛️</span><span>Visualization Controls</span></div>', unsafe_allow_html=True)
    
    st.markdown("**Target Variable Selection**: Choose between physical hub-height wind velocity or electrical generator output.")
    var_choice = st.radio("Display Target:", ["Physical Wind Speed (m/s) with 90% Conformal Interval", "Estimated 2.0 MW Turbine Power Proxy (kW)"], horizontal=True)

    # Context Guide for Slider
    render_control_context_card(
        param_name="Displayed Wind Test Window (Hours)",
        description="Duration of consecutive test-set hours from the 2014 Gujarat offshore split to plot. Allows inspection of multi-day wind ramps, diurnal breeze patterns, and conformal uncertainty band width across varying wind regimes.",
        unit="Hours (h)",
        min_val=24,
        max_val=336,
        current_val="User Selected (Step: 24h)",
        impact="Wider windows illustrate synoptic meteorological shifts; narrower windows highlight hourly gust variations.",
        icon="🌬️",
    )

    n_wind_samples = st.slider("Displayed Time Window (Hours):", min_value=24, max_value=336, value=96, step=24, key="wind_slider")
    st.markdown(f"<div style='font-size:0.82rem; color:var(--text-secondary); font-weight:600;'>Active window: <b>{n_wind_samples} hours</b> ({n_wind_samples/24:.1f} days of continuous wind records)</div>", unsafe_allow_html=True)

    eval_c = get_wind_test_evaluation(max_samples=n_wind_samples)
    if eval_c and "timestamps" in eval_c:
        explanation = explain_forecast(
            "wind speed", eval_c["y_pred"], "m/s", f"{len(eval_c['y_pred'])} displayed hours",
            "NREL Gujarat 2014 test partition",
            historical_mae=get_verified_metric("EXP_WIND_DL_C2_TEST", "mae"), mae_unit="m/s",
            interval_coverage=wind_coverage_90 if np.isfinite(wind_coverage_90) else None,
            nominal_coverage=0.90 if np.isfinite(wind_coverage_90) else None,
        )
        _render_explanation("How to read this forecast", explanation)
        if "Wind Speed" in var_choice:
            fig_c = create_actual_vs_predicted_chart(
                timestamps=eval_c["timestamps"],
                y_true=eval_c["y_true"],
                y_pred=eval_c["y_pred"],
                y_low=eval_c["y_low_90"],
                y_upp=eval_c["y_upp_90"],
                title="100m Wind Speed: Actual vs Forecast with 90% Conformal Uncertainty Interval",
                unit="m/s",
                nominal_label="90% Conformal Prediction Interval",
            )
        else:
            fig_c = create_actual_vs_predicted_chart(
                timestamps=eval_c["timestamps"],
                y_true=eval_c["p_true_kw"],
                y_pred=eval_c["p_pred_kw"],
                y_low=eval_c["p_low_kw"],
                y_upp=eval_c["p_upp_kw"],
                title="Illustrative 2 MW Turbine Power Proxy: Estimated Power vs Persistence",
                unit="kW",
                nominal_label="90% transformed wind-speed interval",
            )
        st.plotly_chart(fig_c, use_container_width=True)

    render_disclaimer(
        "The optional turbine conversion uses a simplified 2.0 MW power curve (cut-in 3.0 m/s, rated 12.0 m/s, cut-out 25.0 m/s). "
        "It estimates output from forecast wind speed; it is not measured generation from a real turbine.",
        warning=True,
    )


# =============================================================================
# PAGE 5: ELECTRICITY DEMAND FORECASTING (MODEL D)
# =============================================================================
elif st.session_state["nav_page"] == "5. Electricity demand":
    render_header(
        "Electricity demand",
        "Review historical one-hour-ahead electricity-use estimates for one Orlando building meter.",
    )
    demand_uncertainty = load_uncertainty_summary().get("demand", {})
    eval_d = get_demand_test_evaluation(max_samples=10000)
    rolling_demand_metrics = eval_d.get("rolling_interval_metrics_90", {}) if eval_d else {}
    fixed_demand_metrics = eval_d.get("fixed_interval_metrics_90", {}) if eval_d else {}
    demand_coverage_90 = float(rolling_demand_metrics.get("coverage", float("nan")))
    demand_test_n = int(rolling_demand_metrics.get("n_samples", 0) or 0)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_model_card("Model D", "Test Demand MAE", _metric_text("EXP_DEMAND_DL_D2_TEST", "mae", 4, "kWh"), "DemandLSTM D2 (h=64, L=1)", color_theme="purple", icon="🏢", badge="TEST SET")
    with m2:
        render_model_card("Model D", "Test Demand RMSE", _metric_text("EXP_DEMAND_DL_D2_TEST", "rmse", 4, "kWh"), "DemandLSTM D2", color_theme="blue", icon="📐", badge="TEST SET")
    with m3:
        render_model_card("Model D", "Test-Set R²", _metric_text("EXP_DEMAND_DL_D2_TEST", "r2", 4), "DemandLSTM D2", color_theme="teal", icon="⭐", badge="TEST SET")
    with m4:
        render_model_card("Model D", "Rolling interval coverage", f"{demand_coverage_90:.2%}" if np.isfinite(demand_coverage_90) else "Unavailable", f"90% target; {demand_test_n:,} sequential test forecasts", color_theme="green", icon="🛡️", badge="PREQUENTIAL")

    if fixed_demand_metrics and rolling_demand_metrics:
        st.caption(
            f"Fixed split-conformal baseline on the same test rows: {fixed_demand_metrics['coverage']:.2%} "
            f"coverage, {fixed_demand_metrics['mpiw']:.2f} kWh mean width. The rolling 168-hour series "
            f"updates only after each actual test-hour outcome is observed: {rolling_demand_metrics['coverage']:.2%} "
            f"coverage, {rolling_demand_metrics['mpiw']:.2f} kWh mean width."
        )

    render_provenance_badge(
        "Building Data Genome Project 2 (Panther_education_Rosalie)",
        "Real Smart Meter",
        "Orlando, FL, USA (28.54°N, 81.38°W)",
        "2016–2017 (17,544 continuous hours)",
        "kWh (Kilowatt-hours)",
    )

    st.markdown('<div class="section-heading"><span>🎛️</span><span>Visualization Controls</span></div>', unsafe_allow_html=True)

    # Context Guide for Slider
    render_control_context_card(
        param_name="Displayed Electricity Demand Window (Hours)",
        description="Continuous sequence length of hourly smart meter electricity consumption to display from the held-out test split. Illustrates building thermal inertia, weekday daytime occupancy spikes, and weekend baseload.",
        unit="Hours (h)",
        min_val=24,
        max_val=336,
        current_val="User Selected (Step: 24h)",
        impact="Adjusts the temporal zoom level. The model's forecast horizon is fixed at 1-step ahead (1 hour ahead).",
        icon="🏢",
    )

    n_dem_samples = st.slider("Displayed Time Window (Hours):", min_value=24, max_value=336, value=96, step=24, key="dem_slider")
    st.markdown(f"<div style='font-size:0.82rem; color:var(--text-secondary); font-weight:600;'>Active window: <b>{n_dem_samples} hours</b> ({n_dem_samples/24:.1f} days of continuous building meter data)</div>", unsafe_allow_html=True)

    if eval_d and "timestamps" in eval_d:
        chart_slice = slice(0, n_dem_samples)
        explanation = explain_forecast(
            "electricity demand", eval_d["y_pred"][chart_slice], "kWh", f"{n_dem_samples} displayed hours",
            "BDG2 Orlando building test partition",
            historical_mae=get_verified_metric("EXP_DEMAND_DL_D2_TEST", "mae"), mae_unit="kWh",
            interval_coverage=demand_coverage_90 if np.isfinite(demand_coverage_90) else None,
            nominal_coverage=0.90 if np.isfinite(demand_coverage_90) else None,
        )
        _render_explanation("How to read this forecast", explanation)
        fig_d = create_actual_vs_predicted_chart(
            timestamps=eval_d["timestamps"][chart_slice],
            y_true=eval_d["y_true"][chart_slice],
            y_pred=eval_d["y_pred"][chart_slice],
            y_low=eval_d["y_low_90"][chart_slice],
            y_upp=eval_d["y_upp_90"][chart_slice],
            title="Demand forecast with prequential rolling 90% interval (168-hour calibration window)",
            unit="kWh",
            nominal_label="Rolling 90% interval",
        )
        st.plotly_chart(fig_d, use_container_width=True)

    if np.isfinite(demand_coverage_90):
        coverage_note = explain_coverage(0.90, demand_coverage_90, demand_test_n, "BDG2 Orlando building test partition")
        st.warning(
            f"Historical interval coverage was {demand_coverage_90:.2%} against a 90% target "
            f"({demand_test_n:,} test observations). {coverage_note['plain_text']}"
        )
    render_disclaimer(
        "Rolling intervals use only residuals available before each forecast; each observed test outcome updates later intervals. "
        "This is a historical prequential replay, not an exchangeability guarantee or a promise about future readings. "
        "The fixed split-conformal baseline is shown separately; calibration precedes the test period by the documented embargo.",
        warning=True,
    )


# =============================================================================
# PAGE 6: UNCERTAINTY ANALYSIS (STAGE 4)
# =============================================================================
elif st.session_state["nav_page"] == "6. How reliable are these estimates?":
    render_header(
        "How reliable are these estimates?",
        "See how often past held-out observations fell within the displayed forecast ranges, and where those ranges missed their target.",
    )

    unc_summary = load_uncertainty_summary()
    for key, dataset_name, label in (
        ("wind", "NREL Gujarat 2014 wind test partition", "Wind speed"),
        ("demand", "BDG2 Orlando building demand test partition", "Electricity demand"),
    ):
        result = unc_summary.get(key, {})
        n_test = int(result.get("test_samples", 0) or 0)
        for interval_key in ("eval_80", "eval_90"):
            interval = result.get(interval_key, {})
            if n_test and interval:
                coverage = explain_coverage(
                    float(interval["nominal_coverage"]), float(interval["empirical_coverage"]), n_test, dataset_name
                )
                st.markdown(f"**{label}, {int(coverage['nominal'] * 100)}% target:** {coverage['plain_text']}")

    st.markdown('<div class="section-heading"><span>🎓</span><span>What Are Prediction Intervals?</span></div>', unsafe_allow_html=True)
    st.markdown(
        "A forecast range shows a band around the central estimate. Historical coverage is the share of evaluated observations "
        "that landed inside that band. A 90% target is not ‘90% accurate’ and does not guarantee that a future value will be inside it."
    )
    with st.expander("Technical details: split conformal intervals"):
        st.markdown(
            "The model’s central estimate is expanded by a residual quantile measured on a separate chronological calibration set. "
            "The usual marginal guarantee depends on calibration and future observations being exchangeable; the seasonal shift in "
            "the demand series weakens that assumption. Coverage across individual hours does not guarantee simultaneous coverage of a whole horizon."
        )
        st.latex(r"P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha")

    demand_prequential_eval = get_demand_test_evaluation(max_samples=10000)
    rolling_result = demand_prequential_eval.get("rolling_interval_metrics_90", {}) if demand_prequential_eval else {}
    if rolling_result:
        st.info(
            f"Demand rolling 168-hour prequential replay (reported separately from the fixed split cards below): "
            f"{rolling_result['coverage']:.2%} observed coverage at a 90% target, "
            f"{rolling_result['mpiw']:.2f} kWh mean width, N={rolling_result['n_samples']:,}. "
            "Each actual updates only subsequent intervals; temporal dependence means no nominal-coverage guarantee applies."
        )

    # Table of Calibration Results
    st.markdown('<div class="section-heading"><span>📊</span><span>Empirical Test Coverage & Calibration Metrics</span></div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown(
            _uncertainty_card_html("🍃 Model C: Wind Speed Calibration", unc_summary.get("wind", {}), "m/s", "ms"),
            unsafe_allow_html=True,
        )
    with t_col2:
        st.markdown(
            _uncertainty_card_html("🏢 Model D: Electricity Demand Calibration", unc_summary.get("demand", {}), "kWh", "kwh"),
            unsafe_allow_html=True,
        )

    # Next Stage Guidance Button
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("⚡ Proceed to Energy-Mix Optimizer (Model E) ➔", type="primary", use_container_width=True):
        set_page("7. Build and compare an energy mix")
        st.rerun()

    render_disclaimer(
        "<b>Marginal vs. Joint Coverage Caution</b>: Conformal guarantees hold marginally for each individual time step. "
        "They do not guarantee simultaneous joint band coverage across all 240 consecutive hours of a multi-day dispatch horizon.",
        warning=True,
    )


# =============================================================================
# PAGE 7: ENERGY-MIX OPTIMIZER (MODEL E)
# =============================================================================
elif st.session_state["nav_page"] == "7. Build and compare an energy mix":
    render_header(
        "Build and compare an energy mix",
        "Change the system capacities and see how the modeled energy balance changes over a historical benchmark window.",
    )
    st.warning(
        "This is a synthetic software benchmark. Solar, wind, and building-demand profiles come from different places and years "
        "and are paired by row position. It is not proof that a system will work at a real site."
    )

    # Preset Action Buttons
    st.markdown('<div class="section-heading"><span>⚡</span><span>Quick Scenario Presets</span></div>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)
    saved_scenarios = load_decision_support_summary()
    cost_preset = saved_scenarios.get("cost_optimal_configuration", {}).get("capacities", {})
    grid_preset = saved_scenarios.get("grid_optimal_configuration", {}).get("capacities", {})
    cost_preset = {
        "solar_kw": int(round(float(cost_preset.get("solar_kw", 150)))),
        "wind_kw": int(round(float(cost_preset.get("wind_kw", 0)))),
        "battery_kwh": int(round(float(cost_preset.get("battery_kwh", 200)))),
    }
    grid_preset = {
        "solar_kw": int(round(float(grid_preset.get("solar_kw", 150)))),
        "wind_kw": int(round(float(grid_preset.get("wind_kw", 300)))),
        "battery_kwh": int(round(float(grid_preset.get("battery_kwh", 200)))),
    }

    if "opt_solar" not in st.session_state:
        st.session_state["opt_solar"] = cost_preset["solar_kw"]
    if "opt_wind" not in st.session_state:
        st.session_state["opt_wind"] = cost_preset["wind_kw"]
    if "opt_batt" not in st.session_state:
        st.session_state["opt_batt"] = cost_preset["battery_kwh"]
    for slider_key, state_key in (
        ("opt_solar_slider", "opt_solar"),
        ("opt_wind_slider", "opt_wind"),
        ("opt_batt_slider", "opt_batt"),
    ):
        if slider_key not in st.session_state:
            st.session_state[slider_key] = st.session_state[state_key]

    with p_col1:
        if st.button("Apply cost-focused preset", type="primary", use_container_width=True):
            st.session_state["opt_solar"] = cost_preset["solar_kw"]
            st.session_state["opt_wind"] = cost_preset["wind_kw"]
            st.session_state["opt_batt"] = cost_preset["battery_kwh"]
            st.session_state["opt_solar_slider"] = cost_preset["solar_kw"]
            st.session_state["opt_wind_slider"] = cost_preset["wind_kw"]
            st.session_state["opt_batt_slider"] = cost_preset["battery_kwh"]
            st.rerun()

    with p_col2:
        if st.button("Apply grid-dependence-focused preset", use_container_width=True):
            st.session_state["opt_solar"] = grid_preset["solar_kw"]
            st.session_state["opt_wind"] = grid_preset["wind_kw"]
            st.session_state["opt_batt"] = grid_preset["battery_kwh"]
            st.session_state["opt_solar_slider"] = grid_preset["solar_kw"]
            st.session_state["opt_wind_slider"] = grid_preset["wind_kw"]
            st.session_state["opt_batt_slider"] = grid_preset["battery_kwh"]
            st.rerun()

    with p_col3:
        if st.button("🔄 Reset to 100% Grid Baseline", use_container_width=True):
            st.session_state["opt_solar"] = 0
            st.session_state["opt_wind"] = 0
            st.session_state["opt_batt"] = 0
            st.session_state["opt_solar_slider"] = 0
            st.session_state["opt_wind_slider"] = 0
            st.session_state["opt_batt_slider"] = 0
            st.rerun()

    st.markdown('<div class="section-heading"><span>🎛️</span><span>Interactive Capacity Controls</span></div>', unsafe_allow_html=True)
    
    # 3 Capacity Control Cards with Full Metadata
    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        render_control_context_card(
            param_name="Solar PV Capacity (kW)",
            description="Peak direct-current (DC) rating of rooftop solar array. Supplies power during daylight hours and charges the battery.",
            unit="Kilowatts (kW)",
            min_val=0,
            max_val=300,
            current_val=f"{st.session_state['opt_solar_slider']} kW",
            impact="Reduces daytime grid imports; surplus generation charges the battery or is curtailed when battery is full.",
            icon="☀️",
        )
        solar_input = st.slider("Solar PV Capacity (kW):", min_value=0, max_value=300, step=25, key="opt_solar_slider")
        st.session_state["opt_solar"] = solar_input

    with c_s2:
        render_control_context_card(
            param_name="Wind Turbine Capacity (kW)",
            description="Nameplate turbine capacity scaled via 2.0 MW power curve. Generates power during both day and night.",
            unit="Kilowatts (kW)",
            min_val=0,
            max_val=600,
            current_val=f"{st.session_state['opt_wind_slider']} kW",
            impact="Provides nighttime clean energy; high capacity increases independence but raises capital and curtailment.",
            icon="🌀",
        )
        wind_input = st.slider("Wind Turbine Capacity (kW):", min_value=0, max_value=600, step=50, key="opt_wind_slider")
        st.session_state["opt_wind"] = wind_input

    with c_s3:
        render_control_context_card(
            param_name="Battery Storage Capacity (kWh)",
            description="Usable BESS energy capacity operating with 84.64% round-trip efficiency, 10%–95% SOC limits, and 0.5C power limit.",
            unit="Kilowatt-hours (kWh)",
            min_val=0,
            max_val=500,
            current_val=f"{st.session_state['opt_batt_slider']} kWh",
            impact="Buffers surplus daytime renewable energy for discharge during peak evening demand hours.",
            icon="🔋",
        )
        batt_input = st.slider("Battery Storage Capacity (kWh):", min_value=0, max_value=500, step=50, key="opt_batt_slider")
        st.session_state["opt_batt"] = batt_input

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        grid_limit_input = st.slider(
            "Maximum grid supply (kW)", min_value=0, max_value=200, value=100, step=10,
            help="Limits how much electricity can be imported in each hourly interval.",
        )
        duration_input = st.slider(
            "Simulation duration (hours)", min_value=24, max_value=240, value=240, step=24,
            help="Uses the first selected number of hourly points from the synthetic benchmark.",
        )
    with control_col2:
        cyclic_soc = st.checkbox(
            "Require the battery to finish at its starting charge",
            value=False,
            help="For a repeating horizon, solve a constrained schedule that ends at the starting 50% charge. It prioritizes serving demand, then reducing grid imports, curtailment, and battery throughput. Charging from the grid is not allowed.",
        )
        st.caption(
            "With this option off, the finite-horizon dispatch may finish at a different charge. "
            "With it on, the scheduler enforces the terminal target or reports infeasibility."
        )

    # Simulation execution
    try:
        dispatch, metrics, costs = simulate_custom_system(
            solar_kw=float(solar_input),
            wind_kw=float(wind_input),
            battery_kwh=float(batt_input),
            grid_import_limit_kw=float(grid_limit_input),
            duration_hours=int(duration_input),
            terminal_soc_ratio=0.50 if cyclic_soc else None,
        )
        _, baseline_metrics, baseline_costs = simulate_custom_system(
            solar_kw=0.0, wind_kw=0.0, battery_kwh=0.0,
            grid_import_limit_kw=float(grid_limit_input), duration_hours=int(duration_input),
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        st.error("This configuration could not be simulated with the current data and battery rules.")
        st.caption(str(exc))
        st.stop()

    # Results Metric Cards
    st.markdown('<div class="section-heading"><span>📊</span><span>Simulation Results & Performance</span></div>', unsafe_allow_html=True)
    explanation = explain_simulation(
        metrics, costs, int(duration_input), baseline_metrics=baseline_metrics, baseline_costs=baseline_costs
    )
    st.info(explanation["supply"])
    st.markdown(f"**Compared with grid-only supply over the same hours:** {explanation['comparison']}")
    st.markdown("**Why it matters:** These totals show the simulated contributions and any supply gaps over the chosen horizon.")
    st.caption(explanation["reliability"])
    st.caption(f"**What to consider next:** {explanation['next']}")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Renewable generation available", f"{metrics['available_renewable_kwh']:,.1f} kWh")
    r2.metric("Renewable energy used", f"{metrics['renewable_used_kwh']:,.1f} kWh", f"{metrics['renewable_penetration']:.1%} of served supply")
    r3.metric("Grid electricity imported", f"{metrics['grid_import_kwh']:,.1f} kWh", f"{metrics['grid_import_kwh'] - baseline_metrics['grid_import_kwh']:+,.1f} kWh vs grid-only")
    r4.metric("Possible unmet demand", f"{metrics['unmet_demand_kwh']:,.2f} kWh")
    r5, r6, r7, r8 = st.columns(4)
    r5.metric("Excess renewable energy", f"{metrics['curtailment_kwh']:,.1f} kWh")
    r6.metric("Battery delivered", f"{metrics['battery_discharge_kwh']:,.1f} kWh")
    r7.metric("Battery ending charge", f"{metrics['final_soc_kwh']:,.1f} kWh", f"{metrics['terminal_soc_delta_kwh']:+,.1f} kWh from start")
    r8.metric("Estimated cost", "$" + f"{costs['total_cost']:,.2f}", "$" + f"{costs['total_cost'] - baseline_costs['total_cost']:+,.2f} vs grid-only")
    st.caption(
        f"Battery delivered {metrics['battery_discharge_kwh']:,.2f} kWh in total: "
        f"{metrics['renewable_origin_battery_discharge_kwh']:,.2f} kWh came from renewable energy "
        f"stored during this run and {metrics['initial_soc_carry_in_discharge_kwh']:,.2f} kWh came from "
        "the battery's starting inventory. Renewable share is (direct renewable supply + renewable-origin "
        "battery output) divided by electricity served; unmet demand is reported separately."
    )

    st.subheader("Hourly energy balance")
    t_slice = min(96, len(dispatch.demand))
    fig_dispatch = create_dispatch_dashboard_chart(
        timestamps=[f"Hour {h + 1}" for h in range(t_slice)],
        demand=dispatch.demand[:t_slice],
        renewable_used=dispatch.renewable_used[:t_slice],
        battery_discharge=dispatch.battery_discharge[:t_slice],
        grid_import=dispatch.grid_import[:t_slice],
        unmet_demand=dispatch.unmet_demand[:t_slice],
        curtailment=dispatch.curtailment[:t_slice],
        battery_soc=dispatch.battery_soc[:t_slice],
        title=f"Energy supplied and demand over the first {t_slice} hours",
    )
    st.plotly_chart(fig_dispatch, use_container_width=True)

    # Pareto Tradeoff Frontier
    st.markdown('<div class="section-heading"><span>🌌</span><span>120-Candidate Mix Pareto Frontier (Cost vs Autonomy)</span></div>', unsafe_allow_html=True)
    cand_df = load_feasibility_scenarios_table()
    if not cand_df.empty:
        fig_pareto = create_tradeoff_pareto_chart(cand_df)
        st.plotly_chart(fig_pareto, use_container_width=True)

    # Conformal Uncertainty Scenarios
    st.markdown('<div class="section-heading"><span>🛡️</span><span>Conformal Stress-Testing Scenarios</span></div>', unsafe_allow_html=True)
    decision_summary = load_decision_support_summary()
    stress = decision_summary.get("uncertainty_stress_evaluation", {})
    scenarios = stress.get("scenarios", [])
    if scenarios:
        fig_scenarios = create_scenario_comparison_bar_chart(scenarios)
        st.plotly_chart(fig_scenarios, use_container_width=True)
    else:
        st.warning("Generated uncertainty stress scenarios are unavailable; rerun scripts/run_decision_support.py.")

    render_disclaimer(
        "<b>Decision-Support PoC Caveat</b>: The integrated scenario pairs independent profile sequences from Stanford, Mumbai, Gujarat, and Orlando by array position; their timestamps are not synchronized "
        "for the sole purpose of stress-testing the multi-modal uncertainty framework. It does not represent an empirical physical installation.",
        warning=True,
    )


# =============================================================================
# PAGE 8: METHODOLOGY & SCIENTIFIC DISCLOSURES
# =============================================================================
elif st.session_state["nav_page"] == "8. Methodology and limitations":
    render_header(
        "Methodology and limitations",
        "Where the records come from, what each model estimates, and which assumptions limit interpretation.",
    )
    st.markdown(
        "**Project title:** A Multimodal, Uncertainty-Aware Decision-Support Framework for Location-Specific Renewable Energy Feasibility and Energy-Mix Optimization."
    )

    st.markdown('<div class="section-heading"><span>📍</span><span>Heterogeneous Data Provenance Matrix</span></div>', unsafe_allow_html=True)
    
    # Render styled high-contrast table instead of plain st.dataframe to eliminate blank/unstyled rendering
    prov_html = (
        '<table class="styled-table">'
        '<thead>'
        '<tr>'
        '<th>Model ID</th>'
        '<th>Architecture</th>'
        '<th>Source Dataset</th>'
        '<th>Geographic Location</th>'
        '<th>Temporal Period</th>'
        '<th>Input Representation</th>'
        '<th>Target Variable & Unit</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
        '<tr>'
        '<td><b>Model A</b></td>'
        '<td>ResNet-18 (Transfer Learning)</td>'
        '<td>Stanford SKIPP\'D</td>'
        '<td>Stanford, CA, USA (37.43°N, 122.17°W)</td>'
        '<td>2017–2019 (14,003 test images)</td>'
        '<td>64×64 RGB Sky Camera Image</td>'
        '<td>PV Power: <code>kW</code> (Kilowatts)</td>'
        '</tr>'
        '<tr>'
        '<td><b>Model B</b></td>'
        '<td>WeatherLSTM (h=32, L=1)</td>'
        '<td>NASA POWER Hourly</td>'
        '<td>Mumbai, India (19.08°N, 72.88°E)</td>'
        '<td>2023 (8,760 hourly records)</td>'
        '<td>24h Sequence (GHI, T, RH, Wind)</td>'
        '<td>Solar Irradiance: <code>Wh/m²</code></td>'
        '</tr>'
        '<tr>'
        '<td><b>Model C</b></td>'
        '<td>WindLSTM (h=64, L=1)</td>'
        '<td>NREL India Wind Toolkit</td>'
        '<td>Gujarat Coastal (23.24°N, 69.66°E)</td>'
        '<td>2014 (8,760 hourly records)</td>'
        '<td>24h Sequence (Speed, Direction, T, P)</td>'
        '<td>100m Speed: <code>m/s</code> & Power Proxy: <code>kW</code></td>'
        '</tr>'
        '<tr>'
        '<td><b>Model D</b></td>'
        '<td>DemandLSTM (h=64, L=1)</td>'
        '<td>Building Data Genome 2</td>'
        '<td>Orlando, FL, USA (28.54°N, 81.38°W)</td>'
        '<td>2016–2017 (17,544 continuous hours)</td>'
        '<td>24h Sequence of Active Load</td>'
        '<td>Building Load: <code>kWh</code></td>'
        '</tr>'
        '</tbody>'
        '</table>'
    )
    st.markdown(prov_html, unsafe_allow_html=True)

    st.markdown('<div class="section-heading"><span>📐</span><span>Physical Energy Balance & Dispatch Formulation</span></div>', unsafe_allow_html=True)
    st.latex(r"R_{\text{used}, t} + G_{\text{import}, t} + B_{\text{dis}, t} + U_t = D_t")
    st.latex(r"R_{\text{avail}, t} = R_{\text{used}, t} + B_{\text{ch}, t} + C_t")
    st.latex(r"\text{Renewable share} = \frac{\sum_t R_{\text{used},t} + \sum_t B_{\text{renewable discharge},t}}{\sum_t D_{\text{served},t}}")
    
    st.markdown(
        """
        <div class="chart-card">
            <div style="font-weight: 700; color: var(--text-primary); font-size: 1.05rem; margin-bottom: 0.5rem;">🔋 Battery Energy Storage System (BESS) Dynamics</div>
            <ul style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6; margin-bottom: 0;">
                <li><b>Efficiency Parameters</b>: Charge efficiency $\\eta_{\\text{ch}} = 0.92$, discharge efficiency $\\eta_{\\text{dis}} = 0.92$, yielding an empirical round-trip efficiency $\\eta_{\\text{rt}} \\approx 84.64\\%$.</li>
                <li><b>Operational Constraints</b>: Strict State-of-Charge limits $0.10 \\le \\text{SOC}_t \\le 0.95$ to prevent degradation; maximum C-rate limited to $0.5\\text{C}$ per hour.</li>
                <li><b>Stored Energy</b>: $\\text{SOC}_t = \\text{SOC}_{t-1} + \\eta_{\\text{ch}} E_{\\text{ch},t} - E_{\\text{dis},t} / \\eta_{\\text{dis}}$; initial inventory is carry-in, separate from energy charged during the simulation.</li>
                <li><b>Cyclic Schedule</b>: When enabled, a mixed-integer schedule enforces terminal SOC and prevents simultaneous charging/discharging. It minimizes unmet demand, grid imports, renewable curtailment, then battery throughput. Grid charging is not allowed.</li>
                <li><b>Conservation Assertion</b>: Exact machine-precision energy conservation $|\\text{residual}| \\le 10^{-6}\\text{ kWh}$ verified at every discrete simulation step across all 120 candidate mixes.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading"><span>🛡️</span><span>Split Conformal Prediction Formulation</span></div>', unsafe_allow_html=True)
    st.markdown(
        r"""
        <div class="chart-card">
            <div style="font-weight: 700; color: var(--text-primary); font-size: 1.05rem; margin-bottom: 0.5rem;">Finite-Sample Distribution-Free Uncertainty Intervals</div>
            <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6;">
                Non-conformity absolute residuals are computed on a dedicated, strictly chronological calibration split $D_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^n$:
            </p>
            <div style="background:var(--canvas); border:1px solid var(--border); border-radius:6px; padding:0.5rem 1rem; margin:0.5rem 0; font-family:monospace; font-size:0.9rem; color:var(--text-primary);">
                s_i = |y_i - f(x_i)|, &nbsp;&nbsp; \hat{q} = \text{Quantile}\left(s_{1:n}, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)
            </div>
            <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6; margin-bottom: 0;">
                The resulting prediction band $C(x_{n+1}) = [f(x_{n+1}) - \hat{q}, f(x_{n+1}) + \hat{q}]$ guarantees valid marginal coverage under exchangeability:
                $P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading"><span>⚠️</span><span>Mandatory Scientific Disclosures & Limitations</span></div>', unsafe_allow_html=True)
    
    render_disclaimer(
        "<b>Geographic & Temporal Disconnection</b>: The four datasets (Stanford SKIPP'D, NASA POWER Mumbai, NREL Gujarat, and BDG2 Orlando) "
        "originate from completely independent geographic locations, weather regimes, and observation years. "
        "They do NOT describe a single physical microgrid or co-located generation facility.",
        warning=True,
    )

    render_disclaimer(
        "<b>Illustrative Decision-Support Proof of Concept</b>: The Stage 5 energy-mix optimization combines these normalized series "
        "exclusively to demonstrate the mathematical framework for multimodal forecasting, conformal risk bounds, and battery storage dispatch. "
        "It should not be used as an investment-grade feasibility study for any specific physical site.",
        warning=True,
    )

    render_disclaimer(
        "<b>Marginal vs. Simultaneous Coverage</b>: Conformal guarantees hold marginally for each individual time step $t$. "
        "They do not provide simultaneous joint interval coverage across all 240 consecutive hours of the multi-day dispatch horizon.",
        warning=True,
    )

    page8_uncertainty = load_uncertainty_summary().get("demand", {})
    page8_coverage = page8_uncertainty.get("eval_90", {}).get("empirical_coverage")
    if page8_coverage is not None:
        render_disclaimer(
            f"<b>Demand distribution shift and undercoverage</b>: Model D observed {float(page8_coverage):.2%} "
            "empirical coverage on the held-out test set against a 90% target. Coverage was also below the 80% target "
            "in every observed test month; residuals show modest hourly and daily dependence. These checks are "
            "consistent with time dependence or distribution change, but do not isolate the cause.",
            warning=True,
        )

    with st.expander("Additional data and model limitations"):
        st.markdown(
            "- SKIPP'D includes one of five training shards in this copy; its held-out test shard is present. "
            "The image model therefore does not represent the complete training benchmark.\n"
            "- The selected wind model uses wind speed alone. Experimental multivariate wind models use direction in degrees as a linear feature; "
            "a circular encoding comparison has not been performed.\n"
            "- Matched held-out persistence and daily-seasonal baselines were checked for the selected NASA, wind, and demand models; "
            "the demand model has slightly worse MAE than persistence, and a weekly baseline needs a 168-hour history not supplied by its 24-hour input.\n"
            "- The 90% demand interval covered less than its nominal target in the recorded held-out period; no test-set tuning was performed."
        )
