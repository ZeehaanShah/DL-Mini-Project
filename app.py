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


# -----------------------------------------------------------------------------
# APP CONFIGURATION & THEMING
# -----------------------------------------------------------------------------
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
    "1. Project Overview",
    "2. Solar PV Nowcasting (Model A)",
    "3. Solar Irradiance Forecasting (Model B)",
    "4. Wind Forecasting (Model C)",
    "5. Electricity Demand Forecasting (Model D)",
    "6. Uncertainty Analysis (Stage 4)",
    "7. Energy-Mix Optimizer (Model E)",
    "8. Methodology & Disclosures",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGE_OPTIONS[0]

def set_page(page_name: str) -> None:
    st.session_state["nav_page"] = page_name


# Sidebar brand header
st.sidebar.markdown(
    '<div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 1.25rem; padding-bottom: 0.85rem; border-bottom: 1px solid #CBD5E1;">'
    '<div style="width: 36px; height: 36px; background: linear-gradient(135deg, #047857 0%, #065F46 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.2rem;">⚡</div>'
    '<div>'
    '<div style="font-size: 1.05rem; font-weight: 700; color: #172033; line-height: 1.2;">Energy Intelligence</div>'
    '<div style="font-size: 0.75rem; color: #475569; font-weight: 500;">Decision-Support System</div>'
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
            'Models Ready'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c_act:
        if st.button("▶ Run Analysis", key="btn_top_run_analysis", type="primary", use_container_width=True):
            set_page("7. Energy-Mix Optimizer (Model E)")
            st.rerun()

st.markdown("<div style='border-bottom: 1px solid #CBD5E1; margin-bottom: 1.25rem; margin-top: 0.5rem;'></div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: PROJECT OVERVIEW (DASHBOARD)
# =============================================================================
if st.session_state["nav_page"] == "1. Project Overview":
    st.markdown('<div class="page-hero-title">Your energy system at a glance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-hero-subtitle">Explore deep-learning forecasts, conformal prediction intervals, and energy-mix optimization.</div>', unsafe_allow_html=True)

    # 4 Model Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_model_card(
            model_id="Model A",
            title="Solar PV Output (Model A)",
            metric_value="2.301 kW",
            metric_label="Test MAE (ResNet-18)",
            color_theme="green",
            icon="☀️",
            badge="↗ 40% vs Base",
        )
        if st.button("View Solar Model", key="btn_card_m1", use_container_width=True):
            set_page("2. Solar PV Nowcasting (Model A)")
            st.rerun()

    with c2:
        render_model_card(
            model_id="Model B",
            title="Solar Irradiance (Model B)",
            metric_value="16.26 Wh/m²",
            metric_label="Test MAE (WeatherLSTM)",
            color_theme="blue",
            icon="▦",
            badge="📊 R² = 0.989",
        )
        if st.button("View Irradiance Model", key="btn_card_m2", use_container_width=True):
            set_page("3. Solar Irradiance Forecasting (Model B)")
            st.rerun()

    with c3:
        render_model_card(
            model_id="Model C",
            title="Wind Speed (Model C)",
            metric_value="0.591 m/s",
            metric_label="Test MAE (WindLSTM)",
            color_theme="teal",
            icon="🌀",
            badge="🍃 93% Cov",
        )
        if st.button("View Wind Model", key="btn_card_m3", use_container_width=True):
            set_page("4. Wind Forecasting (Model C)")
            st.rerun()

    with c4:
        render_model_card(
            model_id="Model D",
            title="Electricity Demand (Model D)",
            metric_value="2.615 kWh",
            metric_label="Test MAE (DemandLSTM)",
            color_theme="purple",
            icon="🏢",
            badge="📈 R² = 0.923",
        )
        if st.button("View Demand Model", key="btn_card_m4", use_container_width=True):
            set_page("5. Electricity Demand Forecasting (Model D)")
            st.rerun()

    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

    # Middle Row: Forecast vs Actual (65%) & Donut Chart (35%)
    col_chart_left, col_chart_right = st.columns([65, 35])

    with col_chart_left:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-header">'
            '<span>📊</span>'
            '<span>Forecast vs Actual – Solar PV Output (Model A)</span>'
            '</div>'
            '<div class="chart-card-sub">Stanford SKIPP\'D Test Set Nowcasting (EXP_SOLAR_DL_S3)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Load and render real Model A test series
        solar_eval = get_solar_nowcasting_test_evaluation(max_samples=260)
        if solar_eval and "timestamps" in solar_eval:
            fig_solar = create_overview_solar_pv_chart(
                timestamps=solar_eval["timestamps"],
                y_true=solar_eval["y_true"],
                y_pred=solar_eval["y_pred"],
                title="Forecast vs Actual – Solar PV Output (Model A)",
                subtitle="Stanford SKIPP'D Test Morning Sequence (EXP_SOLAR_DL_S3)",
            )
            st.plotly_chart(fig_solar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Loading Solar PV nowcasting sequence...")

    with col_chart_right:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-header">'
            '<span>🍩</span>'
            '<span>Energy Mix Snapshot (Optimized Scenario)</span>'
            '</div>'
            '<div class="chart-card-sub">Optimal generation mix (illustrative scenario)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        fig_donut = create_energy_mix_donut_chart(
            solar_val=152.0,
            wind_val=112.0,
            battery_val=63.0,
            grid_val=20.0,
            total_label="347 kW",
            title="System Snapshot – Illustrative Energy Mix",
            subtitle="Optimal generation mix (illustrative scenario)",
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # Quick Actions Row
    st.markdown(
        '<div style="margin-top: 1.25rem; margin-bottom: 0.75rem;">'
        '<div style="font-size: 1.25rem; font-weight: 700; color: #172033; display: flex; align-items: center; gap: 0.5rem;">'
        '<span>⚡</span>'
        '<span>Quick Actions</span>'
        '</div>'
        '<div style="font-size: 0.85rem; color: #475569;">Try the core analytical capabilities of the Decision-Support System.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    qa_col1, qa_col2, qa_col3 = st.columns(3)
    with qa_col1:
        render_quick_action_card(
            title="Try Solar Image Prediction",
            subtitle="Upload a sky camera image for instantaneous PV power nowcasting",
            theme="teal",
            icon="📷",
        )
        if st.button("📷 Upload Sky Image & Predict", key="btn_qa_solar", type="primary", use_container_width=True):
            set_page("2. Solar PV Nowcasting (Model A)")
            st.rerun()

    with qa_col2:
        render_quick_action_card(
            title="Explore Uncertainty Analysis",
            subtitle="View 90% conformal prediction intervals and empirical calibration",
            theme="blue",
            icon="📊",
        )
        if st.button("📊 View Prediction Intervals", key="btn_qa_unc", type="primary", use_container_width=True):
            set_page("6. Uncertainty Analysis (Stage 4)")
            st.rerun()

    with qa_col3:
        render_quick_action_card(
            title="Optimize Energy Mix",
            subtitle="Simulate solar, wind, and battery dispatch under grid constraints",
            theme="purple",
            icon="⚙️",
        )
        if st.button("⚙️ Simulate Energy Mix", key="btn_qa_opt", type="primary", use_container_width=True):
            set_page("7. Energy-Mix Optimizer (Model E)")
            st.rerun()

    # Mini Forecast Previews Row
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-header">'
            '<span>🍃</span>'
            '<span>Wind Speed Forecast with Uncertainty (Model C)</span>'
            '</div>'
            '<div class="chart-card-sub">NREL Gujarat Test Split (90% Conformal Prediction Interval)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        wind_eval = get_wind_test_evaluation(max_samples=72)
        if wind_eval and "timestamps" in wind_eval:
            fig_mini_wind = create_mini_forecast_chart(
                timestamps=wind_eval["timestamps"][:48],
                y_true=wind_eval["y_true"][:48],
                y_pred=wind_eval["y_pred"][:48],
                y_low=wind_eval["y_low_90"][:48],
                y_upp=wind_eval["y_upp_90"][:48],
                title="Wind Speed Forecast with Uncertainty (Model C)",
                unit="m/s",
                theme="green",
            )
            st.plotly_chart(fig_mini_wind, use_container_width=True, config={"displayModeBar": False})

    with m_col2:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-header">'
            '<span>🏢</span>'
            '<span>Electricity Demand Forecast (Model D)</span>'
            '</div>'
            '<div class="chart-card-sub">BDG2 Panther_education_Rosalie Test Split (90% Conformal Interval)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        demand_eval = get_demand_test_evaluation(max_samples=72)
        if demand_eval and "timestamps" in demand_eval:
            fig_mini_demand = create_mini_forecast_chart(
                timestamps=demand_eval["timestamps"][:48],
                y_true=demand_eval["y_true"][:48],
                y_pred=demand_eval["y_pred"][:48],
                y_low=demand_eval["y_low_90"][:48],
                y_upp=demand_eval["y_upp_90"][:48],
                title="Electricity Demand Forecast (Model D)",
                unit="kWh",
                theme="purple",
            )
            st.plotly_chart(fig_mini_demand, use_container_width=True, config={"displayModeBar": False})

    # Prominent Amber Disclaimer
    render_disclaimer(
        "Research prototype disclosure: Datasets are collected from geographically independent locations and disjoint calendar periods. "
        "The combined multi-source energy mix represents an illustrative decision-support proof of concept, not an empirical single-site microgrid.",
        warning=True,
    )

    # Expandable Complete Experiment Registry
    with st.expander("📑 View Complete Experiment Registry (52 Verified Experiments)"):
        reg_df = load_registry_data()
        if not reg_df.empty:
            st.dataframe(
                reg_df[["experiment_id", "dataset", "model_name", "partition", "mae", "rmse", "r2", "metric_unit"]],
                use_container_width=True,
                height=300,
            )


# =============================================================================
# PAGE 2: SOLAR PV NOWCASTING (MODEL A)
# =============================================================================
elif st.session_state["nav_page"] == "2. Solar PV Nowcasting (Model A)":
    render_header(
        "Model A: Solar Sky-Image PV Nowcasting",
        "Instantaneous rooftop PV power generation estimation directly from hemispherical ground-based sky camera imagery.",
    )

    # Model Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_model_card("Model A", "Test-Set MAE", "2.301 kW", "ResNet-18 (Fine-Tuned)", color_theme="green", icon="☀️", badge="TEST SET")
    with m2:
        render_model_card("Model A", "Test-Set RMSE", "3.766 kW", "ResNet-18 (Fine-Tuned)", color_theme="blue", icon="📐", badge="TEST SET")
    with m3:
        render_model_card("Model A", "Persistence MAE", "3.842 kW", "Naive 1-min persistence", color_theme="teal", icon="⏱", badge="BASELINE")
    with m4:
        render_model_card("Model A", "Error Reduction", "-40.1%", "Superiority over persistence", color_theme="purple", icon="🏆", badge="VERIFIED")

    render_provenance_badge(
        "Stanford SKIPP'D Ground Fisheye Camera",
        "Vision Benchmark",
        "Stanford, CA, USA (37.43°N, 122.17°W)",
        "2017–2019 (14,003 test images)",
        "kW (Kilowatts)",
    )

    st.markdown('<div class="section-heading"><span>📷</span><span>Interactive Sky Image Inference</span></div>', unsafe_allow_html=True)
    st.markdown("Test the frozen ResNet-18 nowcaster using either verified test sky images or by uploading an image.")

    tab_sample, tab_upload = st.tabs(["🖼️ Built-in Test Sky Images (Ground Truth)", "⬆️ Upload Custom Sky Image"])

    sample_dir = PROJECT_ROOT / "data" / "sample_images"
    sample_files = sorted(list(sample_dir.glob("*.png"))) if sample_dir.exists() else []

    with tab_sample:
        if sample_files:
            sample_options = {
                "Sample 1: High Noon Clear Sky (Ground Truth: 24.26 kW)": sample_files[1] if len(sample_files) > 1 else sample_files[0],
                "Sample 2: Sunny Skies (Ground Truth: 22.88 kW)": sample_files[3] if len(sample_files) > 3 else sample_files[0],
                "Sample 3: Transition / Cloud Cover (Ground Truth: 10.26 kW)": sample_files[2] if len(sample_files) > 2 else sample_files[0],
                "Sample 4: Night / Dawn (Ground Truth: 0.00 kW)": sample_files[0],
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
                            '<div class="chart-card" style="border-top: 4px solid #047857;">'
                            '<div style="font-size: 0.85rem; color: #475569; font-weight: 600;">PREDICTED ROOFTOP PV GENERATION</div>'
                            f'<div style="font-size: 2.2rem; font-weight: 800; color: #172033; margin: 0.35rem 0;">{pred_kw:.2f} kW</div>'
                            '<div style="font-size: 0.82rem; color: #047857; font-weight: 700;">Model: ResNet-18 (EXP_SOLAR_DL_S3)</div>'
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
                                '<div class="chart-card" style="border-top: 4px solid #0369A1;">'
                                '<div style="font-size: 0.85rem; color: #475569; font-weight: 600;">PREDICTED PV POWER</div>'
                                f'<div style="font-size: 2.2rem; font-weight: 800; color: #172033; margin: 0.35rem 0;">{pred_kw:.2f} kW</div>'
                                '<div style="font-size: 0.82rem; color: #475569;">Resized to 64×64 and ImageNet normalized.</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            st.error(f"Inference error: {e}")

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
elif st.session_state["nav_page"] == "3. Solar Irradiance Forecasting (Model B)":
    render_header(
        "Model B: Hourly Solar Irradiance Forecasting",
        "Hourly surface downward shortwave radiation (GHI) sequence forecasting using LSTM neural networks.",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_model_card("Model B", "Test-Set MAE", "16.26 Wh/m²", "WeatherLSTM (h=32, L=1)", color_theme="blue", icon="▦", badge="TEST SET")
    with m2:
        render_model_card("Model B", "Test-Set RMSE", "30.68 Wh/m²", "WeatherLSTM", color_theme="green", icon="📐", badge="TEST SET")
    with m3:
        render_model_card("Model B", "Coefficient of Det. (R²)", "0.989", "Near-perfect tracking", color_theme="teal", icon="⭐", badge="R² SCORE")
    with m4:
        render_model_card("Model B", "Baseline Naive MAE", "112.43 Wh/m²", "24h seasonal naive", color_theme="purple", icon="⏱", badge="BASELINE")

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
            '<div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:0.75rem; margin-top:0.35rem;">'
            '<div style="font-size:0.82rem; color:#475569; font-weight:600;">FORECAST HORIZON</div>'
            '<div style="font-size:1.15rem; font-weight:800; color:#172033;">1-Hour Ahead (t+1)</div>'
            '<div style="font-size:0.78rem; color:#047857;">Fixed recurrent single-step inference</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c_slider2:
        n_samples = st.slider("Displayed Time Window (Hours):", min_value=24, max_value=336, value=96, step=24)
        st.markdown(f"<div style='font-size:0.82rem; color:#475569; font-weight:600;'>Active window: <b>{n_samples} hours</b> ({n_samples/24:.1f} days of continuous test series)</div>", unsafe_allow_html=True)

    eval_b = get_weather_test_evaluation(max_samples=n_samples)
    if eval_b and "timestamps" in eval_b:
        fig_b = create_actual_vs_predicted_chart(
            timestamps=eval_b["timestamps"],
            y_true=eval_b["y_true"],
            y_pred=eval_b["y_pred"],
            title=f"Hourly Solar Irradiance: Actual vs Predicted (Showing {n_samples} Hours)",
            unit="Wh/m²",
        )
        st.plotly_chart(fig_b, use_container_width=True)

        residuals = eval_b["y_true"] - eval_b["y_pred"]
        st.markdown(
            f'<div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:0.65rem 1rem; font-size:0.88rem; color:#172033;">'
            f'<b>Residual Evaluation</b>: Mean Residual: <code>{np.mean(residuals):+.2f} Wh/m²</code> &nbsp;|&nbsp; '
            f'Max Absolute Error: <code>{np.max(np.abs(residuals)):.2f} Wh/m²</code> &nbsp;|&nbsp; '
            f'Standard Deviation of Error: <code>{np.std(residuals):.2f} Wh/m²</code>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 4: WIND FORECASTING (MODEL C)
# =============================================================================
elif st.session_state["nav_page"] == "4. Wind Forecasting (Model C)":
    render_header(
        "Model C: Hub-Height Wind Speed & Turbine Power Proxy",
        "100m hub-height wind speed forecasting with 90% split-conformal uncertainty intervals and 2.0 MW turbine power proxy.",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_model_card("Model C", "Test Wind MAE", "0.591 m/s", "WindLSTM (h=64, L=1)", color_theme="teal", icon="🌀", badge="TEST SET")
    with m2:
        render_model_card("Model C", "Test Wind RMSE", "0.822 m/s", "WindLSTM", color_theme="blue", icon="📐", badge="TEST SET")
    with m3:
        render_model_card("Model C", "90% Empirical Coverage", "92.83%", "Nominal: 90.0% (Valid)", color_theme="green", icon="🛡️", badge="CONFORMAL")
    with m4:
        render_model_card("Model C", "Turbine Power MAE", "47.89 kW", "Illustrative 2 MW proxy", color_theme="purple", icon="⚡", badge="PROXY")

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
    st.markdown(f"<div style='font-size:0.82rem; color:#475569; font-weight:600;'>Active window: <b>{n_wind_samples} hours</b> ({n_wind_samples/24:.1f} days of continuous wind records)</div>", unsafe_allow_html=True)

    eval_c = get_wind_test_evaluation(max_samples=n_wind_samples)
    if eval_c and "timestamps" in eval_c:
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
                title="Illustrative 2 MW Turbine Power Proxy: Estimated Power vs Persistence",
                unit="kW",
            )
        st.plotly_chart(fig_c, use_container_width=True)

    render_disclaimer(
        "The 2.0 MW turbine power curve (cut-in 3.0 m/s, rated 12.0 m/s, cut-out 25.0 m/s) is an <b>illustrative engineering proxy</b>, "
        "not empirical SCADA output from an operational wind farm.",
        warning=True,
    )


# =============================================================================
# PAGE 5: ELECTRICITY DEMAND FORECASTING (MODEL D)
# =============================================================================
elif st.session_state["nav_page"] == "5. Electricity Demand Forecasting (Model D)":
    render_header(
        "Model D: Building Electricity Demand Forecasting",
        "Commercial education building hourly electricity consumption forecasting with 90% conformal prediction intervals.",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_model_card("Model D", "Test Demand MAE", "2.615 kWh", "DemandLSTM (h=64, L=1)", color_theme="purple", icon="🏢", badge="TEST SET")
    with m2:
        render_model_card("Model D", "Test Demand RMSE", "4.067 kWh", "DemandLSTM", color_theme="blue", icon="📐", badge="TEST SET")
    with m3:
        render_model_card("Model D", "Coefficient of Det. (R²)", "0.923", "Strong load tracking", color_theme="teal", icon="⭐", badge="R² SCORE")
    with m4:
        render_model_card("Model D", "90% Empirical Coverage", "85.47%", "Nominal: 90.0% (Shifted)", color_theme="green", icon="🛡️", badge="CONFORMAL")

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
    st.markdown(f"<div style='font-size:0.82rem; color:#475569; font-weight:600;'>Active window: <b>{n_dem_samples} hours</b> ({n_dem_samples/24:.1f} days of continuous building meter data)</div>", unsafe_allow_html=True)

    eval_d = get_demand_test_evaluation(max_samples=n_dem_samples)
    if eval_d and "timestamps" in eval_d:
        fig_d = create_actual_vs_predicted_chart(
            timestamps=eval_d["timestamps"],
            y_true=eval_d["y_true"],
            y_pred=eval_d["y_pred"],
            y_low=eval_d["y_low_90"],
            y_upp=eval_d["y_upp_90"],
            title="Building Electricity Demand: Actual vs Forecast with 90% Conformal Interval",
            unit="kWh",
            nominal_label="90% Conformal Prediction Interval",
        )
        st.plotly_chart(fig_d, use_container_width=True)

    render_disclaimer(
        "<b>Seasonal Distribution Shift Disclosure</b>: Demand empirical coverage (85.47%) slightly undercovers the 90% nominal level. "
        "Calibration took place on steady summer cooling load (May–September 2017), whereas the test period underwent autumn/winter transitions "
        "and holiday shutdowns, creating residual variance expansion.",
        warning=True,
    )


# =============================================================================
# PAGE 6: UNCERTAINTY ANALYSIS (STAGE 4)
# =============================================================================
elif st.session_state["nav_page"] == "6. Uncertainty Analysis (Stage 4)":
    render_header(
        "Stage 4: Split Conformal Prediction & Uncertainty Calibration",
        "Distribution-free, finite-sample prediction intervals for renewable resource and demand volatility.",
    )

    st.markdown('<div class="section-heading"><span>🎓</span><span>What Are Prediction Intervals?</span></div>', unsafe_allow_html=True)
    st.markdown(
        """
        Unlike standard deep-learning models that output only a single deterministic point forecast, conformal prediction 
        constructs rigorous prediction intervals $[\\hat{y} - q_{1-\\alpha}, \\hat{y} + q_{1-\\alpha}]$ with guaranteed finite-sample 
        marginal coverage:
        """
    )
    st.latex(r"P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha")

    # Table of Calibration Results
    unc_summary = load_uncertainty_summary()
    st.markdown('<div class="section-heading"><span>📊</span><span>Empirical Test Coverage & Calibration Metrics</span></div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown(
            '<div class="chart-card" style="border-top: 4px solid #0D9488;">'
            '<div style="font-weight: 700; color: #172033; font-size: 1.1rem; margin-bottom: 0.5rem;">🍃 Model C: Wind Speed Calibration</div>'
            '<div style="font-size: 0.85rem; color: #475569; margin-bottom: 0.85rem;">Calibration Samples: 1,290 | Test Samples: 1,242</div>'
            '<table class="styled-table">'
            '<thead><tr><th>Metric</th><th>80% Nominal Level</th><th>90% Nominal Level</th></tr></thead>'
            '<tbody>'
            '<tr><td>Nominal Coverage Target</td><td>80.0%</td><td>90.0%</td></tr>'
            '<tr><td><b>Empirical Test Coverage</b></td><td><span style="background:#D1FAE5; color:#065F46; padding:2px 8px; border-radius:9999px; font-weight:700;">84.22% (Valid)</span></td><td><span style="background:#D1FAE5; color:#065F46; padding:2px 8px; border-radius:9999px; font-weight:700;">92.83% (Valid)</span></td></tr>'
            '<tr><td>Mean Interval Width (MPIW)</td><td>2.09 m/s</td><td>3.06 m/s</td></tr>'
            '<tr><td>Winkler Interval Score</td><td>3.05</td><td>3.92</td></tr>'
            '</tbody>'
            '</table>'
            '<div style="font-size:0.80rem; color:#065F46; font-weight:600;">✔ Meets and exceeds nominal coverage targets under stationary wind distribution.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with t_col2:
        st.markdown(
            '<div class="chart-card" style="border-top: 4px solid #7C3AED;">'
            '<div style="font-weight: 700; color: #172033; font-size: 1.1rem; margin-bottom: 0.5rem;">🏢 Model D: Electricity Demand Calibration</div>'
            '<div style="font-size: 0.85rem; color: #475569; margin-bottom: 0.85rem;">Calibration Samples: 2,607 | Test Samples: 2,561</div>'
            '<table class="styled-table">'
            '<thead><tr><th>Metric</th><th>80% Nominal Level</th><th>90% Nominal Level</th></tr></thead>'
            '<tbody>'
            '<tr><td>Nominal Coverage Target</td><td>80.0%</td><td>90.0%</td></tr>'
            '<tr><td><b>Empirical Test Coverage</b></td><td><span style="background:#FFFBEB; color:#92400E; padding:2px 8px; border-radius:9999px; font-weight:700; border:1px solid #F59E0B;">66.03% (Undercovered)</span></td><td><span style="background:#FFFBEB; color:#92400E; padding:2px 8px; border-radius:9999px; font-weight:700; border:1px solid #F59E0B;">85.47% (Undercovered)</span></td></tr>'
            '<tr><td>Mean Interval Width (MPIW)</td><td>4.96 kWh</td><td>9.53 kWh</td></tr>'
            '<tr><td>Winkler Interval Score</td><td>15.48</td><td>20.60</td></tr>'
            '</tbody>'
            '</table>'
            '<div style="font-size:0.80rem; color:#92400E; font-weight:600;">⚠️ Undercoverage driven by distribution shift: summer cooling calibration vs autumn/winter test.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Next Stage Guidance Button
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("⚡ Proceed to Energy-Mix Optimizer (Model E) ➔", type="primary", use_container_width=True):
        set_page("7. Energy-Mix Optimizer (Model E)")
        st.rerun()

    render_disclaimer(
        "<b>Marginal vs. Joint Coverage Caution</b>: Conformal guarantees hold marginally for each individual time step. "
        "They do not guarantee simultaneous joint band coverage across all 240 consecutive hours of a multi-day dispatch horizon.",
        warning=True,
    )


# =============================================================================
# PAGE 7: ENERGY-MIX OPTIMIZER (MODEL E)
# =============================================================================
elif st.session_state["nav_page"] == "7. Energy-Mix Optimizer (Model E)":
    render_header(
        "Model E: Uncertainty-Aware Renewable Energy-Mix Optimizer",
        "Deterministic hourly energy-balance simulation, BESS storage dispatch, and multi-objective capacity optimization.",
    )

    # Preset Action Buttons
    st.markdown('<div class="section-heading"><span>⚡</span><span>Quick Scenario Presets</span></div>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)

    if "opt_solar" not in st.session_state:
        st.session_state["opt_solar"] = 150
    if "opt_wind" not in st.session_state:
        st.session_state["opt_wind"] = 0
    if "opt_batt" not in st.session_state:
        st.session_state["opt_batt"] = 200

    with p_col1:
        if st.button("🏆 Apply Cost-Optimal Mix", type="primary", use_container_width=True):
            st.session_state["opt_solar"] = 150
            st.session_state["opt_wind"] = 0
            st.session_state["opt_batt"] = 200
            st.rerun()

    with p_col2:
        if st.button("⚡ Apply Autonomy-Optimal Mix", use_container_width=True):
            st.session_state["opt_solar"] = 150
            st.session_state["opt_wind"] = 300
            st.session_state["opt_batt"] = 200
            st.rerun()

    with p_col3:
        if st.button("🔄 Reset to 100% Grid Baseline", use_container_width=True):
            st.session_state["opt_solar"] = 0
            st.session_state["opt_wind"] = 0
            st.session_state["opt_batt"] = 0
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
            current_val=f"{st.session_state['opt_solar']} kW",
            impact="Reduces daytime grid imports; surplus generation charges the battery or is curtailed when battery is full.",
            icon="☀️",
        )
        solar_input = st.slider("Solar PV Capacity (kW):", min_value=0, max_value=300, value=st.session_state["opt_solar"], step=25)
        st.session_state["opt_solar"] = solar_input

    with c_s2:
        render_control_context_card(
            param_name="Wind Turbine Capacity (kW)",
            description="Nameplate turbine capacity scaled via 2.0 MW power curve. Generates power during both day and night.",
            unit="Kilowatts (kW)",
            min_val=0,
            max_val=600,
            current_val=f"{st.session_state['opt_wind']} kW",
            impact="Provides nighttime clean energy; high capacity increases independence but raises capital and curtailment.",
            icon="🌀",
        )
        wind_input = st.slider("Wind Turbine Capacity (kW):", min_value=0, max_value=600, value=st.session_state["opt_wind"], step=50)
        st.session_state["opt_wind"] = wind_input

    with c_s3:
        render_control_context_card(
            param_name="Battery Storage Capacity (kWh)",
            description="Usable BESS energy capacity operating with 84.64% round-trip efficiency, 10%–95% SOC limits, and 0.5C power limit.",
            unit="Kilowatt-hours (kWh)",
            min_val=0,
            max_val=500,
            current_val=f"{st.session_state['opt_batt']} kWh",
            impact="Buffers surplus daytime renewable energy for discharge during peak evening demand hours.",
            icon="🔋",
        )
        batt_input = st.slider("Battery Storage Capacity (kWh):", min_value=0, max_value=500, value=st.session_state["opt_batt"], step=50)
        st.session_state["opt_batt"] = batt_input

    # Simulation execution
    dispatch, metrics, costs = simulate_custom_system(
        solar_kw=float(solar_input),
        wind_kw=float(wind_input),
        battery_kwh=float(batt_input),
        grid_import_limit_kw=200.0,
        duration_hours=240,
    )

    # Results Metric Cards
    st.markdown('<div class="section-heading"><span>📊</span><span>Simulation Results & Performance</span></div>', unsafe_allow_html=True)
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    with r1:
        st.metric("Renewable Penetration", f"{metrics['renewable_penetration']*100:.1f}%")
    with r2:
        st.metric("Grid Import", f"{metrics['grid_import_kwh']:,.1f} kWh")
    with r3:
        st.metric("Unmet Demand", f"{metrics['unmet_demand_kwh']:.1f} kWh")
    with r4:
        st.metric("Renewable Curtailment", f"{metrics['curtailment_kwh']:,.1f} kWh")
    with r5:
        st.metric("Total 20-Yr Cost", f"${costs['total_cost']:,.2f}")
    with r6:
        savings = (1414.63 - costs["total_cost"]) / 1414.63 * 100
        st.metric("Cost Savings", f"{savings:+.1f}%")

    # Donut Chart & Stacked Dispatch Area Chart
    ch_col1, ch_col2 = st.columns([35, 65])
    with ch_col1:
        fig_donut_opt = create_energy_mix_donut_chart(
            solar_val=solar_input,
            wind_val=wind_input,
            battery_val=batt_input,
            grid_val=metrics['grid_import_kwh'] / 240.0,
            total_label=f"{solar_input+wind_input+batt_input} kW",
            title="System Generation Mix",
            subtitle="Configured capacity breakdown",
        )
        st.plotly_chart(fig_donut_opt, use_container_width=True)

    with ch_col2:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-header">'
            '<span>⚡</span>'
            '<span>Energy Supply & Demand Balance (First 96 Hours)</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        t_slice = 96
        timestamps = [f"Hour {h}" for h in range(t_slice)]
        fig_dispatch = create_dispatch_dashboard_chart(
            timestamps=timestamps,
            demand=dispatch.demand[:t_slice],
            renewable_used=dispatch.renewable_used[:t_slice],
            battery_discharge=dispatch.battery_discharge[:t_slice],
            grid_import=dispatch.grid_import[:t_slice],
            curtailment=dispatch.curtailment[:t_slice],
            battery_soc=dispatch.battery_soc[:t_slice],
            title="Hourly Energy Dispatch & Supply Mix",
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
    scenarios = [
        {"scenario": "Conservative Stress (R_low, D_upp)", "demand_kwh": 12931.6, "renewable_kwh": 2795.3, "grid_import_kwh": 10136.3, "curtailment_kwh": 0.0},
        {"scenario": "Central Expected (R_cent, D_cent)", "demand_kwh": 11788.6, "renewable_kwh": 4517.6, "grid_import_kwh": 7271.0, "curtailment_kwh": 0.0},
        {"scenario": "Favorable Optimistic (R_upp, D_low)", "demand_kwh": 10645.5, "renewable_kwh": 6239.9, "grid_import_kwh": 4405.6, "curtailment_kwh": 0.0},
    ]
    fig_scenarios = create_scenario_comparison_bar_chart(scenarios)
    st.plotly_chart(fig_scenarios, use_container_width=True)

    render_disclaimer(
        "<b>Decision-Support PoC Caveat</b>: The integrated scenario synchronizes four independent real datasets across Stanford, Mumbai, Gujarat, and Orlando "
        "for the sole purpose of stress-testing the multi-modal uncertainty framework. It does not represent an empirical physical installation.",
        warning=True,
    )


# =============================================================================
# PAGE 8: METHODOLOGY & SCIENTIFIC DISCLOSURES
# =============================================================================
elif st.session_state["nav_page"] == "8. Methodology & Disclosures":
    render_header(
        "Methodology & Scientific Disclosures",
        "Comprehensive documentation of deep learning models, data provenance, physical conversion models, and research limitations.",
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
    st.latex(r"R_{\text{used}, t} + G_{\text{import}, t} + U_t = D_{\text{served}, t} + U_t = D_t")
    st.latex(r"R_{\text{avail}, t} = R_{\text{used}, t} + B_{\text{ch}, t} + C_t")
    
    st.markdown(
        """
        <div class="chart-card">
            <div style="font-weight: 700; color: #172033; font-size: 1.05rem; margin-bottom: 0.5rem;">🔋 Battery Energy Storage System (BESS) Dynamics</div>
            <ul style="color: #334155; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0;">
                <li><b>Efficiency Parameters</b>: Charge efficiency $\\eta_{\\text{ch}} = 0.92$, discharge efficiency $\\eta_{\\text{dis}} = 0.92$, yielding an empirical round-trip efficiency $\\eta_{\\text{rt}} \\approx 84.64\\%$.</li>
                <li><b>Operational Constraints</b>: Strict State-of-Charge limits $0.10 \\le \\text{SOC}_t \\le 0.95$ to prevent degradation; maximum C-rate limited to $0.5\\text{C}$ per hour.</li>
                <li><b>Conservation Assertion</b>: Exact machine-precision energy conservation $|\\text{residual}| \\le 10^{-6}\\text{ kWh}$ verified at every discrete simulation step across all 120 candidate mixes.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading"><span>🛡️</span><span>Split Conformal Prediction Formulation</span></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chart-card">
            <div style="font-weight: 700; color: #172033; font-size: 1.05rem; margin-bottom: 0.5rem;">Finite-Sample Distribution-Free Uncertainty Intervals</div>
            <p style="color: #334155; font-size: 0.88rem; line-height: 1.6;">
                Non-conformity absolute residuals are computed on a dedicated, strictly chronological calibration split $D_{\\text{cal}} = \\{(x_i, y_i)\\}_{i=1}^n$:
            </p>
            <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:0.5rem 1rem; margin:0.5rem 0; font-family:monospace; font-size:0.9rem; color:#172033;">
                s_i = |y_i - f(x_i)|, &nbsp;&nbsp; \hat{q} = \text{Quantile}\left(s_{1:n}, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)
            </div>
            <p style="color: #334155; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0;">
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

    render_disclaimer(
        "<b>Demand Distribution Shift & Undercoverage</b>: Model D observed 85.47% empirical coverage on the test set against a 90.0% nominal target. "
        "This undercoverage stems from an authentic seasonal shift between summer cooling calibration and autumn/winter test conditions.",
        warning=True,
    )
