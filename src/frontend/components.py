"""
Reusable UI layout components and styling utilities for the Streamlit dashboard.
Redesigned with high-contrast, WCAG AA compliant light-theme tokens,
robust sidebar visibility, and zero-indent HTML rendering to eliminate code-block artefacts.
"""
from typing import Any, Dict, List, Optional
import streamlit as st


def apply_custom_styles() -> None:
    """
    Inject high-contrast, WCAG AA compliant light-theme CSS styling.
    Strictly adheres to recommended design tokens:
    - Main page background: #F8FAFC
    - Card/Panel background: #FFFFFF
    - Primary text: #172033
    - Secondary text: #475569
    - Muted text: #64748B
    - Primary green: #047857
    - Primary blue: #0369A1
    - Borders: #CBD5E1
    - Sidebar background: #FFFFFF
    - Sidebar text: #334155
    - Active sidebar background: #D1FAE5
    - Active sidebar text: #065F46
    - Warning background: #FFFBEB
    - Warning text: #92400E
    - Warning border: #F59E0B
    """
    css_content = """<style>
/* Modern typography import */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* Global Application Typography and Canvas */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #172033;
}

.stApp {
    background-color: #F8FAFC !important;
    color: #172033 !important;
}

/* Headings and Text Elements */
h1, h2, h3, h4, h5, h6 {
    color: #172033 !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
}

p, span, label {
    color: #172033;
}

/* Top Navigation Bar */
.top-navbar-card {
    background: #FFFFFF;
    padding: 0.85rem 1.5rem;
    border-radius: 12px;
    border: 1px solid #CBD5E1;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.25rem;
}

.brand-container {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.brand-logo {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #047857 0%, #065F46 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-size: 1.35rem;
    font-weight: 800;
    box-shadow: 0 2px 6px rgba(4, 120, 87, 0.25);
}

.brand-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #172033;
    line-height: 1.2;
    margin: 0;
    letter-spacing: -0.015em;
}

.brand-subtitle {
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
    margin: 0;
}

.navbar-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #D1FAE5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    padding: 0.42rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #047857;
    border-radius: 50%;
    display: inline-block;
}

/* Page Hero Headings */
.page-hero-title {
    font-size: 2.0rem;
    font-weight: 800;
    color: #172033;
    letter-spacing: -0.025em;
    margin-bottom: 0.35rem;
    line-height: 1.2;
}

.page-hero-subtitle {
    font-size: 1.02rem;
    color: #475569;
    font-weight: 400;
    margin-bottom: 1.25rem;
    line-height: 1.45;
}

/* Section Headings */
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #172033;
    margin-top: 1.75rem;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* SIDEBAR STYLING - High Contrast & Visible States */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #CBD5E1 !important;
}

/* Radio Navigation Buttons */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.3rem !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 0.65rem 0.95rem !important;
    border-radius: 8px !important;
    margin-bottom: 0.25rem !important;
    transition: all 0.15s ease !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    background-color: transparent !important;
    border-left: 3px solid transparent !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label span,
section[data-testid="stSidebar"] div[role="radiogroup"] > label div {
    color: #334155 !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #F1F5F9 !important;
    border-color: #E2E8F0 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover span {
    color: #172033 !important;
}

/* Active Radio Item */
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: #D1FAE5 !important;
    border-left: 4px solid #047857 !important;
    border-color: #A7F3D0 #A7F3D0 #A7F3D0 #047857 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] span,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) span {
    color: #065F46 !important;
    font-weight: 700 !important;
}

/* Sidebar Scope Card */
.scope-card {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.scope-header {
    font-size: 0.92rem;
    font-weight: 700;
    color: #172033;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0.5rem;
}

.scope-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: #334155;
    margin-bottom: 0.45rem;
    line-height: 1.3;
}

.scope-check {
    color: #047857;
    font-size: 0.95rem;
    font-weight: 800;
}

/* 4 MODEL OVERVIEW METRIC CARDS */
.model-card {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}

.model-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07);
    transform: translateY(-2px);
}

.card-green {
    border-top: 4px solid #047857;
}

.card-blue {
    border-top: 4px solid #0369A1;
}

.card-teal {
    border-top: 4px solid #0D9488;
}

.card-purple {
    border-top: 4px solid #7C3AED;
}

.card-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}

.card-icon-bubble {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.bubble-green { background: #D1FAE5; color: #065F46; }
.bubble-blue { background: #E0F2FE; color: #0369A1; }
.bubble-teal { background: #CCFBF1; color: #0D9488; }
.bubble-purple { background: #F3E8FF; color: #6D28D9; }

.card-badge-pill {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.22rem 0.55rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.pill-green { background: #D1FAE5; color: #065F46; border: 1px solid #A7F3D0; }
.pill-blue { background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
.pill-teal { background: #CCFBF1; color: #0D9488; border: 1px solid #99F6E4; }
.pill-purple { background: #F3E8FF; color: #6D28D9; border: 1px solid #DDD6FE; }

.card-model-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: #172033;
    margin-bottom: 0.25rem;
}

.card-metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #172033;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
    line-height: 1.15;
}

.card-metric-sub {
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
}

.card-sparkline {
    width: 100%;
    height: 32px;
    margin-top: 0.65rem;
    display: block;
}

/* CHART & CONTAINER CARDS */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-bottom: 1.25rem;
}

.chart-card-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #172033;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.chart-card-sub {
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 0.85rem;
}

/* QUICK ACTION CARDS */
.quick-action-card {
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    margin-bottom: 0.5rem;
}

.quick-action-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.12);
}

.qa-teal {
    background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
}

.qa-blue {
    background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
}

.qa-purple {
    background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
}

.qa-content {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.qa-icon-box {
    width: 44px;
    height: 44px;
    background: rgba(255, 255, 255, 0.20);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
}

.qa-text-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
}

.qa-text-sub {
    font-size: 0.82rem;
    color: rgba(255, 255, 255, 0.90);
    margin: 0;
    font-weight: 400;
}

.qa-arrow {
    font-size: 1.3rem;
    color: #FFFFFF;
    opacity: 0.95;
}

/* CONTROL CONTEXT PANELS (SLIDER GUIDES) */
.control-context-card {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-left: 4px solid #047857;
    border-radius: 10px;
    padding: 0.95rem 1.15rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.control-context-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #172033;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.control-context-desc {
    font-size: 0.83rem;
    color: #475569;
    line-height: 1.4;
    margin-bottom: 0.45rem;
}

.control-context-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    font-size: 0.75rem;
}

.control-badge {
    background: #F1F5F9;
    color: #334155;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-weight: 600;
    border: 1px solid #E2E8F0;
}

.control-badge-active {
    background: #D1FAE5;
    color: #065F46;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-weight: 700;
    border: 1px solid #A7F3D0;
}

/* STYLED HIGH-CONTRAST TABLES */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    overflow: hidden;
    margin: 1rem 0;
    font-size: 0.88rem;
}

.styled-table th {
    background-color: #F1F5F9;
    color: #172033;
    font-weight: 700;
    padding: 9px 12px;
    text-align: left;
    border-bottom: 2px solid #CBD5E1;
    border-right: 1px solid #E2E8F0;
}

.styled-table th:last-child {
    border-right: none;
}

.styled-table td {
    padding: 9px 12px;
    color: #172033;
    border-bottom: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    vertical-align: middle;
}

.styled-table td:last-child {
    border-right: none;
}

.styled-table tr:last-child td {
    border-bottom: none;
}

.styled-table tr:hover td {
    background-color: #F8FAFC;
}

/* SCIENTIFIC DISCLAIMER BOXES */
.callout-amber {
    background-color: #FFFBEB;
    border: 1px solid #F59E0B;
    border-left: 5px solid #F59E0B;
    padding: 1rem 1.25rem;
    border-radius: 10px;
    margin: 1.25rem 0;
    font-size: 0.92rem;
    color: #92400E;
    box-shadow: 0 1px 2px rgba(245, 158, 11, 0.05);
    line-height: 1.45;
}

.callout-info {
    background-color: #F0F9FF;
    border: 1px solid #0369A1;
    border-left: 5px solid #0369A1;
    padding: 1rem 1.25rem;
    border-radius: 10px;
    margin: 1.25rem 0;
    font-size: 0.92rem;
    color: #075985;
    line-height: 1.45;
}

.callout-provenance {
    background-color: #F0FDF4;
    border: 1px solid #047857;
    border-left: 5px solid #047857;
    padding: 0.95rem 1.25rem;
    border-radius: 10px;
    margin: 1rem 0;
    font-size: 0.90rem;
    color: #065F46;
    line-height: 1.45;
}

/* HIGH-CONTRAST INTERACTIVE BUTTONS */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.15rem !important;
    font-size: 0.92rem !important;
    transition: all 0.15s ease-in-out !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #047857 0%, #065F46 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #047857 !important;
    box-shadow: 0 2px 4px rgba(4, 120, 87, 0.25) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #065F46 0%, #064E3B 100%) !important;
    box-shadow: 0 4px 10px rgba(4, 120, 87, 0.35) !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) {
    background-color: #FFFFFF !important;
    color: #172033 !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

.stButton > button[kind="secondary"]:hover, .stButton > button:not([kind="primary"]):hover {
    background-color: #F8FAFC !important;
    border-color: #64748B !important;
    color: #172033 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    transform: translateY(-1px) !important;
}

/* Streamlit Slider High Contrast */
div[data-testid="stSlider"] label {
    color: #172033 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}

div[data-testid="stSlider"] div[data-testid="stSliderThumbValue"] {
    color: #047857 !important;
    font-weight: 800 !important;
    font-size: 0.88rem !important;
}

div[data-testid="stSlider"] div[data-testid="stSliderTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stSliderTickBarMax"] {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.80rem !important;
}

/* Streamlit Expander Styling */
div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
}

div[data-testid="stExpander"] details summary span {
    color: #172033 !important;
    font-weight: 700 !important;
}

/* Streamlit Tabs Styling */
div[data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #CBD5E1 !important;
}

div[data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    color: #047857 !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #047857 !important;
}
</style>"""
    st.markdown(css_content, unsafe_allow_html=True)


def render_top_navbar() -> None:
    """Render top application brand banner matching the prototype."""
    html_code = (
        '<div class="top-navbar-card">'
        '<div style="display: flex; align-items: center; justify-content: space-between;">'
        '<div class="brand-container">'
        '<div class="brand-logo">⚡</div>'
        '<div>'
        '<h2 class="brand-title">Renewable Energy Intelligence</h2>'
        '<p class="brand-subtitle">A Multimodal, Uncertainty-Aware Decision-Support Framework</p>'
        '</div>'
        '</div>'
        '<div style="display: flex; align-items: center; gap: 0.75rem;">'
        '<span class="navbar-status-badge">'
        '<span class="status-dot"></span>'
        'Models Ready • Research Prototype'
        '</span>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def render_sidebar_scope() -> None:
    """Render the Project Scope checklist card at the bottom of the sidebar."""
    html_code = (
        '<div class="scope-card">'
        '<div class="scope-header">'
        '<span>🎓</span>'
        '<span>Project Scope</span>'
        '</div>'
        '<div class="scope-item">'
        '<span class="scope-check">✔</span>'
        '<span>4 Deep Learning Models (CNN + LSTM)</span>'
        '</div>'
        '<div class="scope-item">'
        '<span class="scope-check">✔</span>'
        '<span>Conformal Uncertainty Intervals</span>'
        '</div>'
        '<div class="scope-item">'
        '<span class="scope-check">✔</span>'
        '<span>Energy-Mix Optimization</span>'
        '</div>'
        '<div class="scope-item">'
        '<span class="scope-check">✔</span>'
        '<span>Real Datasets (Multiple Locations)</span>'
        '</div>'
        '<div class="scope-item">'
        '<span class="scope-check">✔</span>'
        '<span>Interactive Dashboard</span>'
        '</div>'
        '</div>'
    )
    st.sidebar.markdown(html_code, unsafe_allow_html=True)


def render_header(title: str, subtitle: Optional[str] = None) -> None:
    """Render standardized page hero title with high contrast."""
    st.markdown(f'<div class="page-hero-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-hero-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_model_card(
    model_id: str,
    title: str,
    metric_value: str,
    metric_label: str,
    color_theme: str = "green",
    icon: str = "☀️",
    badge: str = "TEST SET",
) -> None:
    """
    Render a clean, high-contrast model overview card with sparkline graphic.
    Constructed as a single unindented HTML string with zero leading spaces to
    completely eliminate CommonMark code-block (</div>) rendering artefacts.
    """
    bubble_class = f"bubble-{color_theme}"
    pill_class = f"pill-{color_theme}"
    card_class = f"card-{color_theme}"

    stroke_color = {
        "green": "#047857",
        "blue": "#0369A1",
        "teal": "#0D9488",
        "purple": "#7C3AED",
    }.get(color_theme, "#047857")

    fill_color = {
        "green": "rgba(4, 120, 87, 0.12)",
        "blue": "rgba(3, 105, 161, 0.12)",
        "teal": "rgba(13, 148, 136, 0.12)",
        "purple": "rgba(124, 58, 237, 0.12)",
    }.get(color_theme, "rgba(4, 120, 87, 0.12)")

    svg_sparkline = (
        f'<svg class="card-sparkline" viewBox="0 0 100 25" preserveAspectRatio="none">'
        f'<path d="M0,20 Q15,5 30,15 T60,8 T85,18 T100,5 L100,25 L0,25 Z" fill="{fill_color}"/>'
        f'<path d="M0,20 Q15,5 30,15 T60,8 T85,18 T100,5" fill="none" stroke="{stroke_color}" stroke-width="2"/>'
        f'</svg>'
    )

    card_html = (
        f'<div class="model-card {card_class}">'
        f'<div class="card-header-row">'
        f'<div class="card-icon-bubble {bubble_class}">{icon}</div>'
        f'<span class="card-badge-pill {pill_class}">{badge}</span>'
        f'</div>'
        f'<div class="card-model-name">{title}</div>'
        f'<div class="card-metric-value">{metric_value}</div>'
        f'<div class="card-metric-sub">{metric_label}</div>'
        f'{svg_sparkline}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_quick_action_card(
    title: str,
    subtitle: str,
    theme: str = "teal",
    icon: str = "📷",
) -> None:
    """Render a styled quick-action card banner."""
    qa_class = f"qa-{theme}"
    html_code = (
        f'<div class="quick-action-card {qa_class}">'
        f'<div class="qa-content">'
        f'<div class="qa-icon-box">{icon}</div>'
        f'<div>'
        f'<div class="qa-text-title">{title}</div>'
        f'<div class="qa-text-sub">{subtitle}</div>'
        f'</div>'
        f'</div>'
        f'<div class="qa-arrow">➔</div>'
        f'</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def render_control_context_card(
    param_name: str,
    description: str,
    unit: Optional[str] = None,
    min_val: Optional[Any] = None,
    max_val: Optional[Any] = None,
    current_val: Optional[Any] = None,
    impact: Optional[str] = None,
    icon: str = "🎛️",
) -> None:
    """
    Render a clean, informative context panel explaining a slider/input control.
    Provides users with complete understanding of parameter meaning, unit, range, and effect.
    """
    unit_badge = f'<span class="control-badge"><b>Unit:</b> {unit}</span>' if unit else ""
    range_badge = f'<span class="control-badge"><b>Permitted Range:</b> [{min_val} – {max_val}]</span>' if min_val is not None and max_val is not None else ""
    active_badge = f'<span class="control-badge-active"><b>Selected:</b> {current_val}</span>' if current_val is not None else ""
    impact_html = f'<div style="font-size: 0.80rem; color: #334155; margin-top: 0.35rem;"><b>💡 Downstream Effect:</b> {impact}</div>' if impact else ""

    html_code = (
        f'<div class="control-context-card">'
        f'<div class="control-context-title">'
        f'<span>{icon} {param_name}</span>'
        f'</div>'
        f'<div class="control-context-desc">{description}</div>'
        f'<div class="control-context-badges">'
        f'{unit_badge}'
        f'{range_badge}'
        f'{active_badge}'
        f'</div>'
        f'{impact_html}'
        f'</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def render_disclaimer(message: str, warning: bool = True) -> None:
    """Render scientific caveat callout box with high contrast."""
    css_class = "callout-amber" if warning else "callout-info"
    prefix = "⚠️ <b>Research Prototype Note</b>: " if warning else "ℹ️ <b>Methodological Note</b>: "
    html_code = (
        f'<div class="{css_class}">'
        f'{prefix}{message}'
        f'</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def render_provenance_badge(
    dataset_name: str,
    provenance_type: str,
    location: str,
    year_period: str,
    target_units: str,
) -> None:
    """Render scientific data provenance badge with exact geography and physical units."""
    html_code = (
        f'<div class="callout-provenance">'
        f'<strong>📍 Dataset Provenance & Physical Units</strong><br>'
        f'• <b>Source</b>: {dataset_name} &nbsp;|&nbsp; '
        f'• <b>Category</b>: <span style="background:#D1FAE5; color:#065F46; font-weight:700; padding:2px 8px; border-radius:9999px; font-size:0.78rem; border:1px solid #A7F3D0;">{provenance_type}</span><br>'
        f'• <b>Location</b>: {location} &nbsp;|&nbsp; '
        f'• <b>Period</b>: {year_period} &nbsp;|&nbsp; '
        f'• <b>Physical Unit</b>: <code>{target_units}</code>'
        f'</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)
