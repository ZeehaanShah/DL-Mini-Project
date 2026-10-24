"""
Reusable UI layout components and styling utilities for the Streamlit dashboard.
Redesigned to match the clean light-theme renewable energy analytics prototype.
"""
from typing import Optional
import streamlit as st


def apply_custom_styles() -> None:
    """Inject clean light-theme CSS styling matching the prototype."""
    st.markdown(
        """
        <style>
        /* Import clean modern typography */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        /* Global App Styling - Light Theme */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0F172A;
        }

        .stApp {
            background-color: #F8FAFC;
        }

        /* Top Brand Header Bar */
        .top-navbar-card {
            background: #FFFFFF;
            padding: 0.85rem 1.5rem;
            border-radius: 14px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
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
            background: linear-gradient(135deg, #059669 0%, #10B981 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.4rem;
            font-weight: 800;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);
        }

        .brand-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
            margin: 0;
            letter-spacing: -0.015em;
        }

        .brand-subtitle {
            font-size: 0.82rem;
            color: #64748B;
            font-weight: 500;
            margin: 0;
        }

        .navbar-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: #ECFDF5;
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
            background-color: #10B981;
            border-radius: 50%;
            display: inline-block;
        }

        /* Hero / Page Titles */
        .page-hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.025em;
            margin-bottom: 0.35rem;
            line-height: 1.2;
        }

        .page-hero-subtitle {
            font-size: 1.05rem;
            color: #475569;
            font-weight: 400;
            margin-bottom: 1.5rem;
        }

        /* 4 Model Overview Metric Cards */
        .model-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            position: relative;
            overflow: hidden;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            height: 100%;
        }

        .model-card:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            transform: translateY(-2px);
        }

        .card-green {
            border-top: 4px solid #10B981;
        }

        .card-blue {
            border-top: 4px solid #0284C7;
        }

        .card-teal {
            border-top: 4px solid #0D9488;
        }

        .card-purple {
            border-top: 4px solid #8B5CF6;
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
            font-size: 1.15rem;
        }

        .bubble-green { background: #ECFDF5; color: #059669; }
        .bubble-blue { background: #E0F2FE; color: #0284C7; }
        .bubble-teal { background: #CCFBF1; color: #0D9488; }
        .bubble-purple { background: #F3E8FF; color: #7C3AED; }

        .card-badge-pill {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .pill-green { background: #DCFCE7; color: #166534; }
        .pill-blue { background: #E0F2FE; color: #075985; }
        .pill-teal { background: #CCFBF1; color: #115E59; }
        .pill-purple { background: #F3E8FF; color: #6B21A8; }

        .card-model-name {
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            margin-bottom: 0.2rem;
        }

        .card-metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
            line-height: 1.1;
        }

        .card-metric-sub {
            font-size: 0.80rem;
            color: #64748B;
            font-weight: 500;
        }

        /* SVG Sparkline Graphic in Card */
        .card-sparkline {
            width: 100%;
            height: 32px;
            margin-top: 0.65rem;
        }

        /* Quick Action Cards */
        .quick-action-card {
            border-radius: 14px;
            padding: 1.15rem 1.35rem;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            margin-bottom: 0.5rem;
        }

        .quick-action-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
        }

        .qa-teal {
            background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        }

        .qa-blue {
            background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        }

        .qa-navy {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        }

        .qa-content {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .qa-icon-box {
            width: 44px;
            height: 44px;
            background: rgba(255, 255, 255, 0.18);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.35rem;
            backdrop-filter: blur(4px);
        }

        .qa-text-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 0;
        }

        .qa-text-sub {
            font-size: 0.82rem;
            color: rgba(255, 255, 255, 0.85);
            margin: 0;
            font-weight: 400;
        }

        .qa-arrow {
            font-size: 1.4rem;
            color: #FFFFFF;
            opacity: 0.9;
        }

        /* Scientific Disclaimer Boxes */
        .callout-amber {
            background-color: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 5px solid #F59E0B;
            padding: 1rem 1.25rem;
            border-radius: 10px;
            margin: 1.25rem 0;
            font-size: 0.92rem;
            color: #92400E;
            box-shadow: 0 1px 2px rgba(245, 158, 11, 0.05);
        }

        .callout-info {
            background-color: #F0F9FF;
            border: 1px solid #BAE6FD;
            border-left: 5px solid #0284C7;
            padding: 1rem 1.25rem;
            border-radius: 10px;
            margin: 1.25rem 0;
            font-size: 0.92rem;
            color: #0369A1;
        }

        .callout-provenance {
            background-color: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-left: 5px solid #10B981;
            padding: 0.95rem 1.25rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-size: 0.92rem;
            color: #14532D;
        }

        /* High-Contrast Interactive Buttons */
        .stButton > button {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.25rem !important;
            font-size: 0.92rem !important;
            transition: all 0.15s ease-in-out !important;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(13, 148, 136, 0.25) !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0F766E 0%, #115E59 100%) !important;
            box-shadow: 0 4px 10px rgba(13, 148, 136, 0.35) !important;
            transform: translateY(-1px) !important;
        }

        .stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        }

        .stButton > button[kind="secondary"]:hover, .stButton > button:not([kind="primary"]):hover {
            background-color: #F8FAFC !important;
            border-color: #94A3B8 !important;
            color: #0F172A !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
            transform: translateY(-1px) !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            padding: 0.65rem 0.85rem !important;
            border-radius: 10px !important;
            margin-bottom: 0.35rem !important;
            transition: all 0.15s ease !important;
            font-size: 0.90rem !important;
            font-weight: 500 !important;
            color: #334155 !important;
            border-left: 3px solid transparent !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background-color: #F1F5F9 !important;
            color: #0F172A !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background-color: #ECFDF5 !important;
            color: #065F46 !important;
            font-weight: 700 !important;
            border-left: 3px solid #0D9488 !important;
        }

        /* Project Scope Checklist in Sidebar */
        .scope-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
        }

        .scope-header {
            font-size: 0.88rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.65rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .scope-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.80rem;
            color: #475569;
            margin-bottom: 0.4rem;
        }

        .scope-check {
            color: #10B981;
            font-size: 0.95rem;
            font-weight: bold;
        }

        /* Chart & Container Cards */
        .chart-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            margin-bottom: 1.25rem;
        }

        .chart-card-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .chart-card-sub {
            font-size: 0.82rem;
            color: #64748B;
            margin-bottom: 0.85rem;
        }

        /* Section dividers */
        .section-heading {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 1.75rem;
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_navbar() -> None:
    """Render top application brand banner matching the prototype."""
    st.markdown(
        """
        <div class="top-navbar-card">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div class="brand-container">
                    <div class="brand-logo">⚡</div>
                    <div>
                        <h2 class="brand-title">Renewable Energy Intelligence</h2>
                        <p class="brand-subtitle">A Multimodal, Uncertainty-Aware Decision-Support Framework</p>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span class="navbar-status-badge">
                        <span class="status-dot"></span>
                        Models Ready • Research Prototype
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_scope() -> None:
    """Render the Project Scope checklist card at the bottom of the sidebar."""
    st.sidebar.markdown(
        """
        <div class="scope-card">
            <div class="scope-header">
                <span>🎓</span>
                <span>Project Scope</span>
            </div>
            <div class="scope-item">
                <span class="scope-check">✔</span>
                <span>4 Deep Learning Models (CNN + LSTM)</span>
            </div>
            <div class="scope-item">
                <span class="scope-check">✔</span>
                <span>Conformal Uncertainty Intervals</span>
            </div>
            <div class="scope-item">
                <span class="scope-check">✔</span>
                <span>Energy-Mix Optimization</span>
            </div>
            <div class="scope-item">
                <span class="scope-check">✔</span>
                <span>Real Datasets (Multiple Locations)</span>
            </div>
            <div class="scope-item">
                <span class="scope-check">✔</span>
                <span>Interactive Dashboard</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: Optional[str] = None) -> None:
    """Render standardized page hero title."""
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
    Render a clean model overview card with sparkline graphic matching the prototype.
    color_theme: 'green', 'blue', 'teal', 'purple'
    """
    bubble_class = f"bubble-{color_theme}"
    pill_class = f"pill-{color_theme}"
    card_class = f"card-{color_theme}"

    # Sparkline wave SVG with theme-matching color
    stroke_color = {
        "green": "#10B981",
        "blue": "#0284C7",
        "teal": "#0D9488",
        "purple": "#8B5CF6",
    }.get(color_theme, "#10B981")

    fill_color = {
        "green": "rgba(16, 185, 129, 0.12)",
        "blue": "rgba(2, 132, 199, 0.12)",
        "teal": "rgba(13, 148, 136, 0.12)",
        "purple": "rgba(139, 92, 246, 0.12)",
    }.get(color_theme, "rgba(16, 185, 129, 0.12)")

    svg_sparkline = f"""
    <svg class="card-sparkline" viewBox="0 0 100 25" preserveAspectRatio="none">
        <path d="M0,20 Q15,5 30,15 T60,8 T85,18 T100,5 L100,25 L0,25 Z" fill="{fill_color}"/>
        <path d="M0,20 Q15,5 30,15 T60,8 T85,18 T100,5" fill="none" stroke="{stroke_color}" stroke-width="2"/>
    </svg>
    """

    st.markdown(
        f"""
        <div class="model-card {card_class}">
            <div class="card-header-row">
                <div class="card-icon-bubble {bubble_class}">{icon}</div>
                <span class="card-badge-pill {pill_class}">{badge}</span>
            </div>
            <div class="card-model-name">{title}</div>
            <div class="card-metric-value">{metric_value}</div>
            <div class="card-metric-sub">{metric_label}</div>
            {svg_sparkline}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_action_card(
    title: str,
    subtitle: str,
    theme: str = "teal",
    icon: str = "📷",
) -> None:
    """Render a styled quick-action card banner."""
    qa_class = f"qa-{theme}"
    st.markdown(
        f"""
        <div class="quick-action-card {qa_class}">
            <div class="qa-content">
                <div class="qa-icon-box">{icon}</div>
                <div>
                    <div class="qa-text-title">{title}</div>
                    <div class="qa-text-sub">{subtitle}</div>
                </div>
            </div>
            <div class="qa-arrow">➔</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer(message: str, warning: bool = True) -> None:
    """Render scientific caveat callout box."""
    css_class = "callout-amber" if warning else "callout-info"
    prefix = "⚠️ <b>Research Prototype Note</b>: " if warning else "ℹ️ <b>Methodological Note</b>: "
    st.markdown(
        f"""
        <div class="{css_class}">
            {prefix}{message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_provenance_badge(
    dataset_name: str,
    provenance_type: str,
    location: str,
    year_period: str,
    target_units: str,
) -> None:
    """Render scientific data provenance badge with exact geography and physical units."""
    st.markdown(
        f"""
        <div class="callout-provenance">
            <strong>📍 Dataset Provenance & Physical Units</strong><br>
            • <b>Source</b>: {dataset_name} &nbsp;|&nbsp; 
            • <b>Category</b>: <span style="background:#DCFCE7; color:#166534; font-weight:700; padding:2px 8px; border-radius:9999px; font-size:0.78rem;">{provenance_type}</span><br>
            • <b>Location</b>: {location} &nbsp;|&nbsp; 
            • <b>Period</b>: {year_period} &nbsp;|&nbsp; 
            • <b>Physical Unit</b>: <code>{target_units}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
