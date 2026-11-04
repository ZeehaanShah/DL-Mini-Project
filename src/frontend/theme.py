"""Centralized theme colors and WCAG contrast utilities for the Streamlit UI."""
from __future__ import annotations

import re
from typing import Mapping, Sequence

_RGB_RE = re.compile(r"^rgba?\(\s*([^)]*)\s*\)$", re.IGNORECASE)


def parse_color(color: str) -> tuple[float, float, float, float]:
    """Parse #RGB[A], #RRGGBB[AA], or numeric rgb()/rgba() into normalized channels."""
    if not isinstance(color, str):
        raise TypeError("color must be a string")
    value = color.strip()
    if value.lower() == "transparent":
        return 0.0, 0.0, 0.0, 0.0
    if value.startswith("#"):
        raw = value[1:]
        if len(raw) in (3, 4):
            raw = "".join(ch * 2 for ch in raw)
        if len(raw) not in (6, 8) or not re.fullmatch(r"[0-9a-fA-F]+", raw):
            raise ValueError(f"Unsupported hex color: {color!r}")
        channels = [int(raw[i:i + 2], 16) / 255.0 for i in range(0, len(raw), 2)]
        alpha = channels[3] if len(channels) == 4 else 1.0
        return channels[0], channels[1], channels[2], alpha
    match = _RGB_RE.fullmatch(value)
    if match:
        parts = [part.strip() for part in match.group(1).split(",")]
        expected = 4 if value.lower().startswith("rgba") else 3
        if len(parts) != expected:
            raise ValueError(f"Invalid color component count: {color!r}")
        rgb: list[float] = []
        for part in parts[:3]:
            if part.endswith("%"):
                channel = float(part[:-1]) / 100.0
            else:
                channel = float(part) / 255.0
            if not 0.0 <= channel <= 1.0:
                raise ValueError(f"RGB channel out of range: {part!r}")
            rgb.append(channel)
        alpha = float(parts[3].rstrip("%")) / (100.0 if parts[3].endswith("%") else 1.0) if expected == 4 else 1.0
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha channel out of range: {parts[3]!r}")
        return rgb[0], rgb[1], rgb[2], alpha
    raise ValueError(f"Unsupported color syntax: {color!r}; use a solid CSS color")


def composite_colors(foreground: str, background: str) -> str:
    """Composite a foreground color over a background and return opaque #RRGGBB."""
    fr, fg, fb, fa = parse_color(foreground)
    br, bg, bb, ba = parse_color(background)
    out_alpha = fa + ba * (1.0 - fa)
    if out_alpha <= 0.0:
        raise ValueError("Cannot resolve a fully transparent composite without an opaque underlay")
    channels = [
        (front * fa + back * ba * (1.0 - fa)) / out_alpha
        for front, back in ((fr, br), (fg, bg), (fb, bb))
    ]
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in channels)


def _linear_channel(channel: float) -> float:
    """Convert an sRGB channel to linear relative luminance."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str, underlay: str = "#FFFFFF") -> float:
    """Calculate WCAG relative luminance after resolving color alpha over an underlay."""
    base = composite_colors(underlay, "#FFFFFF") if parse_color(underlay)[3] < 1.0 else composite_colors(underlay, underlay)
    effective = composite_colors(color, base)
    red, green, blue, _ = parse_color(effective)
    return 0.2126 * _linear_channel(red) + 0.7152 * _linear_channel(green) + 0.0722 * _linear_channel(blue)


def contrast_ratio(foreground: str, background: str, underlay: str = "#FFFFFF") -> float:
    """Return the WCAG contrast ratio for foreground/background, compositing alpha first."""
    background_rgba = parse_color(background)
    if background_rgba[3] < 1.0:
        effective_background = composite_colors(background, underlay)
    else:
        effective_background = composite_colors(background, background)
    foreground_rgba = parse_color(foreground)
    effective_foreground = composite_colors(foreground, effective_background) if foreground_rgba[3] < 1.0 else composite_colors(foreground, foreground)
    high, low = sorted((relative_luminance(effective_foreground), relative_luminance(effective_background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def accessible_foreground(
    background: str,
    *,
    underlay: str = "#FFFFFF",
    candidates: Sequence[str] = ("#172033", "#FFFFFF"),
    minimum_ratio: float = 4.5,
) -> str:
    """Choose the candidate foreground with highest contrast; fail if none meets the target."""
    if minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must exceed 1.0")
    if not candidates:
        raise ValueError("At least one foreground candidate is required")
    scored = [(contrast_ratio(candidate, background, underlay), candidate) for candidate in candidates]
    best_ratio, best = max(scored, key=lambda item: item[0])
    if best_ratio < minimum_ratio:
        raise ValueError(
            f"No candidate foreground meets {minimum_ratio:.1f}:1 against {background}; best is {best_ratio:.2f}:1"
        )
    return best


def _build_theme(mode: str) -> dict[str, str]:
    """Build validated semantic CSS/Plotly tokens for the selected supported theme."""
    if mode == "light":
        values = {
            "canvas": "#F8FAFC", "surface": "#FFFFFF", "surface_muted": "#F1F5F9",
            "surface_hover": "#E2E8F0", "border": "#64748B", "grid": "#CBD5E1",
            "brand": "#047857", "brand_strong": "#065F46", "brand_deep": "#064E3B",
            "teal": "#0D9488", "teal_action": "#0F766E", "blue": "#0369A1",
            "blue_deep": "#075985", "purple": "#7C3AED", "purple_deep": "#6D28D9",
            "amber": "#B45309", "danger": "#B91C1C", "danger_deep": "#7F1D1D",
            "success_surface": "#D1FAE5", "success_border": "#047857",
            "warning_surface": "#FFFBEB", "warning_accent": "#B45309",
            "info_surface": "#F0F9FF", "info_border": "#0369A1",
            "error_surface": "#FEF2F2", "error_border": "#B91C1C",
            "focus": "#0369A1", "text_secondary_preferred": "#475569", "text_muted_preferred": "#64748B",
        }
        foregrounds = {"text_primary": "#172033", "text_secondary": "#475569", "text_muted": "#64748B",
                       "text_on_brand": "#FFFFFF", "success_text_preferred": "#065F46",
                       "warning_text_preferred": "#92400E", "info_text_preferred": "#075985",
                       "error_text_preferred": "#991B1B"}
    elif mode == "dark":
        values = {
            "canvas": "#0E1117", "surface": "#1F2937", "surface_muted": "#273449",
            "surface_hover": "#334155", "border": "#94A3B8", "grid": "#475569",
            "brand": "#047857", "brand_strong": "#065F46", "brand_deep": "#064E3B",
            "teal": "#0D9488", "teal_action": "#0F766E", "blue": "#0369A1",
            "blue_deep": "#075985", "purple": "#7C3AED", "purple_deep": "#6D28D9",
            "amber": "#FBBF24", "danger": "#EF4444", "danger_deep": "#7F1D1D",
            "success_surface": "#064E3B", "success_border": "#34D399",
            "warning_surface": "#78350F", "warning_accent": "#FBBF24",
            "info_surface": "#0C4A6E", "info_border": "#38BDF8",
            "error_surface": "#7F1D1D", "error_border": "#F87171",
            "focus": "#38BDF8", "text_secondary_preferred": "#CBD5E1", "text_muted_preferred": "#CBD5E1",
        }
        foregrounds = {"text_primary": "#F8FAFC", "text_secondary": "#CBD5E1", "text_muted": "#CBD5E1",
                       "text_on_brand": "#FFFFFF", "success_text_preferred": "#D1FAE5",
                       "warning_text_preferred": "#FEF3C7", "info_text_preferred": "#E0F2FE",
                       "error_text_preferred": "#FEE2E2"}
    else:
        raise ValueError(f"Unsupported theme mode {mode!r}; expected 'light' or 'dark'")

    # Derive the actual ink colors from their paired surfaces and enforce AA for normal text.
    values["text_primary"] = accessible_foreground(values["canvas"], candidates=(foregrounds["text_primary"], "#172033", "#FFFFFF"))
    values["text_on_surface"] = accessible_foreground(values["surface"], candidates=(foregrounds["text_secondary"], values["text_primary"], "#FFFFFF"))
    values["text_secondary"] = accessible_foreground(values["surface"], candidates=(foregrounds["text_secondary"], values["text_primary"], "#FFFFFF"))
    values["text_muted"] = accessible_foreground(values["surface"], candidates=(foregrounds["text_muted"], values["text_primary"], "#FFFFFF"))
    values["text_on_brand"] = accessible_foreground(values["brand_strong"], candidates=(foregrounds["text_on_brand"], values["text_primary"], "#FFFFFF"))
    values["success_text"] = accessible_foreground(values["success_surface"], candidates=(foregrounds["success_text_preferred"], values["text_primary"], "#FFFFFF"))
    values["warning_text"] = accessible_foreground(values["warning_surface"], candidates=(foregrounds["warning_text_preferred"], values["text_primary"], "#FFFFFF"))
    values["info_text"] = accessible_foreground(values["info_surface"], candidates=(foregrounds["info_text_preferred"], values["text_primary"], "#FFFFFF"))
    values["error_text"] = accessible_foreground(values["error_surface"], candidates=(foregrounds["error_text_preferred"], values["text_primary"], "#FFFFFF"))
    values["text_on_teal"] = accessible_foreground(values["teal_action"], candidates=("#FFFFFF", values["text_primary"]))
    values["text_on_blue"] = accessible_foreground(values["blue_deep"], candidates=("#FFFFFF", values["text_primary"]))
    values["text_on_purple"] = accessible_foreground(values["purple_deep"], candidates=("#FFFFFF", values["text_primary"]))
    # CSS aliases used by existing component markup; keep their meaning semantic.
    values["teal_surface"] = values["success_surface"]
    values["blue_surface"] = values["info_surface"]
    values["purple_surface"] = values["surface_muted"]
    values["success_soft_surface"] = values["success_surface"]
    values["chart_actual"] = values["text_primary"]
    values["chart_forecast"] = values["blue"]
    values["chart_solar"] = values["amber"]
    values["chart_wind"] = "#38BDF8" if mode == "dark" else values["blue"]
    values["chart_battery"] = "#C4B5FD" if mode == "dark" else values["purple_deep"]
    values["chart_grid"] = values["text_secondary"]
    values["chart_unmet"] = values["danger"]
    values["chart_curtailment"] = values["warning_accent"]
    values["chart_focus"] = "#34D399" if mode == "dark" else values["brand"]
    return values


THEMES: Mapping[str, Mapping[str, str]] = {"light": _build_theme("light"), "dark": _build_theme("dark")}


def get_theme_tokens(mode: str = "light") -> Mapping[str, str]:
    """Return the validated semantic palette for light or dark rendering."""
    try:
        return THEMES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported theme mode {mode!r}; expected 'light' or 'dark'") from exc


def get_active_theme_mode() -> str:
    """Read the configured Streamlit base theme, falling back safely to the current light theme."""
    try:
        import streamlit as st
        configured = st.get_option("theme.base")
    except Exception:
        configured = None
    return configured if configured in ("light", "dark") else "light"


def css_variables(mode: str = "light") -> str:
    """Return CSS variable declarations for the selected semantic palette."""
    tokens = get_theme_tokens(mode)
    declarations = "\n".join(f"  --{name.replace('_', '-')}: {value};" for name, value in tokens.items())
    return f":root {{\n{declarations}\n}}"


def apply_plotly_theme(figure, mode: str | None = None):
    """Apply opaque, contrast-aware Plotly defaults to an existing figure."""
    selected_mode = mode or get_active_theme_mode()
    tokens = get_theme_tokens(selected_mode)
    figure.update_layout(
        template="plotly_dark" if selected_mode == "dark" else "plotly_white",
        paper_bgcolor=tokens["canvas"],
        plot_bgcolor=tokens["surface"],
        font={"color": tokens["text_primary"], "family": "Plus Jakarta Sans, Inter, -apple-system, sans-serif"},
        hoverlabel={"bgcolor": tokens["surface"], "bordercolor": tokens["border"], "font": {"color": tokens["text_primary"]}},
        colorway=[tokens["chart_solar"], tokens["chart_wind"], tokens["chart_battery"], tokens["chart_grid"]],
    )
    figure.update_xaxes(
        title_font_color=tokens["text_primary"], tickfont_color=tokens["text_secondary"],
        linecolor=tokens["border"], gridcolor=tokens["grid"], zerolinecolor=tokens["grid"],
    )
    figure.update_yaxes(
        title_font_color=tokens["text_primary"], tickfont_color=tokens["text_secondary"],
        linecolor=tokens["border"], gridcolor=tokens["grid"], zerolinecolor=tokens["grid"],
    )
    return figure
