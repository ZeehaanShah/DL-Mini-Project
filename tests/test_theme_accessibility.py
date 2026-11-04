"""Regression checks for WCAG contrast helpers and theme-aware chart defaults."""
import pytest
import plotly.graph_objects as go
from pathlib import Path
import re

from src.frontend.theme import (
    accessible_foreground, apply_plotly_theme, composite_colors, contrast_ratio,
    css_variables, get_theme_tokens, parse_color,
)


@pytest.mark.parametrize("color,expected", [
    ("#abc", (170 / 255, 187 / 255, 204 / 255, 1.0)),
    ("rgba(0, 128, 255, 0.5)", (0.0, 128 / 255, 1.0, 0.5)),
])
def test_parse_supported_colors(color, expected):
    assert parse_color(color) == pytest.approx(expected)


def test_alpha_compositing_and_wcag_known_ratio():
    assert composite_colors("rgba(0, 0, 0, 0.5)", "#FFFFFF") == "#808080"
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)


@pytest.mark.parametrize("invalid", ["not-a-color", "linear-gradient(red, blue)", "#12", "rgb(300,0,0)"])
def test_invalid_or_unsupported_colors_fail_closed(invalid):
    with pytest.raises((TypeError, ValueError)):
        parse_color(invalid)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_semantic_text_colors_meet_wcag_aa(mode):
    tokens = get_theme_tokens(mode)
    for foreground, background in [
        ("text_primary", "canvas"), ("text_secondary", "surface"),
        ("text_muted", "surface"), ("text_on_brand", "brand_strong"),
        ("success_text", "success_surface"), ("warning_text", "warning_surface"),
        ("info_text", "info_surface"), ("error_text", "error_surface"),
        ("text_on_teal", "teal_action"), ("text_on_blue", "blue_deep"),
        ("text_on_purple", "purple_deep"),
    ]:
        assert contrast_ratio(tokens[foreground], tokens[background]) >= 4.5, (mode, foreground, background)
    assert contrast_ratio(tokens["focus"], tokens["canvas"]) >= 3.0
    assert contrast_ratio(tokens["border"], tokens["surface"]) >= 3.0
    for series in ("chart_solar", "chart_wind", "chart_battery", "chart_grid", "chart_unmet", "chart_curtailment", "chart_focus"):
        assert contrast_ratio(tokens[series], tokens["surface"]) >= 3.0, (mode, series)


def test_foreground_selection_checks_actual_surface():
    assert accessible_foreground("#FFFFFF") == "#172033"
    assert accessible_foreground("#111827") == "#FFFFFF"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_plotly_theme_uses_opaque_theme_colors(mode):
    tokens = get_theme_tokens(mode)
    fig = apply_plotly_theme(go.Figure(), mode)
    assert fig.layout.paper_bgcolor == tokens["canvas"]
    assert fig.layout.plot_bgcolor == tokens["surface"]
    assert fig.layout.font.color == tokens["text_primary"]
    assert fig.layout.hoverlabel.bgcolor == tokens["surface"]
    assert fig.layout.hoverlabel.font.color == tokens["text_primary"]
    assert fig.layout.xaxis.gridcolor == tokens["grid"]
    assert "--text-primary:" in css_variables(mode)


def test_existing_donut_chart_applies_accessible_default_theme():
    from src.frontend.charts import create_energy_mix_donut_chart

    fig = create_energy_mix_donut_chart()
    tokens = get_theme_tokens("light")
    assert fig.layout.paper_bgcolor == tokens["canvas"]
    assert fig.layout.plot_bgcolor == tokens["surface"]
    assert fig.layout.font.color == tokens["text_primary"]
    pie = fig.data[0]
    assert len(pie.textfont.color) == len(pie.marker.colors)
    assert all(
        contrast_ratio(foreground, background, underlay=tokens["surface"]) >= 4.5
        for foreground, background in zip(pie.textfont.color, pie.marker.colors)
    )


def test_component_css_variables_resolve_to_theme_tokens():
    components = Path("src/frontend/components.py").read_text(encoding="utf-8")
    referenced = set(re.findall(r"var\(--([a-z-]+)\)", components))
    declared = {name.replace("_", "-") for name in get_theme_tokens("light")}
    assert referenced <= declared
