"""Regression tests for user-facing explanations derived from model outputs."""
import pytest

from src.frontend.explanations import explain_coverage, explain_forecast, explain_simulation


def test_forecast_explanation_summarizes_values_and_labels_historical_replay():
    result = explain_forecast(
        "Solar irradiance", [10.0, 20.0, 30.0], "Wh/m²", "3 displayed hours",
        "NASA POWER test partition", historical_mae=4.0, mae_unit="Wh/m²",
    )
    assert "averages 20.00 Wh/m²" in result["result"]
    assert "highest displayed value is 30.00 Wh/m²" in result["result"]
    assert "not a live reading" in result["meaning"]
    assert "4.00 Wh/m²" in result["reliability"]


def test_coverage_explanation_reports_count_dataset_and_target_status():
    result = explain_coverage(0.90, 0.8547, 2561, "BDG2 test partition")
    assert result["sample_count"] == 2561
    assert result["dataset"] == "BDG2 test partition"
    assert result["inside_count"] == round(0.8547 * 2561)
    assert result["met_target"] is False
    assert "BDG2 test partition" in result["plain_text"]
    assert "does not guarantee future coverage" in result["plain_text"]


def test_simulation_explanation_changes_for_shortfall_and_compares_baseline():
    metrics = {
        "total_demand_kwh": 100.0, "grid_import_kwh": 20.0,
        "unmet_demand_kwh": 5.0, "battery_discharge_kwh": 10.0,
        "curtailment_kwh": 2.0, "renewable_penetration": 0.75,
    }
    baseline = {"grid_import_kwh": 100.0, "unmet_demand_kwh": 0.0}
    result = explain_simulation(
        metrics, {"total_cost": 50.0}, 24, baseline_metrics=baseline,
        baseline_costs={"total_cost": 80.0},
    )
    assert "leaves 5.00 kWh" in result["supply"]
    assert "imports 20.00 kWh" in result["supply"]
    assert "grid imports changed by -80.00 kWh" in result["comparison"]
    assert "unmet demand changed by +5.00 kWh" in result["comparison"]
    assert "not a site forecast" in result["reliability"]


def test_coverage_rejects_invalid_fraction_or_sample_count():
    with pytest.raises(ValueError):
        explain_coverage(1.1, 0.8, 100, "dataset")
    with pytest.raises(ValueError):
        explain_coverage(0.9, 0.8, 0, "dataset")
