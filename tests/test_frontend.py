"""
Unit and integration tests for the Streamlit frontend and model loader pipelines.

Verifies:
1. Application prerequisites, model checkpoint availability, and safe loading.
2. Real-time inference on Model A (Solar ResNet) with valid and invalid inputs.
3. Model A, B, C, and D evaluation data loading and test trajectories.
4. Conformal uncertainty summaries and table parsing.
5. Interactive energy dispatch simulation and optimizer integration.
6. Prototype chart generators (donut chart, overview solar PV chart, mini forecast chart).
7. Error handling for corrupt or missing inputs.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from PIL import Image

import src.utils.torch_loader  # Ensures Windows DLLs are initialized
import src.frontend.model_loaders as ml
import src.frontend.charts as charts
import src.frontend.components as components


class TestFrontendDataLoaders:
    """Test loading and integrity of registry, summaries, and candidate tables."""

    def test_load_registry_data(self):
        df = ml.load_registry_data()
        assert not df.empty, "Experiment registry should not be empty"
        assert "experiment_id" in df.columns
        assert "task" in df.columns
        assert len(df) >= 48, f"Expected at least 48 registry entries, got {len(df)}"

    def test_load_uncertainty_summary(self):
        summary = ml.load_uncertainty_summary()
        assert isinstance(summary, dict)
        assert "wind" in summary or "coverage_summary" in summary or "timestamp" in summary

    def test_load_decision_support_summary(self):
        summary = ml.load_decision_support_summary()
        assert isinstance(summary, dict)
        assert "framework_title" in summary
        assert "cost_optimal_configuration" in summary
        assert "baseline_100_percent_grid" in summary

    def test_load_feasibility_scenarios_table(self):
        df = ml.load_feasibility_scenarios_table()
        assert not df.empty
        assert "solar_kw" in df.columns
        assert "wind_kw" in df.columns
        assert "battery_kwh" in df.columns
        assert "renewable_penetration" in df.columns
        assert len(df) == 120, f"Expected 120 candidate combinations, got {len(df)}"


class TestModelInferenceIntegration:
    """Test safe inference execution for deep learning models."""

    def test_missing_model_checkpoints_return_unavailable(self, monkeypatch, tmp_path):
        """All model loaders should report unavailable when their required artifacts are absent."""
        monkeypatch.setattr(ml, "PROJECT_ROOT", tmp_path)
        loaders = (
            ml.load_solar_nowcaster,
            ml.load_weather_forecaster,
            ml.load_wind_forecaster,
            ml.load_demand_forecaster,
        )
        evaluators = (
            ml.get_solar_nowcasting_test_evaluation,
            ml.get_weather_test_evaluation,
            ml.get_wind_test_evaluation,
            ml.get_demand_test_evaluation,
        )
        for loader in loaders:
            loader.clear()
        for evaluator in evaluators:
            evaluator.clear()
        try:
            for loader in loaders:
                assert loader() is None
            for evaluator in evaluators:
                assert evaluator(max_samples=1) == {}
        finally:
            for loader in loaders:
                loader.clear()
            for evaluator in evaluators:
                evaluator.clear()

    def test_solar_nowcaster_inference_on_sample_images(self):
        sample_dir = Path("data/sample_images")
        if not sample_dir.exists():
            pytest.skip("Sample images directory not found")

        samples = list(sample_dir.glob("*.png"))
        assert len(samples) > 0, "No sample sky images found"

        for sample_path in samples:
            img = Image.open(sample_path)
            pv_pred = ml.predict_solar_pv_from_image(img)
            assert isinstance(pv_pred, float)
            assert pv_pred >= 0.0, f"Predicted PV power must be non-negative, got {pv_pred}"
            assert np.isfinite(pv_pred), "Prediction must be finite"

    def test_solar_nowcaster_invalid_input_handling(self):
        with pytest.raises((TypeError, RuntimeError)):
            ml.predict_solar_pv_from_image(12345)

    def test_corrupt_image_upload_has_actionable_validation_error(self, monkeypatch):
        """Corrupt uploaded bytes are rejected before model inference with a readable message."""
        monkeypatch.setattr(ml, "load_solar_nowcaster", lambda: object())
        with pytest.raises(ValueError, match="valid PNG or JPEG"):
            ml.predict_solar_pv_from_image(b"not an image")

    def test_too_small_image_upload_has_actionable_validation_error(self, monkeypatch):
        """Very small uploaded images are rejected with the supported minimum dimensions."""
        monkeypatch.setattr(ml, "load_solar_nowcaster", lambda: object())
        image = Image.new("RGB", (16, 16))
        with pytest.raises(ValueError, match="at least 32 x 32"):
            ml.predict_solar_pv_from_image(image)

    def test_solar_nowcasting_test_evaluation(self):
        solar_eval = ml.get_solar_nowcasting_test_evaluation(max_samples=50)
        assert "timestamps" in solar_eval
        assert "y_true" in solar_eval
        assert "y_pred" in solar_eval
        assert len(solar_eval["y_true"]) == 50
        assert len(solar_eval["y_pred"]) == 50
        assert solar_eval["unit"] == "kW"
        assert solar_eval["mae"] >= 0.0

    def test_weather_evaluation_series(self):
        eval_b = ml.get_weather_test_evaluation(max_samples=24)
        assert "y_true" in eval_b
        assert "y_pred" in eval_b
        assert len(eval_b["y_true"]) == 24
        assert len(eval_b["y_pred"]) == 24
        assert eval_b["unit"] == "Wh/m²"
        assert eval_b["mae"] > 0.0
        assert "projected_mae" in eval_b
        assert "projected_r2" in eval_b
        assert eval_b["changed_prediction_count"] == eval_b["invalid_prediction_count"]
        assert eval_b["invalid_prediction_fraction"] == pytest.approx(eval_b["invalid_prediction_count"] / 24)
        assert np.isclose(eval_b["mae"], np.mean(np.abs(eval_b["y_true"] - eval_b["raw_y_pred"])), atol=0.01)
        assert np.isclose(eval_b["projected_mae"], np.mean(np.abs(eval_b["y_true"] - eval_b["y_pred"])), atol=0.01)
        assert np.isclose(eval_b["projected_r2"], 1 - np.sum((eval_b["y_true"] - eval_b["y_pred"]) ** 2) / np.sum((eval_b["y_true"] - np.mean(eval_b["y_true"])) ** 2), atol=1e-4)

    def test_wind_evaluation_series_and_uncertainty(self):
        eval_c = ml.get_wind_test_evaluation(max_samples=24)
        assert "y_true" in eval_c
        assert "y_pred" in eval_c
        assert "y_low_90" in eval_c
        assert "y_upp_90" in eval_c
        assert "p_pred_kw" in eval_c
        assert np.all(eval_c["y_low_90"] <= eval_c["y_upp_90"])
        assert np.all(eval_c["y_low_90"] >= 0.0), "Wind lower bounds must be clipped >= 0"
        assert eval_c["q_90"] == pytest.approx(1.5316, abs=1e-4)
        assert eval_c["interval_metrics_90"]["n_samples"] == 24
        assert eval_c["interval_metrics_90"]["mpiw"] == pytest.approx(2 * eval_c["q_90"], abs=1e-3)

    def test_demand_evaluation_series_and_uncertainty(self):
        eval_d = ml.get_demand_test_evaluation(max_samples=24)
        assert "y_true" in eval_d
        assert "y_pred" in eval_d
        assert "y_low_90" in eval_d
        assert "y_upp_90" in eval_d
        assert np.all(eval_d["y_low_90"] <= eval_d["y_upp_90"])
        assert np.all(eval_d["y_low_90"] >= 0.0), "Demand lower bounds must be clipped >= 0"
        assert eval_d["interval_method"] == "prequential rolling conformal, 168-hour residual window"
        assert eval_d["fixed_interval_metrics_90"]["n_samples"] == 24
        assert eval_d["rolling_interval_metrics_90"]["n_samples"] == 24
        assert eval_d["rolling_valid_count"] == 24
        assert np.all(eval_d["y_low_90_fixed"] <= eval_d["y_upp_90_fixed"])

    def test_demand_full_test_baseline_and_rolling_metrics_match_series(self):
        from src.uncertainty.conformal import calculate_interval_metrics

        evaluation = ml.get_demand_test_evaluation(max_samples=10000)
        n = len(evaluation["y_true"])
        assert n == 2561
        assert evaluation["fixed_interval_metrics_90"]["coverage"] == pytest.approx(0.8547, abs=1e-4)
        assert evaluation["fixed_interval_metrics_80"]["coverage"] == pytest.approx(0.6603, abs=1e-4)
        rolling = evaluation["rolling_interval_metrics_90"]
        recomputed = calculate_interval_metrics(
            evaluation["y_true"], evaluation["y_low_90"], evaluation["y_upp_90"], alpha=0.10
        )
        assert rolling == recomputed
        assert rolling["n_samples"] == evaluation["rolling_valid_count"] == n
        assert evaluation["rolling_skipped_count"] == 0
        assert evaluation["rolling_window_hours"] == 168
        assert np.all(evaluation["y_low_90"] <= evaluation["y_upp_90"])


class TestInteractiveSimulatorIntegration:
    """Test interactive simulation and optimizer calls from the frontend."""

    def test_custom_simulation_execution(self):
        dispatch, metrics, costs = ml.simulate_custom_system(
            solar_kw=50.0,
            wind_kw=100.0,
            battery_kwh=50.0,
            grid_import_limit_kw=100.0,
            duration_hours=48,
        )
        assert len(dispatch.demand) == 48
        assert metrics["renewable_penetration"] >= 0.0
        assert metrics["renewable_penetration"] <= 1.0
        assert costs["total_cost"] > 0.0

        # Physical conservation assertion
        np.testing.assert_allclose(
            dispatch.served_demand + dispatch.unmet_demand,
            dispatch.demand,
            atol=1e-5,
        )

    def test_benchmark_time_series_availability(self):
        ts_data = ml.get_benchmark_time_series(duration_hours=48)
        assert "solar_irradiance" in ts_data
        assert "wind_speed" in ts_data
        assert "demand" in ts_data
        assert len(ts_data["solar_irradiance"]) == 48
        assert len(ts_data["wind_speed"]) == 48
        assert len(ts_data["demand"]) == 48


class TestPlotlyChartGenerators:
    """Verify that chart generator functions return valid plotly Figures without error."""

    def test_create_actual_vs_predicted_chart(self):
        timestamps = [f"2023-01-01 {h:02d}:00" for h in range(10)]
        y_true = np.ones(10) * 5.0
        y_pred = np.ones(10) * 4.8
        y_low = np.ones(10) * 3.5
        y_upp = np.ones(10) * 6.5

        fig = charts.create_actual_vs_predicted_chart(
            timestamps=timestamps,
            y_true=y_true,
            y_pred=y_pred,
            y_low=y_low,
            y_upp=y_upp,
            unit="kW",
        )
        assert fig is not None
        assert len(fig.data) >= 3

    def test_create_energy_mix_donut_chart(self):
        fig = charts.create_energy_mix_donut_chart(
            solar_val=150.0,
            wind_val=100.0,
            battery_val=50.0,
            grid_val=50.0,
            total_label="350 kW",
        )
        assert fig is not None
        assert len(fig.data) >= 1

    def test_create_overview_solar_pv_chart(self):
        timestamps = [f"07:{m:02d}" for m in range(30)]
        y_true = np.linspace(0.5, 20.0, 30)
        y_pred = np.linspace(0.4, 19.5, 30)

        fig = charts.create_overview_solar_pv_chart(
            timestamps=timestamps,
            y_true=y_true,
            y_pred=y_pred,
        )
        assert fig is not None
        assert len(fig.data) == 2

    def test_create_mini_forecast_chart(self):
        timestamps = [f"Oct {d}" for d in range(10)]
        y_true = np.ones(10) * 8.0
        y_pred = np.ones(10) * 7.5
        y_low = np.ones(10) * 6.0
        y_upp = np.ones(10) * 9.5

        fig = charts.create_mini_forecast_chart(
            timestamps=timestamps,
            y_true=y_true,
            y_pred=y_pred,
            y_low=y_low,
            y_upp=y_upp,
            unit="m/s",
            theme="green",
        )
        assert fig is not None
        assert len(fig.data) >= 3

    def test_create_tradeoff_pareto_chart(self):
        df = pd.DataFrame({
            "solar_kw": [0.0, 50.0, 100.0],
            "wind_kw": [0.0, 100.0, 200.0],
            "battery_kwh": [0.0, 50.0, 100.0],
            "renewable_penetration": [0.0, 0.35, 0.65],
            "grid_import_kwh": [1000.0, 650.0, 350.0],
            "total_cost": [120.0, 110.0, 135.0],
        })
        fig = charts.create_tradeoff_pareto_chart(df)
        assert fig is not None
        assert len(fig.data) >= 2

    def test_create_dispatch_dashboard_chart(self):
        timestamps = [f"2023-01-01 {h:02d}:00" for h in range(10)]
        demand = np.ones(10) * 50.0
        ren_used = np.ones(10) * 30.0
        batt_dis = np.ones(10) * 10.0
        grid_imp = np.ones(10) * 10.0
        curtail = np.zeros(10)
        soc = np.ones(10) * 25.0

        fig = charts.create_dispatch_dashboard_chart(
            timestamps=timestamps,
            demand=demand,
            renewable_used=ren_used,
            battery_discharge=batt_dis,
            grid_import=grid_imp,
            curtailment=curtail,
            battery_soc=soc,
            unmet_demand=np.zeros(10),
        )
        assert fig is not None
        assert len(fig.data) >= 5
        assert any(trace.name == "Unmet Demand" for trace in fig.data)

    def test_app_file_integrity_and_imports(self):
        app_path = Path("app.py")
        assert app_path.exists(), "app.py must exist at project root"
        code = app_path.read_text(encoding="utf-8")
        assert "A Multimodal, Uncertainty-Aware Decision-Support Framework" in code
        assert "st.set_page_config" in code
        assert "sidebar" in code
        # Test compiling app.py code
        compiled = compile(code, str(app_path), "exec")
        assert compiled is not None


def test_prediction_domain_helpers_project_and_reject_invalid_values():
    projected, count = ml._project_nonnegative_predictions(
        np.array([-1.0, 0.0, 2.0]), "test target"
    )
    np.testing.assert_array_equal(projected, [0.0, 0.0, 2.0])
    assert count == 1
    with pytest.raises(ValueError, match="NaN or infinite"):
        ml._project_nonnegative_predictions(np.array([float("nan")]), "test target")
    with pytest.raises(ValueError, match="materially negative"):
        ml._validate_nonnegative_targets(np.array([-0.1]), "test target")
    np.testing.assert_array_equal(
        ml._validate_nonnegative_targets(np.array([-1e-5, 1.0]), "test target"),
        [0.0, 1.0],
    )


def test_custom_simulation_cost_includes_initial_battery_energy_value():
    dispatch, metrics, costs = ml.simulate_custom_system(
        solar_kw=0.0, wind_kw=0.0, battery_kwh=50.0,
        grid_import_limit_kw=100.0, duration_hours=24,
    )
    assert metrics["solar_generation_available_kwh"] == 0.0
    assert metrics["wind_generation_available_kwh"] == 0.0
    if metrics["initial_soc_carry_in_discharge_kwh"] > 0:
        assert costs["carry_in_energy_cost"] > 0.0
