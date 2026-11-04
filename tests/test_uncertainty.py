"""
Unit tests for Split Conformal Prediction and Uncertainty Quantification.

Tests cover:
1. Quantile calculation and finite-sample correction formula.
2. Input validation (empty arrays, non-finite values, shape mismatch, invalid alpha).
3. Calibration/test leakage prevention (test data never touches calibrator).
4. Physical non-negative clipping impact on width and coverage.
5. Interval evaluation metrics (empirical coverage, MPIW, Winkler score).
6. State lifecycle (cannot predict before fit).
"""
import pytest
import numpy as np
import torch

from src.uncertainty.conformal import (
    SplitConformalCalibrator,
    calculate_interval_metrics,
    evaluate_conformal_intervals,
    predict_rolling_conformal_intervals,
)


class TestConformalQuantiles:
    """Test conformal quantile calculation and finite-sample indexing."""

    def test_finite_sample_formula_n100(self):
        """Verify k = ceil((n+1)*(1-alpha)) matches exact order statistic."""
        # For n=99, alpha=0.10: (99+1)*0.9 = 90 -> 90th value
        residuals = np.arange(1, 100, dtype=np.float64)  # values 1 to 99
        calibrator = SplitConformalCalibrator(alpha=0.10)
        # Use dummy y_true and y_pred where |y_true - y_pred| = residuals
        y_true = residuals
        y_pred = np.zeros_like(residuals)
        calibrator.fit(y_true, y_pred)

        q_90 = calibrator.compute_quantile(alpha=0.10)
        # 90th order statistic is 90.0
        assert q_90 == 90.0

        # For alpha=0.20 (80% coverage): (99+1)*0.8 = 80 -> 80th value
        q_80 = calibrator.compute_quantile(alpha=0.20)
        assert q_80 == 80.0

    def test_finite_sample_formula_n100_fractional(self):
        """Verify ceiling operation for fractional (n+1)*(1-alpha)."""
        # For n=100, alpha=0.10: (100+1)*0.9 = 90.9 -> ceil is 91
        residuals = np.arange(1, 101, dtype=np.float64)  # 1 to 100
        calibrator = SplitConformalCalibrator(alpha=0.10)
        calibrator.fit(residuals, np.zeros_like(residuals))

        q = calibrator.compute_quantile(alpha=0.10)
        assert q == 91.0

    def test_quantile_monotonicity_with_alpha(self):
        """Smaller alpha (higher confidence) must yield larger quantile q."""
        np.random.seed(42)
        residuals = np.random.exponential(scale=2.0, size=500)
        calibrator = SplitConformalCalibrator().fit(residuals, np.zeros_like(residuals))

        q_95 = calibrator.compute_quantile(alpha=0.05)
        q_90 = calibrator.compute_quantile(alpha=0.10)
        q_80 = calibrator.compute_quantile(alpha=0.20)
        q_50 = calibrator.compute_quantile(alpha=0.50)

        assert q_95 >= q_90 >= q_80 >= q_50

    def test_extreme_small_sample_size(self):
        """Small sample size clamps k to n without crash."""
        calibrator = SplitConformalCalibrator(alpha=0.01)  # 99% nominal
        calibrator.fit([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])  # n=3
        # (3+1)*0.99 = 3.96 -> ceil is 4 > n=3, clamped to 3
        q = calibrator.compute_quantile(alpha=0.01)
        assert q == 3.0


class TestConformalValidation:
    """Test input validation, edge cases, and error handling."""

    def test_invalid_alpha(self):
        """Alpha <= 0 or >= 1 must raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
            SplitConformalCalibrator(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
            SplitConformalCalibrator(alpha=1.0)

        with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
            SplitConformalCalibrator(alpha=-0.1)

        cal = SplitConformalCalibrator(alpha=0.1)
        cal.fit([1.0, 2.0], [1.0, 2.0])
        with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
            cal.compute_quantile(alpha=1.5)

    def test_unfitted_access_raises(self):
        """Calling predict_interval or compute_quantile before fit raises ValueError."""
        cal = SplitConformalCalibrator()
        assert not cal.is_fitted
        with pytest.raises(ValueError, match="not fitted"):
            cal.compute_quantile()
        with pytest.raises(ValueError, match="not fitted"):
            cal.predict_interval([1.0, 2.0])
        with pytest.raises(ValueError, match="not fitted"):
            _ = cal.residuals

    def test_shape_mismatch_fit_raises(self):
        """Mismatched y_true and y_pred in fit raises ValueError."""
        cal = SplitConformalCalibrator()
        with pytest.raises(ValueError, match="Shape mismatch"):
            cal.fit([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_empty_array_raises(self):
        """Empty array inputs raise ValueError."""
        cal = SplitConformalCalibrator()
        with pytest.raises(ValueError, match="cannot be empty"):
            cal.fit([], [])

    def test_non_finite_values_raise(self):
        """NaN or Inf inputs raise ValueError."""
        cal = SplitConformalCalibrator()
        with pytest.raises(ValueError, match="non-finite"):
            cal.fit([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="non-finite"):
            cal.fit([1.0, 2.0, 3.0], [1.0, np.inf, 3.0])

    def test_multidimensional_array_raises(self):
        """2D matrices that cannot be squeezed to 1D raise ValueError."""
        cal = SplitConformalCalibrator()
        mat2d = np.ones((5, 5))
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            cal.fit(mat2d, mat2d)

    def test_torch_tensor_compatibility(self):
        """PyTorch tensors should be transparently converted and processed."""
        cal = SplitConformalCalibrator(alpha=0.10)
        t_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        t_pred = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.0])
        cal.fit(t_true, t_pred)
        assert cal.is_fitted
        assert cal.n_calibration_samples == 5

        t_test = torch.tensor([2.5, 3.5])
        lower, upper = cal.predict_interval(t_test)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert len(lower) == 2


class TestLeakageAndPartitionIntegrity:
    """Test that calibration and test partitions remain strictly isolated."""

    def test_zero_test_leakage(self):
        """Verify calibration calibrator is completely unaffected by test observations."""
        cal_true = np.array([10.0, 12.0, 11.0, 13.0, 10.5])
        cal_pred = np.array([9.8, 12.1, 10.9, 13.2, 10.4])

        test_true = np.array([100.0, 200.0, 300.0])  # Extreme out-of-distribution values
        test_pred = np.array([10.0, 20.0, 30.0])

        calibrator = SplitConformalCalibrator(alpha=0.10)
        calibrator.fit(cal_true, cal_pred)

        # Baseline quantile on calibration only
        q_before = calibrator.compute_quantile()
        residuals_before = calibrator.residuals

        # Predicting test points
        lower, upper = calibrator.predict_interval(test_pred)

        # Calibrator state MUST NOT change
        assert calibrator.compute_quantile() == q_before
        np.testing.assert_array_equal(calibrator.residuals, residuals_before)
        assert calibrator.n_calibration_samples == len(cal_true)


class TestPhysicalClippingAndIntervalMetrics:
    """Test physical clipping behavior and interval evaluation metrics."""

    def test_clipping_preserves_coverage_and_reduces_width(self):
        """Clipping lower bound at 0 for non-negative data preserves coverage and reduces width."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.uniform(0.0, 10.0, size=n)  # Strictly non-negative
        # Add zero-mean noise
        y_pred = y_true + np.random.normal(0, 1.5, size=n)

        # Calibrate on first 500
        calibrator = SplitConformalCalibrator(alpha=0.10).fit(y_true[:500], y_pred[:500])

        # Evaluate on next 500
        eval_res = evaluate_conformal_intervals(
            calibrator=calibrator,
            y_true=y_true[500:],
            y_pred=y_pred[500:],
            alpha=0.10,
            clip_min=0.0,
        )

        raw = eval_res["raw_metrics"]
        clipped = eval_res["clipped_metrics"]

        # Coverage must not decrease because y_true >= 0
        assert clipped["coverage"] >= raw["coverage"]
        # Width must be strictly less than or equal
        assert clipped["mpiw"] < raw["mpiw"]
        assert eval_res["clipping_effect"]["width_reduction"] > 0

    def test_calculate_interval_metrics_inverted_bounds_raises(self):
        """When lower > upper, calculate_interval_metrics must raise ValueError."""
        y_true = np.array([5.0, 6.0])
        lower = np.array([7.0, 8.0])
        upper = np.array([4.0, 5.0])
        with pytest.raises(ValueError, match="lower > upper"):
            calculate_interval_metrics(y_true, lower, upper, alpha=0.10)

    def test_winkler_score_properties(self):
        """Winkler score should be equal to width for perfect coverage, higher for misses."""
        y_true = np.array([5.0, 10.0])
        # Perfect coverage: intervals [4, 6] and [9, 11], width=2
        m_covered = calculate_interval_metrics(y_true, [4.0, 9.0], [6.0, 11.0], alpha=0.10)
        assert m_covered["coverage"] == 1.0
        assert m_covered["mpiw"] == 2.0
        assert m_covered["winkler_score"] == 2.0

        # Undercoverage: y_true=5, interval=[6, 8] (miss by 1.0)
        # Score = (8 - 6) + (2 / 0.10) * (6 - 5) = 2 + 20 = 22
        m_miss = calculate_interval_metrics(np.array([5.0]), np.array([6.0]), np.array([8.0]), alpha=0.10)
        assert m_miss["coverage"] == 0.0
        assert m_miss["winkler_score"] == 22.0


class TestRollingConformalIntervals:
    """Validate ordered rolling updates, warm-up, and missing-value policy."""

    def test_rolling_quantile_uses_only_prior_outcomes(self):
        calibration = np.arange(1.0, 169.0)
        lower, upper, quantiles = predict_rolling_conformal_intervals(
            calibration, np.zeros_like(calibration), [1000.0, 0.0], [0.0, 0.0],
            alpha=0.10, window_size=168, clip_min=0.0,
        )
        # ceil((168 + 1) * 0.9) selects the 153rd ordered residual.
        assert quantiles[0] == 153.0
        assert quantiles[1] == 154.0
        assert lower[0] <= upper[0] and lower[1] <= upper[1]

    def test_rolling_requires_full_warmup_history(self):
        with pytest.raises(ValueError, match="Insufficient calibration history"):
            predict_rolling_conformal_intervals(
                np.ones(167), np.zeros(167), [1.0], [1.0], window_size=168,
            )

    def test_missing_outcomes_do_not_enter_rolling_calibration(self):
        calibration = np.ones(168)
        lower, upper, quantiles = predict_rolling_conformal_intervals(
            calibration, np.zeros_like(calibration), [np.nan, 10.0], [4.0, 4.0],
            window_size=168, clip_min=0.0,
        )
        assert quantiles[0] == quantiles[1] == 1.0
        assert np.all(lower <= upper)

    def test_missing_forecast_is_marked_and_not_used_for_update(self):
        calibration = np.ones(168)
        lower, upper, quantiles = predict_rolling_conformal_intervals(
            calibration, np.zeros_like(calibration), [100.0, 5.0], [np.nan, 4.0],
            window_size=168, clip_min=0.0,
        )
        assert np.isnan(lower[0]) and np.isnan(upper[0]) and np.isnan(quantiles[0])
        assert quantiles[1] == 1.0
        assert lower[1] <= upper[1]

    def test_nonfinite_calibration_is_rejected(self):
        calibration = np.ones(168)
        calibration[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            predict_rolling_conformal_intervals(
                calibration, np.zeros_like(calibration), [1.0], [1.0], window_size=168,
            )
