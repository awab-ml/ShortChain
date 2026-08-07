"""Tests for calibration (ECE + Platt/isotonic scalers)."""

from __future__ import annotations

import numpy as np
import pytest

from shortchain.evaluation.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    create_calibrator,
    expected_calibration_error,
    reliability_data,
)


class TestECE:
    def test_perfectly_calibrated(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=2000)
        # sample p then flip with prob (1-p) so P(y=1)=p by construction
        p = rng.uniform(0, 1, size=2000)
        y = (rng.uniform(0, 1, size=2000) < p).astype(int)
        assert expected_calibration_error(y, p, n_bins=10) < 0.05

    def test_miscalibrated(self):
        # scores anti-correlated with truth -> high ECE
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=1000)
        scores = np.where(y == 1, 0.1, 0.9)
        assert expected_calibration_error(y, scores, n_bins=10) > 0.3

    def test_reliability_shapes(self):
        bins = reliability_data(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 10)
        assert len(bins[0]) == 10 and len(bins[1]) == 10 and len(bins[2]) == 10


class TestCalibrators:
    def _miscalibrated(self, n=4000):
        rng = np.random.default_rng(42)
        z = rng.normal(size=n)
        p_true = 1 / (1 + np.exp(-z))
        y = (rng.uniform(size=n) < p_true).astype(int)
        # model view: underconfident scores
        p_raw = np.clip(p_true - 0.3 + 0.1 * rng.normal(size=n), 0, 1)
        return y, p_raw, p_true

    def test_platt_improves_ece(self):
        y, p_raw, _ = self._miscalibrated()
        before = expected_calibration_error(y, p_raw, n_bins=10)
        cal = PlattCalibrator().fit(p_raw[:2000], y[:2000])
        after = expected_calibration_error(y[2000:], cal.transform(p_raw[2000:]), n_bins=10)
        assert after < before

    def test_isotonic_improves_ece(self):
        y, p_raw, _ = self._miscalibrated()
        before = expected_calibration_error(y, p_raw, n_bins=10)
        cal = IsotonicCalibrator().fit(p_raw[:2000], y[:2000])
        after = expected_calibration_error(y[2000:], cal.transform(p_raw[2000:]), n_bins=10)
        assert after < before

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            PlattCalibrator().transform(np.array([0.5]))
        with pytest.raises(RuntimeError):
            IsotonicCalibrator().transform(np.array([0.5]))

    def test_create_calibrator(self):
        assert isinstance(create_calibrator("platt"), PlattCalibrator)
        assert isinstance(create_calibrator("isotonic"), IsotonicCalibrator)
        with pytest.raises(ValueError):
            create_calibrator("bogus")
