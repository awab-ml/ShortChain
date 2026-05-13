"""Tests for ThresholdTuner."""

from __future__ import annotations

import numpy as np
import pytest

from tabagent.evaluation.threshold_tuner import ThresholdTuner


class TestThresholdTuner:
    """Tests for threshold sweep logic."""

    def _make_data(self):
        """Create test data with clear threshold sensitivity."""
        # 10 positive, 10 negative
        y_true = np.array([1] * 10 + [0] * 10)
        # Positives: probabilities from 0.25 to 0.70
        # Negatives: probabilities from 0.05 to 0.30
        y_proba = np.array([
            0.70, 0.65, 0.60, 0.55, 0.50,   # 5 clear positives
            0.40, 0.35, 0.30, 0.28, 0.25,   # 5 borderline positives
            0.30, 0.25, 0.20, 0.18, 0.15,   # 5 borderline negatives
            0.10, 0.08, 0.06, 0.05, 0.03,   # 5 clear negatives
        ])
        return y_true, y_proba

    def test_finds_optimal_threshold(self) -> None:
        """Should find a threshold that maximizes F1."""
        y_true, y_proba = self._make_data()
        tuner = ThresholdTuner(target_metric="f1")
        best_t, results = tuner.find_optimal(y_true, y_proba)

        # Best threshold should be < 0.5 because many positives
        # have proba between 0.25-0.40
        assert best_t < 0.50
        assert best_t in results

    def test_returns_all_thresholds(self) -> None:
        """Should return metrics for every threshold tested."""
        y_true, y_proba = self._make_data()
        thresholds = [0.2, 0.3, 0.4, 0.5]
        tuner = ThresholdTuner(thresholds=thresholds)
        _, results = tuner.find_optimal(y_true, y_proba)

        assert set(results.keys()) == set(thresholds)
        for t, metrics in results.items():
            assert "f1" in metrics
            assert "precision" in metrics
            assert "recall" in metrics

    def test_recall_increases_at_lower_threshold(self) -> None:
        """Lower threshold should yield higher recall."""
        y_true, y_proba = self._make_data()
        tuner = ThresholdTuner(thresholds=[0.2, 0.5])
        _, results = tuner.find_optimal(y_true, y_proba)

        assert results[0.2]["recall"] >= results[0.5]["recall"]

    def test_precision_increases_at_higher_threshold(self) -> None:
        """Higher threshold should yield higher precision."""
        y_true, y_proba = self._make_data()
        tuner = ThresholdTuner(thresholds=[0.2, 0.5])
        _, results = tuner.find_optimal(y_true, y_proba)

        assert results[0.5]["precision"] >= results[0.2]["precision"]

    def test_target_metric_recall(self) -> None:
        """When targeting recall, should pick lower threshold."""
        y_true, y_proba = self._make_data()
        tuner_f1 = ThresholdTuner(target_metric="f1")
        tuner_recall = ThresholdTuner(target_metric="recall")

        best_f1, _ = tuner_f1.find_optimal(y_true, y_proba)
        best_recall, _ = tuner_recall.find_optimal(y_true, y_proba)

        # Recall-optimizing should pick equal or lower threshold
        assert best_recall <= best_f1

    def test_empty_results(self) -> None:
        """Empty predictions should return default threshold."""
        y_true = np.array([], dtype=int)
        y_proba = np.array([], dtype=float)
        tuner = ThresholdTuner(thresholds=[0.3, 0.5])
        best_t, results = tuner.find_optimal(y_true, y_proba)

        # Should not crash
        assert isinstance(best_t, float)

    def test_default_thresholds(self) -> None:
        """Default thresholds should cover 0.15 to 0.50."""
        tuner = ThresholdTuner()
        assert 0.15 in tuner.thresholds
        assert 0.50 in tuner.thresholds
        assert len(tuner.thresholds) == 8

    def test_metrics_are_valid_floats(self) -> None:
        """All returned metrics should be valid floats in [0, 1]."""
        y_true, y_proba = self._make_data()
        tuner = ThresholdTuner()
        _, results = tuner.find_optimal(y_true, y_proba)

        for t, metrics in results.items():
            for name, val in metrics.items():
                assert 0.0 <= val <= 1.0, f"t={t}, {name}={val}"
