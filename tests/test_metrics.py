"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shortchain.evaluation.metrics import (
    compute_metrics,
    r_precision,
    recall_at_k,
    format_metrics,
)


# ---------------------------------------------------------------------------
# R-precision tests
# ---------------------------------------------------------------------------

class TestRPrecision:
    def test_perfect_ranking(self):
        """When top-R predictions are all relevant, P@R = 1.0."""
        y_true = np.array([1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        task_ids = np.array(["t1"] * 5)
        tool_names = np.array(["a", "b", "c", "d", "e"])

        assert r_precision(y_true, y_scores, task_ids, tool_names) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """When top-R predictions are all irrelevant, P@R = 0.0."""
        y_true = np.array([0, 0, 1, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        task_ids = np.array(["t1"] * 5)
        tool_names = np.array(["a", "b", "c", "d", "e"])

        assert r_precision(y_true, y_scores, task_ids, tool_names) == pytest.approx(0.0)

    def test_multi_task_averaging(self):
        """Macro-average over two tasks."""
        y_true = np.array([1, 0, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.1, 0.05, 0.1, 0.9, 0.5])
        task_ids = np.array(["t1", "t1", "t1", "t2", "t2", "t2"])
        tool_names = np.array(["a", "b", "c", "d", "e", "f"])

        # t1: R=1, top-1=[a] → P@R=1.0
        # t2: R=1, top-1=[e] → P@R=0.0
        # macro avg = 0.5
        assert r_precision(y_true, y_scores, task_ids, tool_names) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Recall@k tests
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_recall_at_k_perfect(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        task_ids = np.array(["t1"] * 5)

        assert recall_at_k(y_true, y_scores, task_ids, k=2) == pytest.approx(1.0)

    def test_recall_at_k_partial(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.3, 0.1, 0.8, 0.5])
        task_ids = np.array(["t1"] * 5)

        # top-2: indices [0, 3] → hits: [1, 0] → recall = 1/3
        assert recall_at_k(y_true, y_scores, task_ids, k=2) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_standard_metrics(self):
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.8, 0.3, 0.6, 0.2])
        metrics = compute_metrics(y_true, y_proba)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_with_task_grouping(self):
        y_true = np.array([1, 0, 0, 1, 0])
        y_proba = np.array([0.9, 0.2, 0.1, 0.8, 0.3])
        X_val = pd.DataFrame({
            "task_id": ["t1", "t1", "t1", "t2", "t2"],
            "tool_name": ["a", "b", "c", "d", "e"],
        })
        metrics = compute_metrics(y_true, y_proba, X_val=X_val, k_values=[3, 5])
        assert "r_precision" in metrics
        assert "recall_at_3" in metrics
        assert "recall_at_5" in metrics


# ---------------------------------------------------------------------------
# format_metrics tests
# ---------------------------------------------------------------------------

class TestFormatMetrics:
    def test_format_output(self):
        metrics = {"accuracy": 0.85, "f1": 0.82}
        output = format_metrics(metrics)
        assert "accuracy" in output
        assert "0.8500" in output
