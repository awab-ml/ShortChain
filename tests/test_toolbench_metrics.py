"""Tests for ToolBench-specific metrics (pass_rate, metrics_by_group)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabagent.evaluation.metrics import pass_rate, metrics_by_group


class TestPassRate:
    """Tests for pass_rate()."""

    def test_perfect_pass_rate(self) -> None:
        """All tasks have a hit in top-k → pass_rate = 1.0."""
        y_true = np.array([1, 0, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.3, 0.2, 0.8, 0.4, 0.1])
        task_ids = np.array(["t1", "t1", "t1", "t2", "t2", "t2"])

        result = pass_rate(y_true, y_scores, task_ids, k=2)
        assert result == 1.0

    def test_zero_pass_rate(self) -> None:
        """No tasks have a hit in top-k → pass_rate = 0.0."""
        y_true = np.array([1, 0, 0, 1, 0, 0])
        # Relevant items have lowest scores
        y_scores = np.array([0.1, 0.9, 0.8, 0.1, 0.9, 0.8])
        task_ids = np.array(["t1", "t1", "t1", "t2", "t2", "t2"])

        result = pass_rate(y_true, y_scores, task_ids, k=1)
        assert result == 0.0

    def test_partial_pass_rate(self) -> None:
        """Half the tasks have hits → pass_rate = 0.5."""
        y_true = np.array([1, 0, 0, 1, 0, 0])
        # Task 1: relevant item has highest score
        # Task 2: relevant item has lowest score
        y_scores = np.array([0.9, 0.3, 0.2, 0.1, 0.9, 0.8])
        task_ids = np.array(["t1", "t1", "t1", "t2", "t2", "t2"])

        result = pass_rate(y_true, y_scores, task_ids, k=1)
        assert result == 0.5

    def test_empty_returns_zero(self) -> None:
        """Empty arrays → 0.0."""
        result = pass_rate(
            np.array([]), np.array([]), np.array([]), k=5
        )
        assert result == 0.0

    def test_k_larger_than_candidates(self) -> None:
        """k > number of candidates should still work."""
        y_true = np.array([1, 0])
        y_scores = np.array([0.9, 0.1])
        task_ids = np.array(["t1", "t1"])

        result = pass_rate(y_true, y_scores, task_ids, k=10)
        assert result == 1.0


class TestMetricsByGroup:
    """Tests for metrics_by_group()."""

    def test_returns_dataframe(self) -> None:
        """Should return a DataFrame with one row per group."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_proba = np.array([0.9, 0.3, 0.8, 0.2, 0.7, 0.4])
        X_val = pd.DataFrame({
            "task_id": ["t1", "t1", "t2", "t2", "t3", "t3"],
            "tool_name": ["a", "b", "c", "d", "e", "f"],
            "app_name": ["Weather", "Weather", "Finance", "Finance", "Finance", "Finance"],
        })

        result = metrics_by_group(y_true, y_proba, X_val, group_col="app_name")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Weather and Finance
        assert "app_name" in result.columns
        assert "accuracy" in result.columns
        assert "n_samples" in result.columns

    def test_missing_group_col(self) -> None:
        """Should return empty DataFrame if group column doesn't exist."""
        y_true = np.array([1, 0])
        y_proba = np.array([0.9, 0.1])
        X_val = pd.DataFrame({"task_id": ["t1", "t1"]})

        result = metrics_by_group(y_true, y_proba, X_val, group_col="nonexistent")
        assert len(result) == 0
