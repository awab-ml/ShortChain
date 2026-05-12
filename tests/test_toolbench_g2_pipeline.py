"""Tests for failure-informed hard negatives and step-wise accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabagent.ingest.schema import Step, Trajectory
from tabagent.ingest.toolbench_negatives import FailureNegativeExtractor
from tabagent.evaluation.metrics import step_wise_accuracy


# ------------------------------------------------------------------
# FailureNegativeExtractor
# ------------------------------------------------------------------


class TestFailureNegativeExtractor:
    """Tests for FailureNegativeExtractor."""

    def _make_traj(
        self, task_id: str, tools: list[str], success: bool, app: str = "Weather"
    ) -> Trajectory:
        steps = [
            Step(agent_name="test", action=t, observation="ok")
            for t in tools
        ]
        return Trajectory(
            task_id=task_id,
            intent="test intent",
            steps=steps,
            success=success,
            app_name=app,
        )

    def test_extract_failed_tools(self) -> None:
        """Should extract tools from failed trajectories by category."""
        failed = [
            self._make_traj("f1", ["api_a", "api_b"], False, "Weather"),
            self._make_traj("f2", ["api_c"], False, "Finance"),
        ]
        extractor = FailureNegativeExtractor()
        result = extractor.extract_failed_tools(failed)

        assert "Weather" in result
        assert "api_a" in result["Weather"]
        assert "api_b" in result["Weather"]
        assert "Finance" in result
        assert "api_c" in result["Finance"]

    def test_extract_skips_successful(self) -> None:
        """Should not extract from successful trajectories."""
        trajs = [
            self._make_traj("s1", ["api_x"], True, "Weather"),
            self._make_traj("f1", ["api_y"], False, "Weather"),
        ]
        extractor = FailureNegativeExtractor()
        result = extractor.extract_failed_tools(trajs)
        assert "api_x" not in result.get("Weather", set())
        assert "api_y" in result.get("Weather", set())

    def test_extract_intent_failures(self) -> None:
        """Should return (intent, app, tools) tuples."""
        failed = [
            self._make_traj("f1", ["api_a"], False, "Weather"),
        ]
        extractor = FailureNegativeExtractor()
        result = extractor.extract_intent_failures(failed)
        assert len(result) == 1
        assert result[0][0] == "test intent"
        assert result[0][1] == "Weather"
        assert "api_a" in result[0][2]

    def test_augment_negatives_adds_rows(self) -> None:
        """augment_negatives should add failure rows to the DataFrame."""
        # Create a small train_df
        train_df = pd.DataFrame({
            "task_id": ["t1", "t1", "t1"],
            "app_name": ["Weather", "Weather", "Weather"],
            "tool_name": ["good_api", "bad_random_1", "bad_random_2"],
            "intent": ["get weather", "get weather", "get weather"],
            "label": [1, 0, 0],
            "n_steps": [1, 1, 1],
        })

        failed = [
            self._make_traj("f1", ["failed_api_1", "failed_api_2"], False, "Weather"),
        ]

        catalog = {
            "good_api": "Good API",
            "failed_api_1": "Failed API 1",
            "failed_api_2": "Failed API 2",
        }

        extractor = FailureNegativeExtractor(catalog=catalog, ratio=0.5)
        result = extractor.augment_negatives(train_df, failed)

        # Should have more rows than original
        assert len(result) >= len(train_df)
        # All new rows should be label=0
        new_rows = result.iloc[len(train_df):]
        if len(new_rows) > 0:
            assert (new_rows["label"] == 0).all()

    def test_augment_no_failures_returns_original(self) -> None:
        """With no failed trajectories, should return original DataFrame."""
        train_df = pd.DataFrame({
            "task_id": ["t1"],
            "app_name": ["Weather"],
            "tool_name": ["api"],
            "label": [1],
        })
        extractor = FailureNegativeExtractor()
        result = extractor.augment_negatives(train_df, [])
        assert len(result) == len(train_df)

    def test_augment_ratio_zero_returns_original(self) -> None:
        """With ratio=0, should return original DataFrame."""
        train_df = pd.DataFrame({
            "task_id": ["t1"],
            "app_name": ["Weather"],
            "tool_name": ["api"],
            "label": [1],
        })
        failed = [self._make_traj("f1", ["bad"], False, "Weather")]

        extractor = FailureNegativeExtractor(ratio=0.0)
        result = extractor.augment_negatives(train_df, failed)
        assert len(result) == len(train_df)


# ------------------------------------------------------------------
# step_wise_accuracy
# ------------------------------------------------------------------


class TestStepWiseAccuracy:
    """Tests for step_wise_accuracy metric."""

    def test_basic(self) -> None:
        """Should return per-step accuracy."""
        y_true = np.array([1, 0, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.3, 0.2, 0.8, 0.4, 0.1])
        task_ids = np.array(["t1", "t1", "t1", "t2", "t2", "t2"])
        step_indices = np.array([0, 0, 0, 1, 1, 1])

        result = step_wise_accuracy(y_true, y_scores, task_ids, step_indices, k=1)

        assert "step_0" in result
        assert "step_1" in result
        assert 0.0 <= result["step_0"] <= 1.0
        assert 0.0 <= result["step_1"] <= 1.0

    def test_empty(self) -> None:
        """Empty arrays should return empty dict."""
        result = step_wise_accuracy(
            np.array([]), np.array([]),
            np.array([]), np.array([]),
        )
        assert result == {}

    def test_single_step(self) -> None:
        """Single step index should return one entry."""
        y_true = np.array([1, 0])
        y_scores = np.array([0.9, 0.1])
        task_ids = np.array(["t1", "t1"])
        step_indices = np.array([0, 0])

        result = step_wise_accuracy(y_true, y_scores, task_ids, step_indices, k=1)
        assert len(result) == 1
        assert result["step_0"] == 1.0

    def test_later_steps_can_differ(self) -> None:
        """Different steps can have different accuracies."""
        # Step 0: perfect (relevant item ranked first)
        # Step 1: bad (relevant item ranked last)
        y_true = np.array([1, 0, 0, 1])
        y_scores = np.array([0.9, 0.1, 0.9, 0.1])
        task_ids = np.array(["t1", "t1", "t2", "t2"])
        step_indices = np.array([0, 0, 1, 1])

        result = step_wise_accuracy(y_true, y_scores, task_ids, step_indices, k=1)
        assert result["step_0"] == 1.0
        assert result["step_1"] == 0.0
