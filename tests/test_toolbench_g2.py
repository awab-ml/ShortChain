"""Tests for step-level expansion and G2/G3 scenario classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from tabagent.ingest.schema import Step, Trajectory
from tabagent.ingest.toolbench_catalog import ToolBenchCatalog
from tabagent.ingest.toolbench_loader import ToolBenchLoader

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "toolbench" / "fixtures"
SAMPLE_PREPROCESSED = FIXTURES / "sample_preprocessed.json"
SAMPLE_TOOLENV = FIXTURES / "sample_toolenv"


# ------------------------------------------------------------------
# Step-level expansion
# ------------------------------------------------------------------


class TestExpandToStepTrajectories:
    """Tests for ToolBenchLoader.expand_to_step_trajectories()."""

    def _make_traj(self, n_steps: int, task_id: str = "t1") -> Trajectory:
        """Create a trajectory with n_steps named step_0, step_1, ..."""
        steps = [
            Step(
                agent_name="test",
                action=f"tool_{i}",
                observation=f"result_{i}",
                thoughts=f"thinking about step {i}",
            )
            for i in range(n_steps)
        ]
        return Trajectory(
            task_id=task_id,
            intent="do something",
            steps=steps,
            success=True,
            app_name="TestApp",
            metadata={"n_system_tools": 2, "system_tool_names": ["a", "b"]},
        )

    def test_single_step_returns_original(self) -> None:
        """A 1-step trajectory should return itself unchanged."""
        traj = self._make_traj(1)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)
        assert len(result) == 1
        assert result[0].task_id == "t1"

    def test_multi_step_count(self) -> None:
        """A 3-step trajectory should produce 3 sub-trajectories."""
        traj = self._make_traj(3)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)
        assert len(result) == 3

    def test_sub_traj_task_ids(self) -> None:
        """Sub-trajectories should have unique step-indexed task_ids."""
        traj = self._make_traj(3, task_id="parent")
        result = ToolBenchLoader.expand_to_step_trajectories(traj)
        assert result[0].task_id == "parent_s0"
        assert result[1].task_id == "parent_s1"
        assert result[2].task_id == "parent_s2"

    def test_sub_traj_tools_used(self) -> None:
        """Each sub-trajectory should have exactly one tool in tools_used."""
        traj = self._make_traj(3)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)
        assert result[0].tools_used == {"tool_0"}
        assert result[1].tools_used == {"tool_1"}
        assert result[2].tools_used == {"tool_2"}

    def test_sub_traj_prior_steps(self) -> None:
        """Each sub-trajectory should contain the correct prior steps."""
        traj = self._make_traj(3)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)

        # Step 0: no prior steps
        assert len(result[0].steps) == 0
        # Step 1: 1 prior step (tool_0)
        assert len(result[1].steps) == 1
        assert result[1].steps[0].tool_name == "tool_0"
        # Step 2: 2 prior steps (tool_0, tool_1)
        assert len(result[2].steps) == 2
        assert result[2].steps[0].tool_name == "tool_0"
        assert result[2].steps[1].tool_name == "tool_1"

    def test_sub_traj_preserves_intent(self) -> None:
        """All sub-trajectories should share the same intent."""
        traj = self._make_traj(3)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)
        for sub in result:
            assert sub.intent == "do something"

    def test_sub_traj_preserves_app_name(self) -> None:
        """All sub-trajectories should share the same app_name."""
        traj = self._make_traj(3)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)
        for sub in result:
            assert sub.app_name == "TestApp"

    def test_sub_traj_metadata(self) -> None:
        """Sub-trajectories should have step metadata."""
        traj = self._make_traj(3)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)
        for i, sub in enumerate(result):
            assert sub.metadata["step_index"] == i
            assert sub.metadata["parent_task_id"] == traj.task_id
            assert sub.metadata["total_steps"] == 3

    def test_previous_tools_feature(self) -> None:
        """Sub-trajectories should produce correct previous_tools context."""
        traj = self._make_traj(3)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)

        # Step 0: no previous tools
        assert result[0].tool_sequence == []
        # Step 1: [tool_0]
        assert result[1].tool_sequence == ["tool_0"]
        # Step 2: [tool_0, tool_1]
        assert result[2].tool_sequence == ["tool_0", "tool_1"]

    def test_last_thought_context(self) -> None:
        """Sub-trajectories should have correct last_thought."""
        traj = self._make_traj(3)
        result = ToolBenchLoader.expand_to_step_trajectories(traj)

        # Step 0: no prior steps → no last_thought
        assert result[0].last_thought is None
        # Step 1: last_thought from step 0
        assert result[1].last_thought == "thinking about step 0"
        # Step 2: last_thought from step 1
        assert result[2].last_thought == "thinking about step 1"

    def test_integration_with_dataset_builder(self) -> None:
        """Expanded step trajectories should work with DatasetBuilder."""
        from tabagent.dataset.builder import DatasetBuilder

        traj = self._make_traj(3)
        expanded = ToolBenchLoader.expand_to_step_trajectories(traj)

        builder = DatasetBuilder()
        df = builder.build(expanded)

        assert len(df) > 0
        assert "label" in df.columns
        # Should have 3 positive examples (one per step)
        assert df["label"].sum() == 3
        # All 3 sub-trajectories have unique task_ids
        assert df["task_id"].nunique() == 3


# ------------------------------------------------------------------
# Scenario classification
# ------------------------------------------------------------------


class TestClassifyScenario:
    """Tests for ToolBenchLoader._classify_scenario()."""

    @pytest.fixture()
    def catalog(self) -> ToolBenchCatalog:
        return ToolBenchCatalog.from_toolenv(SAMPLE_TOOLENV)

    @pytest.fixture()
    def loader(self, catalog: ToolBenchCatalog) -> ToolBenchLoader:
        return ToolBenchLoader(catalog=catalog, success_only=False)

    def test_g1_single_tool(self, loader: ToolBenchLoader) -> None:
        """Single tool in system prompt → G1."""
        traj = Trajectory(
            task_id="t1", intent="test", steps=[],
            metadata={"n_system_tools": 1, "system_tool_names": ["open_weather"]},
        )
        assert loader._classify_scenario(traj) == "G1"

    def test_g2_same_category(self, loader: ToolBenchLoader) -> None:
        """Multiple tools from same category → G2.

        Both open_weather and (a hypothetical second weather tool) would
        be same category. Since our fixtures only have one tool per
        category, we test with tools that can't be resolved (defaults to G2).
        """
        traj = Trajectory(
            task_id="t1", intent="test", steps=[],
            metadata={"n_system_tools": 3, "system_tool_names": ["a", "b", "c"]},
        )
        # No tools found in catalog → defaults to G2 (can't prove G3)
        assert loader._classify_scenario(traj) == "G2"

    def test_g3_cross_category(self, loader: ToolBenchLoader) -> None:
        """Tools from different categories → G3."""
        traj = Trajectory(
            task_id="t1", intent="test", steps=[],
            metadata={
                "n_system_tools": 2,
                "system_tool_names": ["open_weather", "stock_tracker"],
            },
        )
        assert loader._classify_scenario(traj) == "G3"

    def test_no_catalog_defaults_g2(self) -> None:
        """Without catalog, multi-tool defaults to G2."""
        loader = ToolBenchLoader(catalog=None, success_only=False)
        traj = Trajectory(
            task_id="t1", intent="test", steps=[],
            metadata={"n_system_tools": 3, "system_tool_names": ["a", "b", "c"]},
        )
        assert loader._classify_scenario(traj) == "G2"


# ------------------------------------------------------------------
# Integration: step-level with fixtures
# ------------------------------------------------------------------


class TestStepLevelWithFixtures:
    """Tests for step-level loading with fixture data."""

    @pytest.fixture()
    def catalog(self) -> ToolBenchCatalog:
        return ToolBenchCatalog.from_toolenv(SAMPLE_TOOLENV)

    @pytest.fixture()
    def loader(self, catalog: ToolBenchCatalog) -> ToolBenchLoader:
        return ToolBenchLoader(catalog=catalog, success_only=False)

    def test_step_level_flag_expands(self, loader: ToolBenchLoader) -> None:
        """step_level=True should produce more trajectories than False."""
        normal = loader.load_with_filter(SAMPLE_PREPROCESSED, scenario="all")
        expanded = loader.load_with_filter(
            SAMPLE_PREPROCESSED, scenario="all", step_level=True
        )
        # Multi-step trajectories should expand
        assert len(expanded) >= len(normal)

    def test_step_level_g1_single_step(self, loader: ToolBenchLoader) -> None:
        """G1 single-step trajectories shouldn't expand."""
        normal = loader.load_with_filter(SAMPLE_PREPROCESSED, scenario="G1")
        expanded = loader.load_with_filter(
            SAMPLE_PREPROCESSED, scenario="G1", step_level=True
        )
        # G1 fixture instances have 1-2 steps, some will expand
        assert len(expanded) >= len(normal)

    def test_g2_filter_excludes_g1(self, loader: ToolBenchLoader) -> None:
        """G2 filter should not include single-tool trajectories."""
        g2 = loader.load_with_filter(SAMPLE_PREPROCESSED, scenario="G2")
        for t in g2:
            assert t.metadata.get("n_system_tools", 0) >= 2

    def test_g3_filter(self, loader: ToolBenchLoader) -> None:
        """G3 filter should return cross-category trajectories."""
        g3 = loader.load_with_filter(SAMPLE_PREPROCESSED, scenario="G3")
        # Fixture 4 has open_weather + stock_tracker (Weather + Finance)
        # which should be classified as G3
        for t in g3:
            assert t.metadata.get("n_system_tools", 0) >= 2
