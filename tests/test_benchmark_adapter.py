"""Tests for the benchmark adapter architecture.

Covers:
- Core ``expand_to_step_trajectories`` transform (Q1: lives in tabagent.data)
- ``BenchmarkAdapter`` protocol compliance
- ``ToolBenchAdapter`` construction and data flow
- Adapter registry (``create_adapter``, ``list_adapters``)
- ``BenchmarkConfig`` in ``TabAgentConfig``
"""

from __future__ import annotations

import tempfile
import json
from pathlib import Path

import pandas as pd
import pytest

from tabagent.benchmarks import create_adapter, list_adapters, ADAPTERS
from tabagent.benchmarks.adapter import BenchmarkAdapter
from tabagent.benchmarks.toolbench import ToolBenchAdapter
from tabagent.config import BenchmarkConfig, TabAgentConfig, load_config
from tabagent.data.transforms import expand_to_step_trajectories
from tabagent.ingest.schema import Step, Trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trajectory() -> Trajectory:
    """A 3-step trajectory for testing."""
    return Trajectory(
        task_id="task_001",
        intent="Find the weather forecast for tomorrow",
        steps=[
            Step(action="search_api(query='weather')", observation="results"),
            Step(action="parse_json(data='{}')", observation="parsed"),
            Step(action="format_response(text='sunny')", observation="done"),
        ],
        success=True,
        app_name="weather_app",
    )


@pytest.fixture
def failed_trajectory() -> Trajectory:
    """A failed trajectory for augmentation tests."""
    return Trajectory(
        task_id="task_002",
        intent="Book a flight",
        steps=[
            Step(action="flight_search(dest='NYC')", observation="error"),
        ],
        success=False,
        app_name="travel_app",
    )


@pytest.fixture
def trajectory_file(sample_trajectory, failed_trajectory, tmp_path) -> Path:
    """Write sample trajectories to a JSONL file."""
    file_path = tmp_path / "trajectories.jsonl"
    records = []
    for traj in [sample_trajectory, failed_trajectory]:
        records.append({
            "task_id": traj.task_id,
            "intent": traj.intent,
            "steps": [
                {
                    "action": s.action,
                    "observation": s.observation,
                    "thoughts": s.thoughts,
                }
                for s in traj.steps
            ],
            "success": traj.success,
            "app_name": traj.app_name,
        })
    with open(file_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return file_path


# ---------------------------------------------------------------------------
# Tests: expand_to_step_trajectories (Q1 — core transform)
# ---------------------------------------------------------------------------

class TestExpandToStepTrajectories:
    """Tests for the core step-expansion transform."""

    def test_expands_correct_count(self, sample_trajectory):
        """Each step with a tool produces one sub-trajectory."""
        expanded = expand_to_step_trajectories(sample_trajectory)
        assert len(expanded) == 3

    def test_step_ids_are_unique(self, sample_trajectory):
        expanded = expand_to_step_trajectories(sample_trajectory)
        ids = [t.task_id for t in expanded]
        assert len(ids) == len(set(ids))
        assert ids == [
            "task_001_step_0",
            "task_001_step_1",
            "task_001_step_2",
        ]

    def test_each_step_has_single_tool(self, sample_trajectory):
        expanded = expand_to_step_trajectories(sample_trajectory)
        for sub in expanded:
            assert len(sub.tools_used) == 1

    def test_step_0_has_correct_tool(self, sample_trajectory):
        expanded = expand_to_step_trajectories(sample_trajectory)
        assert expanded[0].tools_used == {"search_api"}

    def test_step_2_has_all_prior_steps(self, sample_trajectory):
        expanded = expand_to_step_trajectories(sample_trajectory)
        last = expanded[2]
        # Should contain steps 0, 1, 2
        assert len(last.steps) == 3

    def test_metadata_has_step_info(self, sample_trajectory):
        expanded = expand_to_step_trajectories(sample_trajectory)
        meta = expanded[1].metadata
        assert meta["step_index"] == 1
        assert meta["total_steps"] == 3
        assert "previous_tools" in meta
        assert meta["previous_tools"] == ["search_api"]

    def test_available_tools_in_metadata(self, sample_trajectory):
        expanded = expand_to_step_trajectories(sample_trajectory)
        for sub in expanded:
            available = sub.metadata["available_tools"]
            assert isinstance(available, list)
            assert len(available) == 3  # all 3 tools from parent

    def test_preserves_intent_and_app(self, sample_trajectory):
        expanded = expand_to_step_trajectories(sample_trajectory)
        for sub in expanded:
            assert sub.intent == sample_trajectory.intent
            assert sub.app_name == sample_trajectory.app_name

    def test_empty_trajectory_returns_empty(self):
        traj = Trajectory(task_id="empty", intent="nothing", steps=[])
        assert expand_to_step_trajectories(traj) == []

    def test_skips_steps_without_tools(self):
        traj = Trajectory(
            task_id="mixed",
            intent="test",
            steps=[
                Step(action="tool_a()", observation="ok"),
                Step(action=None, observation="no tool"),  # no tool
                Step(action="tool_b()", observation="ok"),
            ],
        )
        expanded = expand_to_step_trajectories(traj)
        assert len(expanded) == 2
        assert expanded[0].tools_used == {"tool_a"}
        assert expanded[1].tools_used == {"tool_b"}


# ---------------------------------------------------------------------------
# Tests: BenchmarkAdapter protocol
# ---------------------------------------------------------------------------

class TestBenchmarkAdapterProtocol:
    """Verify the protocol is runtime-checkable."""

    def test_toolbench_adapter_satisfies_protocol(self):
        # Protocol with non-method members only supports isinstance(), not issubclass()
        adapter = ToolBenchAdapter()
        assert isinstance(adapter, BenchmarkAdapter)

    def test_minimal_adapter_satisfies_protocol(self):
        """A minimal class with the required methods passes the check."""

        class DummyAdapter:
            name = "dummy"

            def load_catalog(self) -> dict[str, str]:
                return {}

            def load_trajectories(self, split: str) -> list:
                return []

            def category_map(self) -> dict[str, str]:
                return {}

            def augment_training(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

        adapter = DummyAdapter()
        assert isinstance(adapter, BenchmarkAdapter)


# ---------------------------------------------------------------------------
# Tests: ToolBenchAdapter
# ---------------------------------------------------------------------------

class TestToolBenchAdapter:
    """Tests for the ToolBench-specific adapter."""

    def test_default_name(self):
        adapter = ToolBenchAdapter()
        assert adapter.name == "toolbench"

    def test_load_catalog_from_file(self, tmp_path):
        catalog_file = tmp_path / "catalog.json"
        catalog = {"tool_a": "Description A", "tool_b": "Description B"}
        with open(catalog_file, "w") as f:
            json.dump(catalog, f)

        adapter = ToolBenchAdapter(catalog_path=catalog_file)
        result = adapter.load_catalog()
        assert result == catalog

    def test_load_catalog_derives_from_trajectories(self, trajectory_file):
        adapter = ToolBenchAdapter(train_path=trajectory_file)
        # Trigger trajectory loading which derives catalog
        adapter.load_trajectories("train")
        catalog = adapter.load_catalog()
        assert len(catalog) > 0

    def test_load_train_filters_failures(self, trajectory_file):
        adapter = ToolBenchAdapter(train_path=trajectory_file)
        train = adapter.load_trajectories("train")
        # Only successful trajectories returned
        assert all(t.success for t in train)

    def test_load_train_caches_failures(self, trajectory_file):
        adapter = ToolBenchAdapter(train_path=trajectory_file)
        adapter.load_trajectories("train")
        assert len(adapter._failed_trajs) > 0

    def test_step_level_expansion(self, trajectory_file):
        config = BenchmarkConfig(step_level=True)
        adapter = ToolBenchAdapter(
            benchmark_config=config,
            train_path=trajectory_file,
        )
        train = adapter.load_trajectories("train")
        # With step expansion, we should have more samples than trajectories
        # The successful trajectory has 3 steps → 3 sub-trajectories
        assert len(train) == 3

    def test_load_test(self, trajectory_file):
        adapter = ToolBenchAdapter(eval_path=trajectory_file)
        test = adapter.load_trajectories("test")
        # Default ingest_config has success_only=True
        assert all(t.success for t in test)

    def test_invalid_split_raises(self):
        adapter = ToolBenchAdapter()
        with pytest.raises(ValueError, match="Unknown split"):
            adapter.load_trajectories("validation")

    def test_augment_training_noop_by_default(self):
        adapter = ToolBenchAdapter()
        df = pd.DataFrame({"label": [1, 0, 0], "feature": [0.1, 0.2, 0.3]})
        result = adapter.augment_training(df)
        assert len(result) == len(df)

    def test_category_map_returns_dict(self):
        adapter = ToolBenchAdapter()
        assert isinstance(adapter.category_map(), dict)


# ---------------------------------------------------------------------------
# Tests: Adapter registry
# ---------------------------------------------------------------------------

class TestAdapterRegistry:
    """Tests for the benchmark adapter registry."""

    def test_toolbench_is_registered(self):
        assert "toolbench" in ADAPTERS

    def test_list_adapters(self):
        adapters = list_adapters()
        assert "toolbench" in adapters
        assert adapters == sorted(adapters)

    def test_create_adapter_toolbench(self):
        cfg = TabAgentConfig()
        adapter = create_adapter("toolbench", cfg)
        assert adapter.name == "toolbench"
        assert isinstance(adapter, BenchmarkAdapter)

    def test_create_adapter_case_insensitive(self):
        cfg = TabAgentConfig()
        adapter = create_adapter("ToolBench", cfg)
        assert adapter.name == "toolbench"

    def test_create_adapter_unknown_raises(self):
        cfg = TabAgentConfig()
        with pytest.raises(ValueError, match="Unknown benchmark adapter"):
            create_adapter("nonexistent", cfg)

    def test_create_adapter_with_kwargs(self, tmp_path):
        cfg = TabAgentConfig()
        adapter = create_adapter(
            "toolbench",
            cfg,
            train_path=str(tmp_path / "train.jsonl"),
            eval_path=str(tmp_path / "eval.jsonl"),
        )
        assert adapter.train_path == tmp_path / "train.jsonl"


# ---------------------------------------------------------------------------
# Tests: BenchmarkConfig
# ---------------------------------------------------------------------------

class TestBenchmarkConfig:
    """Tests for the BenchmarkConfig in TabAgentConfig."""

    def test_default_benchmark_config(self):
        cfg = TabAgentConfig()
        assert cfg.benchmark.adapter == "toolbench"
        assert cfg.benchmark.step_level is False
        assert cfg.benchmark.use_failure_negatives is False
        assert cfg.benchmark.failure_negative_ratio == 0.3

    def test_benchmark_config_from_yaml(self, tmp_path):
        yaml_content = """
benchmark:
  adapter: "toolbench"
  step_level: true
  use_failure_negatives: true
  failure_negative_ratio: 0.5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)
        cfg = load_config(config_file)
        assert cfg.benchmark.step_level is True
        assert cfg.benchmark.use_failure_negatives is True
        assert cfg.benchmark.failure_negative_ratio == 0.5

    def test_backward_compatibility(self):
        """Existing config fields are unchanged."""
        cfg = TabAgentConfig()
        # All pre-existing fields still accessible
        assert cfg.ingest is not None
        assert cfg.features is not None
        assert cfg.dataset is not None
        assert cfg.classifier is not None
        assert cfg.evaluation is not None
