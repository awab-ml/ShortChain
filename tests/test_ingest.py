"""Tests for the trajectory ingestion module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from shortchain.ingest.schema import Step, Trajectory
from shortchain.ingest.loader import JSONLTrajectoryLoader, load_trajectories
from shortchain.config import IngestConfig


# ---------------------------------------------------------------------------
# Step model tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_tool_name_extraction(self):
        step = Step(action="send_email")
        assert step.tool_name == "send_email"

    def test_tool_name_with_args(self):
        step = Step(action="send_email(to='john@example.com')")
        assert step.tool_name == "send_email"

    def test_tool_name_none_when_no_action(self):
        step = Step(thoughts="Just thinking...")
        assert step.tool_name is None

    def test_tool_name_none_when_empty(self):
        step = Step(action="")
        assert step.tool_name is None


# ---------------------------------------------------------------------------
# Trajectory model tests
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_tools_used_auto_derived(self):
        traj = Trajectory(
            task_id="t1",
            intent="test",
            steps=[
                Step(action="tool_a"),
                Step(action="tool_b"),
                Step(action="tool_a"),  # duplicate
            ],
        )
        assert traj.tools_used == {"tool_a", "tool_b"}

    def test_tools_used_explicit(self):
        """Explicit tools_used should be preserved."""
        traj = Trajectory(
            task_id="t1",
            intent="test",
            tools_used={"x", "y"},
            steps=[Step(action="tool_a")],
        )
        assert traj.tools_used == {"x", "y"}

    def test_n_steps(self):
        traj = Trajectory(task_id="t1", intent="test", steps=[Step(), Step()])
        assert traj.n_steps == 2

    def test_tool_sequence(self):
        traj = Trajectory(
            task_id="t1",
            intent="test",
            steps=[Step(action="a"), Step(thoughts="no action"), Step(action="b")],
        )
        assert traj.tool_sequence == ["a", "b"]

    def test_last_thought(self):
        traj = Trajectory(
            task_id="t1",
            intent="test",
            steps=[
                Step(thoughts="first thought"),
                Step(action="tool"),
                Step(thoughts="last thought"),
            ],
        )
        assert traj.last_thought == "last thought"

    def test_last_thought_none(self):
        traj = Trajectory(task_id="t1", intent="test", steps=[Step(action="a")])
        assert traj.last_thought is None

    def test_summary(self):
        traj = Trajectory(task_id="t1", intent="Send email", app_name="gmail")
        s = traj.summary()
        assert s["task_id"] == "t1"
        assert s["app"] == "gmail"


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestJSONLLoader:
    def _write_jsonl(self, records: list[dict], path: Path) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_load_jsonl(self, tmp_path: Path):
        records = [
            {
                "task_id": "t1",
                "intent": "Do something",
                "success": True,
                "steps": [
                    {"action": "tool_a", "agent_name": "Agent1"},
                    {"action": "tool_b", "agent_name": "Agent1"},
                ],
            },
            {
                "task_id": "t2",
                "intent": "Do another thing",
                "success": True,
                "steps": [{"action": "tool_c"}],
            },
        ]
        self._write_jsonl(records, tmp_path / "data.jsonl")
        trajs = load_trajectories(tmp_path)
        assert len(trajs) == 2
        assert trajs[0].task_id == "t1"
        assert trajs[0].tools_used == {"tool_a", "tool_b"}

    def test_filter_unsuccessful(self, tmp_path: Path):
        records = [
            {"task_id": "t1", "intent": "ok", "success": True, "steps": []},
            {"task_id": "t2", "intent": "fail", "success": False, "steps": []},
        ]
        self._write_jsonl(records, tmp_path / "data.jsonl")
        trajs = load_trajectories(tmp_path)
        assert len(trajs) == 1
        assert trajs[0].task_id == "t1"

    def test_load_all_when_not_filtering(self, tmp_path: Path):
        records = [
            {"task_id": "t1", "intent": "ok", "success": True, "steps": []},
            {"task_id": "t2", "intent": "fail", "success": False, "steps": []},
        ]
        self._write_jsonl(records, tmp_path / "data.jsonl")
        config = IngestConfig(success_only=False)
        trajs = load_trajectories(tmp_path, config=config)
        assert len(trajs) == 2

    def test_score_field_as_success(self, tmp_path: Path):
        records = [
            {"task_id": "t1", "intent": "ok", "score": 1.0, "steps": []},
            {"task_id": "t2", "intent": "fail", "score": 0.0, "steps": []},
        ]
        self._write_jsonl(records, tmp_path / "data.jsonl")
        trajs = load_trajectories(tmp_path)
        assert len(trajs) == 1

    def test_load_json_file(self, tmp_path: Path):
        data = [
            {"task_id": "t1", "intent": "test", "success": True, "steps": [{"action": "x"}]},
        ]
        path = tmp_path / "data.json"
        with open(path, "w") as f:
            json.dump(data, f)
        trajs = load_trajectories(path)
        assert len(trajs) == 1

    def test_simple_step_format(self, tmp_path: Path):
        """Steps as plain strings (tool names only)."""
        records = [
            {"task_id": "t1", "intent": "test", "success": True, "steps": ["tool_a", "tool_b"]},
        ]
        self._write_jsonl(records, tmp_path / "data.jsonl")
        trajs = load_trajectories(tmp_path)
        assert trajs[0].tools_used == {"tool_a", "tool_b"}

    def test_empty_directory(self, tmp_path: Path):
        trajs = load_trajectories(tmp_path)
        assert trajs == []

    def test_load_example_data(self):
        """Integration test: load the shipped example trajectories."""
        example_path = Path(__file__).parent.parent / "data" / "example"
        if not example_path.exists():
            pytest.skip("Example data not found")
        trajs = load_trajectories(example_path)
        assert len(trajs) >= 10
        for t in trajs:
            assert t.task_id
            assert t.intent
            assert t.success
