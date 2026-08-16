"""Tests for the offline OtelTrajectoryLoader (Section 2.7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shortchain.config import ProjectionConfig
from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    OtelTrajectoryLoader,
)
from shortchain.ingest.base import TrajectoryLoader


def make_span(
    span_id: str,
    name: str,
    parent: str | None,
    start: int,
    end: int,
    attrs: dict | None = None,
    resource: dict | None = None,
) -> OtelSpan:
    return OtelSpan(
        trace_id="a" * 32,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        start_time_unix_nano=start,
        end_time_unix_nano=end,
        attributes=attrs or {},
        resource=resource or {},
    )


def make_trace() -> OtelTrace:
    root = make_span(
        "root",
        "shortchain.task",
        None,
        0,
        100,
        attrs={
            "shortchain.task_root": True,
            "traceloop.span.kind": "workflow",
            "shortchain.task_id": "job-7",
            "shortchain.intent": "Process the order",
            "shortchain.success": True,
        },
    )
    tool = make_span(
        "tool1",
        "execute_tool process_order",
        "root",
        30,
        40,
        attrs={
            "gen_ai.operation.name": "execute_tool",
            "traceloop.span.kind": "tool",
            "gen_ai.tool.name": "process_order",
            "gen_ai.tool.call.result": "done",
        },
    )
    return OtelTrace(trace_id="a" * 32, spans=[root, tool], complete_reason="explicit")


def _dump_trace(trace: OtelTrace) -> dict:
    return trace.model_dump()


def write_trace_json(dir: Path, name: str, trace: OtelTrace) -> Path:
    path = dir / name
    path.write_text(json.dumps(_dump_trace(trace)))
    return path


# ---------------------------------------------------------------------------
# Single file / list / jsonl / directory
# ---------------------------------------------------------------------------


class TestLoaderFiles:
    def test_single_trace_file(self, tmp_path: Path):
        path = write_trace_json(tmp_path, "a.json", make_trace())
        trajs = OtelTrajectoryLoader().load(path)
        assert len(trajs) == 1
        assert trajs[0].task_id == "job-7"
        assert trajs[0].tools_used == {"process_order"}

    def test_list_of_traces_in_one_file(self, tmp_path: Path):
        t1 = make_trace()
        t2 = make_trace()
        t2.spans[0].attributes["shortchain.task_id"] = "job-8"
        path = tmp_path / "two.json"
        path.write_text(json.dumps([_dump_trace(t1), _dump_trace(t2)]))
        trajs = OtelTrajectoryLoader().load(path)
        assert [t.task_id for t in trajs] == ["job-7", "job-8"]

    def test_jsonl_of_traces(self, tmp_path: Path):
        path = tmp_path / "traces.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_dump_trace(make_trace())) + "\n")
        assert len(OtelTrajectoryLoader().load(path)) == 1

    def test_directory_loads_all_files(self, tmp_path: Path):
        write_trace_json(tmp_path, "a.json", make_trace())
        write_trace_json(tmp_path, "b.json", make_trace())
        nested = tmp_path / "sub"
        nested.mkdir()
        write_trace_json(nested, "c.jsonl", make_trace())
        trajs = OtelTrajectoryLoader().load(tmp_path)
        assert len(trajs) == 3

    def test_empty_directory(self, tmp_path: Path):
        assert OtelTrajectoryLoader().load(tmp_path) == []

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            OtelTrajectoryLoader().load(tmp_path / "nope.json")

    def test_implements_protocol(self):
        assert isinstance(OtelTrajectoryLoader(), TrajectoryLoader)


# ---------------------------------------------------------------------------
# Quality flags apply through the loader
# ---------------------------------------------------------------------------


class TestLoaderFlags:
    def test_drops_by_default(self, tmp_path: Path):
        t = make_trace()
        t.spans[0].attributes.pop("shortchain.success")
        path = write_trace_json(tmp_path, "a.json", t)  # success unknown
        assert OtelTrajectoryLoader().load(path) == []

    def test_keeps_unknown_success_when_flagged_off(self, tmp_path: Path):
        t = make_trace()
        t.spans[0].attributes.pop("shortchain.success")
        path = write_trace_json(tmp_path, "a.json", t)
        cfg = ProjectionConfig(require_known_success=False)
        trajs = OtelTrajectoryLoader(cfg).load(path)
        assert len(trajs) == 1
        assert trajs[0].success is False
        assert trajs[0].metadata["success_source"] == "unknown"

    def test_success_false_trace_loaded(self, tmp_path: Path):
        t = make_trace()
        t.spans[0].attributes["shortchain.success"] = False
        path = write_trace_json(tmp_path, "a.json", t)
        trajs = OtelTrajectoryLoader().load(path)
        assert len(trajs) == 1
        assert trajs[0].success is False