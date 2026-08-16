"""Tests for the projection quality gate (Section 3.1)."""

from __future__ import annotations

from shortchain.config import ProjectionConfig
from shortchain.ingest.quality import TrajectoryQualityGate
from shortchain.ingest.schema import Span, Trajectory


def kept_trajectory(**kw) -> Trajectory:
    defaults = dict(
        task_id="t1",
        intent="Refund order 9921",
        spans=[Span(action="lookup_order"), Span(action="refund_order")],
        success=True,
        metadata={"success_source": "association"},
    )
    defaults.update(kw)
    return Trajectory(**defaults)


def kept_dict(**kw) -> dict:
    defaults = {
        "task_id": "t1",
        "intent": "Refund order 9921",
        "success": True,
        "spans": [
            {"action": "lookup_order", "metadata": {}},
            {"action": "refund_order", "metadata": {}},
        ],
        "metadata": {"success_source": "association", "otel.trace_id": "t1"},
    }
    defaults.update(kw)
    return defaults


def gate(**kw) -> TrajectoryQualityGate:
    return TrajectoryQualityGate(**kw)


# ---------------------------------------------------------------------------
# Trajectory objects
# ---------------------------------------------------------------------------


class TestTrajectoryChecks:
    def test_keeps_valid(self):
        report = gate().check(kept_trajectory())
        assert report.kept is True
        assert report.drop_reason is None

    def test_missing_intent(self):
        report = gate().check(kept_trajectory(intent="   "))
        assert report.kept is False
        assert report.drop_reason == "missing_intent"

    def test_zero_tool_spans(self):
        report = gate().check(kept_trajectory(spans=[]))
        assert report.drop_reason == "zero_tool_spans"

    def test_success_source_missing_is_unknown(self):
        """Poison-default guard: missing success_source drops even when
        success is True (Trajectory.success defaults to True in schema)."""
        traj = kept_trajectory(metadata={})
        report = gate().check(traj)
        assert report.kept is False
        assert report.drop_reason == "success_unknown"

    def test_success_source_unknown(self):
        traj = kept_trajectory(metadata={"success_source": "unknown"})
        assert gate().check(traj).drop_reason == "success_unknown"

    def test_known_failure_kept_by_default(self):
        """require_success_true is OFF at the gate; loader filters later."""
        traj = kept_trajectory(
            success=False, metadata={"success_source": "association"}
        )
        assert gate().check(traj).kept is True

    def test_success_false_when_required(self):
        traj = kept_trajectory(success=False, metadata={"success_source": "association"})
        report = gate(require_success_true=True).check(traj)
        assert report.drop_reason == "success_false"

    def test_too_many_spans(self):
        traj = kept_trajectory(
            spans=[Span(action=f"t{i}") for i in range(201)],
            metadata={"success_source": "association"},
        )
        assert gate().check(traj).drop_reason == "too_many_spans"

    def test_duplicate_trace(self):
        seen: set[str] = set()
        gate().check(kept_trajectory(), seen_trace_ids=seen)
        report = gate().check(kept_trajectory(), seen_trace_ids=seen)
        assert report.drop_reason == "duplicate_trace"

    def test_none_record(self):
        assert gate().check(None).drop_reason == "zero_tool_spans"


# ---------------------------------------------------------------------------
# Raw dicts (serialized records — the receiver writes these)
# ---------------------------------------------------------------------------


class TestDictChecks:
    def test_keeps(self):
        assert gate().check(kept_dict()).kept is True

    def test_missing_intent(self):
        assert gate().check(kept_dict(intent="")).drop_reason == "missing_intent"

    def test_zero_tool_spans(self):
        assert gate().check(kept_dict(spans=[])).drop_reason == "zero_tool_spans"

    def test_missing_success_key_is_unknown(self):
        """A projected dict with NO success key cannot pass the gate, even
        though the loader would default success to True on the Trajectory."""
        record = kept_dict()
        del record["success"]
        assert "success" not in record  # guard: key truly absent
        report = gate().check(record)
        assert report.kept is False
        assert report.drop_reason == "success_unknown"

    def test_success_key_present_but_source_missing(self):
        record = kept_dict()
        del record["metadata"]["success_source"]
        report = gate().check(record)
        assert report.kept is False
        assert report.drop_reason == "success_unknown"

    def test_success_false_kept_by_default_from_dict(self):
        record = kept_dict(success=False)
        record["metadata"]["success_source"] = "association"
        assert gate().check(record).kept is True

    def test_duplicate_trace_from_dict(self):
        seen: set[str] = set()
        gate().check(kept_dict(), seen_trace_ids=seen)
        assert gate().check(kept_dict(), seen_trace_ids=seen).drop_reason == "duplicate_trace"


# ---------------------------------------------------------------------------
# Writer contract
# ---------------------------------------------------------------------------


class TestWriteRecord:
    def test_omits_tools_used(self):
        record = gate().write_record(kept_trajectory())
        assert "tools_used" not in record
        assert record["success"] is True
        assert record["metadata"]["success_source"] == "association"
        # enum-serializable: json.dumps must not explode on the set
        import json

        json.dumps(record)

    def test_spans_metadata_round_trip(self):
        record = gate().write_record(kept_trajectory())
        assert record["spans"][0]["action"] == "lookup_order"
        assert record["spans"][0]["metadata"] == {}
        assert record["metadata"]["success_source"] == "association"

    def test_written_record_passes_gate(self):
        traj = kept_trajectory()
        record = gate().write_record(traj)
        assert gate().check(record).kept is True

    def test_written_record_missing_source_fails_gate(self):
        traj = kept_trajectory(metadata={})
        record = gate().write_record(traj)
        assert gate().check(record).drop_reason == "success_unknown"

    def test_max_spans_zero_disables_cap(self):
        traj = kept_trajectory(
            spans=[Span(action=f"t{i}") for i in range(300)],
            metadata={"success_source": "association"},
        )
        custom = gate(config=ProjectionConfig(max_spans=0))
        assert custom.check(traj).kept is True