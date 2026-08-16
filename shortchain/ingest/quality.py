"""Projection quality gate — reusable by the receiver and offline loader.

Applied after ``OtelTraceProjector`` (or immediately in ``OtelTrajectoryLoader``)
so messy runtime traces cannot poison ``success_only`` labels. Operates on both
:class:`Trajectory` objects and raw serialized dicts (the poison-default guard:
a projected record with no ``success`` key must not reach the training path,
even though ``Trajectory.success`` defaults to ``True`` in ``schema.py``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from shortchain.config import ProjectionConfig
from shortchain.ingest.schema import Trajectory


class QualityReport(BaseModel):
    """Outcome of one quality-gate check."""

    kept: bool = True
    drop_reason: str | None = None


class TrajectoryQualityGate:
    """Drop reasons: missing_intent | zero_tool_spans | success_unknown |
    success_false | too_many_spans | duplicate_trace."""

    def __init__(
        self,
        config: ProjectionConfig | None = None,
        *,
        require_success_true: bool = False,
    ) -> None:
        self.config = config or ProjectionConfig()
        self.require_success_true = require_success_true

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        record: Trajectory | dict[str, Any] | None,
        *,
        seen_trace_ids: set[str] | None = None,
    ) -> QualityReport:
        """Validate one record; returns kept + optional drop reason."""
        if record is None:
            return QualityReport(kept=False, drop_reason="zero_tool_spans")
        if isinstance(record, Trajectory):
            return self._check_trajectory(record, seen_trace_ids)
        return self._check_dict(record, seen_trace_ids)

    def write_record(self, trajectory: Trajectory) -> dict[str, Any]:
        """Serialize a kept trajectory for JSONL (training path contract).

        - ``model_dump(mode="json")`` so ``tools_used`` (a set) becomes a
          list instead of raising.
        - Omit ``tools_used``: the loader derives it from ``spans`` via the
          validator (avoids stale lists).
        - Always emit ``success`` and ``metadata.success_source``.
        """
        record = trajectory.model_dump(mode="json")
        record.pop("tools_used", None)
        metadata = dict(record.get("metadata") or {})
        metadata.setdefault(
            "success_source",
            trajectory.metadata.get("success_source", "unknown"),
        )
        record["metadata"] = metadata
        record["success"] = bool(trajectory.success)
        return record

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_trajectory(
        self,
        trajectory: Trajectory,
        seen: set[str] | None,
    ) -> QualityReport:
        if self.config.require_intent and not (trajectory.intent or "").strip():
            return QualityReport(kept=False, drop_reason="missing_intent")

        tool_spans = [s for s in trajectory.spans if s.tool_name]
        if self.config.require_tool_spans and not tool_spans:
            return QualityReport(kept=False, drop_reason="zero_tool_spans")

        if self.config.require_known_success:
            source = trajectory.metadata.get("success_source")
            if not source or source == "unknown":
                return QualityReport(kept=False, drop_reason="success_unknown")
            if self.require_success_true and not trajectory.success:
                return QualityReport(kept=False, drop_reason="success_false")

        if self.config.max_spans and len(trajectory.spans) > self.config.max_spans:
            return QualityReport(kept=False, drop_reason="too_many_spans")

        if seen is not None:
            trace_id = trajectory.metadata.get("otel.trace_id") or trajectory.task_id
            if trace_id in seen:
                return QualityReport(kept=False, drop_reason="duplicate_trace")
            seen.add(trace_id)

        return QualityReport(kept=True)

    def _check_dict(
        self,
        record: dict[str, Any],
        seen: set[str] | None,
    ) -> QualityReport:
        intent = record.get("intent")
        if self.config.require_intent and not (intent or "").strip():
            return QualityReport(kept=False, drop_reason="missing_intent")

        spans = record.get("spans") or []
        if self.config.require_tool_spans:
            tool_sequence = [
                s.get("action")
                for s in spans
                if isinstance(s, dict) and s.get("action")
            ]
            if not tool_sequence:
                return QualityReport(kept=False, drop_reason="zero_tool_spans")

        metadata = dict(record.get("metadata") or {})
        source = metadata.get("success_source")
        if self.config.require_known_success:
            # Both keys are mandatory on serialized records: a hand-built
            # dump missing either could poison labels via the Trajectory
            # defaults (success=True, metadata freely omitted).
            if "success" not in record or not source or source == "unknown":
                return QualityReport(kept=False, drop_reason="success_unknown")
            if self.require_success_true and record.get("success") is False:
                return QualityReport(kept=False, drop_reason="success_false")

        if self.config.max_spans and len(spans) > self.config.max_spans:
            return QualityReport(kept=False, drop_reason="too_many_spans")

        if seen is not None:
            trace_id = metadata.get("otel.trace_id") or record.get("task_id")
            if trace_id in seen:
                return QualityReport(kept=False, drop_reason="duplicate_trace")
            seen.add(trace_id)

        return QualityReport(kept=True)