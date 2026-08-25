"""Trace assembly for the OTLP receiver (PR 4).

Groups raw OTLP spans by ``trace_id``, waits for a completion signal
(SDK ``shortchain.task`` root ended + settle time, idle timeout, or
max-age cap), then pipelines each assembled trace through
projector → quality gate → JSONL writer.

Bounds (enforced while buffering, not only after projection):
- ``max_inflight_traces``: on a new trace_id at the cap, first try to
  evict the oldest idle in-flight trace (flush as ``max_inflight_evict``);
  if nothing is idle, the offending spans are dropped and the receiver
  answers HTTP 200 + OTLP ``partial_success`` (never 429).
- ``max_spans_in``: per-trace cap on NON-PROTECTED spans; the SDK root
  (``shortchain.task`` / ``shortchain.task_root=true``) is never dropped
  and may evict skip/other/extra-LLM spans to stay near the cap (K13).
- ``seen_trace_ids`` LRU: dedup after flush; late spans are counted and
  dropped (never rewrite JSONL).

One ``threading.Lock`` guards ``append``/``flush``; the 1s tick and HTTP
handlers share it. ``workers=1`` is mandatory (multi-worker uvicorn would
split one trace_id across assemblers).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shortchain.config import RuntimeConfig
from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    OtelTraceProjector,
    _first_attr,
    classify,
)
from shortchain.ingest.quality import TrajectoryQualityGate
from shortchain.ingest.schema import Trajectory
from shortchain.utils.logging import get_logger

log = get_logger(__name__)

FLUSH_REASON_METRIC = {
    "explicit": "traces_flushed_explicit",
    "idle": "traces_flushed_idle",
    "max_age": "traces_flushed_max_age",
    "max_inflight_evict": "traces_flushed_evict",
    "shutdown": "traces_flushed_shutdown",
}


def _is_protected(span: OtelSpan) -> bool:
    """SDK root spans are never dropped by the assembler (K13)."""
    return span.name == "shortchain.task" or bool(
        _first_attr(span.attributes, "shortchain.task_root")
    )


@dataclass
class InflightTrace:
    """Spans buffered for one trace_id."""

    trace_id: str
    spans: list[OtelSpan] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    explicit_at: float | None = None  # monotonic time the SDK root ended


# ---------------------------------------------------------------------------
# JSONL writer (training artifact; secret material; chmod 600)
# ---------------------------------------------------------------------------


class JsonlTrajectoryWriter:
    """Append-only Trajectory JSONL writer with 0o600 creation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trajectory: Trajectory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = TrajectoryQualityGate().write_record(trajectory)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Metrics (in-process counters; exposed via receiver /metrics)
# ---------------------------------------------------------------------------


class RuntimeMetrics:
    """Trivial thread-safe counters — no external dependency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._labeled: dict[str, dict[str, int]] = {}
        self._inflight = 0

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def labeled_inc(self, name: str, label: str) -> None:
        with self._lock:
            self._labeled.setdefault(name, {})
            self._labeled[name][label] = self._labeled[name].get(label, 0) + 1

    def set_inflight(self, value: int) -> None:
        self._inflight = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._counters,
                "labeled": {k: dict(v) for k, v in self._labeled.items()},
                "inflight": self._inflight,
            }


# ---------------------------------------------------------------------------
# TraceAssembler
# ---------------------------------------------------------------------------


class TraceAssembler:
    """Buffer spans by trace_id and flush completed traces to JSONL."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        writer: JsonlTrajectoryWriter | None = None,
        projector: OtelTraceProjector | None = None,
        metrics: RuntimeMetrics | None = None,
        seen_size: int = 100_000,
        clock: Any = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.writer = writer or JsonlTrajectoryWriter(self.config.output)
        self.projector = projector or OtelTraceProjector(self.config.projection)
        self.gate = TrajectoryQualityGate(
            self.config.projection,
            require_success_true=self.config.require_success_true,
        )
        self.metrics = metrics or RuntimeMetrics()
        self._seen_size = seen_size
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.Lock()
        self._inflight: dict[str, InflightTrace] = {}
        self._seen: OrderedDict[str, None] = OrderedDict()

    def _now(self) -> float:
        return self._clock()

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def append(self, spans: list[OtelSpan]) -> int:
        """Buffer spans by trace_id.

        Returns the number of spans REJECTED because of the inflight cap
        (new trace ids) — the receiver reports these via OTLP
        ``partial_success``.
        """
        rejected = 0
        now = self._now()
        with self._lock:
            for span in spans:
                if self._route(span, now):
                    rejected += 1
            self.metrics.set_inflight(len(self._inflight))
            self.metrics.inc("spans_received", len(spans))
        return rejected

    def tick(self, now: float | None = None) -> list[str]:
        """Flush traces whose completion rule fired (1s background tick)."""
        now = now if now is not None else self._now()
        reasons: list[str] = []
        with self._lock:
            for trace_id in list(self._inflight):
                trace = self._inflight[trace_id]
                reason = self._completion_reason(trace, now)
                if reason is None:
                    continue
                self._flush(trace, reason)
                reasons.append(reason)
            self.metrics.set_inflight(len(self._inflight))
        return reasons

    def flush_all(self, reason: str = "shutdown") -> int:
        """Force-flush every in-flight trace (SIGTERM / tests)."""
        with self._lock:
            for trace in list(self._inflight.values()):
                self._flush(trace, reason)
            self.metrics.set_inflight(len(self._inflight))
        return 0

    def stats(self) -> dict[str, Any]:
        """Snapshot of metrics plus buffer sizes (for /metrics or logs)."""
        with self._lock:
            return {
                "inflight": len(self._inflight),
                "seen": len(self._seen),
                "metrics": self.metrics.snapshot(),
            }

    # ------------------------------------------------------------------
    # Internals (call under self._lock)
    # ------------------------------------------------------------------

    def _route(self, span: OtelSpan, now: float) -> bool:
        """Place one span; True when it was dropped at the inflight cap."""
        if span.trace_id in self._seen:
            self.metrics.inc("late_spans")
            return False

        trace = self._inflight.get(span.trace_id)
        if trace is None:
            if len(self._inflight) >= self.config.max_inflight_traces:
                if not self._evict_idle():
                    self.metrics.inc("rejected_traces_inflight")
                    return True
            trace = self._inflight[span.trace_id] = InflightTrace(trace_id=span.trace_id)
            trace.first_seen = trace.last_seen = now

        trace.last_seen = now

        if _is_protected(span):
            # Reserved: the SDK root may evict noise to stay near the cap.
            self._evict_if_full(trace)
            trace.explicit_at = now
            trace.spans.append(span)
            return False

        if len(trace.spans) >= self.config.max_spans_in:
            self.metrics.inc("spans_overflow")
            return False
        trace.spans.append(span)
        return False

    def _evict_idle(self) -> bool:
        """Flush the oldest trace idle ≥1s to make room; False if none."""
        now = self._now()
        idle = [
            t for t in self._inflight.values()
            if now - t.last_seen >= 1.0
        ]
        if not idle:
            return False
        idle.sort(key=lambda t: t.last_seen)
        self._flush(idle[0], "max_inflight_evict")
        return True

    def _evict_if_full(self, trace: InflightTrace) -> None:
        """Drop one non-protected span when the per-trace cap is reached."""
        if len(trace.spans) < self.config.max_spans_in:
            return
        # Eviction order: skip -> other -> extra LLM -> extra tool; never root.
        for role in ("skip", "other", "llm", "tool"):
            for i, buffered in enumerate(trace.spans):
                if _is_protected(buffered):
                    continue
                if classify(buffered) == role:
                    trace.spans.pop(i)
                    return
        assert False, "unreachable: cap must contain at least one non-protected span"

    def _completion_reason(self, trace: InflightTrace, now: float) -> str | None:
        """First matching rule: explicit (root + settle), idle, max_age."""
        if trace.explicit_at is not None:
            if now - trace.explicit_at >= self.config.settle_timeout_s:
                return "explicit"
            return None  # wait for the settle so late children can arrive
        if now - trace.last_seen >= self.config.idle_timeout_s:
            return "idle"
        if now - trace.first_seen >= self.config.max_trace_age_s:
            return "max_age"
        return None

    def _flush(self, trace: InflightTrace, reason: str) -> None:
        """Project, gate, and append one completed trace."""
        self._inflight.pop(trace.trace_id, None)
        self._mark_seen(trace.trace_id)
        self.metrics.inc(FLUSH_REASON_METRIC.get(reason, "traces_flushed_other"))

        otel_trace = OtelTrace(
            trace_id=trace.trace_id,
            spans=trace.spans,
            complete_reason=reason,
        )
        result = self.projector.project(otel_trace)
        if result.trajectory is None:
            drop = str(result.drop_reason or "unknown")
            self.metrics.labeled_inc("traces_dropped", drop)
            log.warning(f"trace_dropped trace_id={trace.trace_id} reason={drop}")
            return

        report = self.gate.check(result.trajectory)
        if not report.kept:
            drop = str(report.drop_reason or "gate")
            self.metrics.labeled_inc("traces_dropped", drop)
            log.warning(f"trace_dropped trace_id={trace.trace_id} reason={drop}")
            return

        framework = str(
            result.trajectory.metadata.get("projection.framework", "unknown")
        )
        self.metrics.labeled_inc("traces_projected", framework)
        log.info(
            f"trace_projected trace_id={trace.trace_id} "
            f"task_id={result.trajectory.task_id} "
            f"n_tools={len(result.trajectory.spans)} "
            f"success_source={result.trajectory.metadata.get('success_source')} "
            f"reason={reason}"
        )
        self.writer.append(result.trajectory)

    def _mark_seen(self, trace_id: str) -> None:
        self._seen[trace_id] = None
        self._seen.move_to_end(trace_id)
        while len(self._seen) > self._seen_size:
            self._seen.popitem(last=False)