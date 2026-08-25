"""Tests for the trace assembler (Section 4.1)."""

from __future__ import annotations

from pathlib import Path

from shortchain.config import ProjectionConfig, RuntimeConfig
from shortchain.ingest.otel import OtelSpan
from shortchain.telemetry.assembler import (
    JsonlTrajectoryWriter,
    RuntimeMetrics,
    TraceAssembler,
)
from shortchain.utils.io import read_jsonl

TID_A = "a" * 32
TID_B = "b" * 32


class FakeClock:
    """Deterministic monotonic clock for assembler tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def set(self, value: float) -> None:
        self._now = value


def span(
    name: str,
    span_id: str = "s1",
    trace_id: str = TID_A,
    attrs: dict | None = None,
    parent: str | None = None,
) -> OtelSpan:
    return OtelSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        start_time_unix_nano=1,
        end_time_unix_nano=2,
        attributes=attrs or {},
    )


def tool_span(name: str, tool: str, trace_id: str = TID_A, span_id: str = "tool1") -> OtelSpan:
    return span(
        f"execute_tool {name}",
        span_id=span_id,
        trace_id=trace_id,
        attrs={
            "gen_ai.operation.name": "execute_tool",
            "traceloop.span.kind": "tool",
            "gen_ai.tool.name": tool,
            "gen_ai.tool.call.result": "ok",
        },
    )


def task_root(
    trace_id: str = TID_A,
    success: bool = True,
    intent: str = "Do the thing",
) -> OtelSpan:
    return span(
        "shortchain.task",
        "root",
        trace_id=trace_id,
        attrs={
            "shortchain.task_root": True,
            "shortchain.task_id": "task-1",
            "shortchain.intent": intent,
            "shortchain.success": success,
        },
    )


def make_config(tmp_path: Path, **overrides) -> RuntimeConfig:
    cfg = RuntimeConfig(
        output=str(tmp_path / "unused.jsonl"),
        idle_timeout_s=30.0,
        settle_timeout_s=2.0,
        max_trace_age_s=300.0,
        max_inflight_traces=512,
        max_spans_in=500,
        require_success_true=True,
        projection=ProjectionConfig(require_intent=False),
    )
    return cfg.model_copy(update=overrides)


def make_assembler(tmp_path: Path, **kw):
    cfg = make_config(tmp_path)
    if "output" in kw:
        out = kw.pop("output")
    else:
        out = tmp_path / "trajectories.jsonl"
    cfg = cfg.model_copy(update={"output": str(out), **kw})
    metrics = RuntimeMetrics()
    clock = FakeClock()
    asm = TraceAssembler(cfg, writer=JsonlTrajectoryWriter(out), metrics=metrics, clock=clock)
    return asm, out, clock


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


# ---------------------------------------------------------------------------
# Completion rules
# ---------------------------------------------------------------------------


class TestCompletion:
    def test_explicit_root_with_settle(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path)
        clock.advance(1.0)
        asm.append([task_root(), tool_span("do_it", "do_it")])
        # Before the settle window: nothing flushed.
        assert asm.tick() == []
        assert not out.exists()
        # 1s after the root ended (settle_timeout_s=2): still waiting.
        clock.advance(1.0)
        assert asm.tick() == []
        # 2s after the root ended: explicit flush fires.
        clock.advance(1.0)
        assert asm.tick() == ["explicit"]
        records = load_records(out)
        assert len(records) == 1
        assert records[0]["task_id"] == "task-1"
        assert records[0]["success"] is True

    def test_idle_timeout(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path)
        clock.advance(1.0)
        asm.append([tool_span("do_it", "do_task")])  # no root → idle rule
        clock.advance(29.0)  # now = 30.0; just under 30s idle
        assert asm.tick() == []
        clock.advance(1.0)  # now = 31.0; idle reached
        assert asm.tick() == ["idle"]

    def test_max_age(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path, idle_timeout_s=30.0, max_trace_age_s=5.0)
        clock.advance(1.0)
        asm.append([tool_span("x", "x")])   # first_seen at t=1
        clock.advance(3.0)                  # t=4: 3s < 5s cap
        assert asm.tick() == []
        clock.advance(1.0)                  # t=5: not yet (first_seen+5)
        assert asm.tick() == []
        clock.advance(1.0)                  # t=6: max_age fires (idle is later)
        assert asm.tick() == ["max_age"]

    def test_settle_wait_does_not_idle_flush(self, tmp_path: Path):
        """An explicit root + settle must NOT be split by idle while waiting."""
        asm, out, clock = make_assembler(tmp_path, idle_timeout_s=5.0)
        clock.advance(1.0)
        asm.append([task_root()])
        assert asm.tick() == []
        clock.advance(1.0)   # 1s after root: in settle window
        assert asm.tick() == []
        clock.advance(1.0)   # 2s after root: explicit, before 5s idle
        assert asm.tick() == ["explicit"]


# ---------------------------------------------------------------------------
# Inflight cap
# ---------------------------------------------------------------------------


class TestInflight:
    def test_evicts_oldest_idle_at_cap(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path, max_inflight_traces=1)
        clock.advance(1.0)
        # A: full trace (root + tool) so the eviction flush produces JSONL.
        asm.append([task_root(trace_id=TID_A), tool_span("a", "ta", trace_id=TID_A)])
        clock.advance(2.0)  # trace A idle for >1s at t=3
        # New trace at the cap: A is evicted (flushed as max_inflight_evict).
        asm.append([tool_span("b", "tb", trace_id=TID_B)])
        assert asm.stats()["metrics"].get("traces_flushed_evict", 0) == 1
        # A's spans were projected+written even though it never settled.
        records = load_records(out)
        assert len(records) == 1
        assert records[0]["spans"][0]["action"] == "ta"

    def test_new_trace_rejected_when_nothing_idle(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path, max_inflight_traces=1)
        clock.advance(1.0)
        asm.append([tool_span("x", "ta", trace_id=TID_A)])
        # Same tick: nothing idle yet → B's span rejected (partial_success).
        rejected = asm.append([tool_span("y", "tb", trace_id=TID_B)])
        assert rejected == 1
        assert asm.stats()["metrics"]["rejected_traces_inflight"] == 1
        # Already-buffered tid spans are still accepted.
        assert asm.append([tool_span("z", "tc", trace_id=TID_A)]) == 0

    def test_max_inflight_eq_config(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path, max_inflight_traces=2)
        clock.advance(1.0)
        asm.append([tool_span("a", "ta", trace_id=TID_A)])
        clock.advance(1.0)
        asm.append([tool_span("b", "tb", trace_id=TID_B)])
        assert asm.stats()["inflight"] == 2


# ---------------------------------------------------------------------------
# Span caps + protected root
# ---------------------------------------------------------------------------


class TestSpanCaps:
    def test_nonprotected_overflow_dropped(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path, max_spans_in=3)
        clock.advance(1.0)
        # Five tool spans: 3 admitted (cap), 2 drop as overflow.
        asm.append([tool_span("t1", f"t{i}", span_id=f"t{i}") for i in range(5)])
        asm.append([task_root()])  # root arrives at cap → evicts ONE tool (last resort)
        clock.advance(100.0)
        asm.tick()
        records = load_records(out)
        assert len(records) == 1
        # 3 admitted - 1 evicted for the reserved root = 2 tools + no root span.
        assert len(records[0]["spans"]) == 2
        assert asm.stats()["metrics"]["spans_overflow"] == 2

    def test_protected_root_never_evicted(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path, max_spans_in=2, idle_timeout_s=10.0)
        clock.advance(1.0)
        asm.append([tool_span("a", "ta"), tool_span("b", "tb")])
        asm.append([task_root()])  # root arrives at cap → evicts ONE tool (last resort)
        clock.advance(100.0)
        asm.tick()
        records = load_records(out)
        assert len(records) == 1
        # The root was reserved: a successful trace must still emit the
        # remaining tool span (hard cap + never-dropped root contract).
        assert len(records[0]["spans"]) == 1
        assert records[0]["success"] is True

    def test_late_span_dropped_never_rewrites(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path)
        clock.advance(1.0)
        asm.append([task_root(), tool_span("t", "t")])
        clock.advance(10.0)
        asm.tick()  # explicit flush
        n_before = len(load_records(out))
        asm.append([tool_span("late", "late")])
        assert asm.stats()["metrics"]["late_spans"] == 1
        assert len(load_records(out)) == n_before


# ---------------------------------------------------------------------------
# Success gating + writer secrecy
# ---------------------------------------------------------------------------


class TestSuccessGate:
    def test_success_false_dropped_at_ingest(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path)
        clock.advance(1.0)
        asm.append([task_root(success=False), tool_span("do", "do")])
        clock.advance(10.0)
        asm.tick()
        assert load_records(out) == []
        assert "success_false" in asm.stats()["metrics"]["labeled"]["traces_dropped"]

    def test_success_false_kept_when_flagged_off(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path, require_success_true=False)
        clock.advance(1.0)
        asm.append([task_root(success=False), tool_span("do", "do")])
        clock.advance(10.0)
        asm.tick()
        records = load_records(out)
        assert len(records) == 1
        assert records[0]["success"] is False

    def test_file_created_0600(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path)
        clock.advance(1.0)
        asm.append([task_root(), tool_span("do", "do")])
        clock.advance(10.0)
        asm.tick()
        assert out.exists()
        assert (out.stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------------------
# Concurrency smoke
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_append_and_tick(self, tmp_path: Path):
        asm, out, clock = make_assembler(tmp_path)
        import threading

        errors: list[Exception] = []

        def writer_thread() -> None:
            try:
                for i in range(50):
                    asm.append([tool_span("w", f"w{i}", span_id=f"w{i}")])
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        thread = threading.Thread(target=writer_thread)
        thread.start()
        for _ in range(20):
            asm.tick()
        thread.join(timeout=5)
        assert not errors
        # Buffer is consistent: append + tick never interleave corruptly.
        assert asm.stats()["inflight"] <= 1