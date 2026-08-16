"""Golden tests for ShortChain.set_task / set_success (Section 6.2).

The load-bearing contract (K13): a post-run ``set_success`` must write onto
the STILL-OPEN SDK root (same trace_id as the tool spans) and then end it —
never start a new span.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace

from shortchain.runtime import task_span as ts
from shortchain.sdk import ShortChain

ROOT = "shortchain.task"

pytestmark = pytest.mark.usefixtures("otel_global_clear")


def finished(memory) -> list:
    return list(memory.get_finished_spans())


class TestGoldenContract:
    """set_task → fake child tool span (ended) → set_success(True)."""

    def test_success_written_on_same_trace_id(self, otel_global_clear):
        memory = otel_global_clear
        ShortChain.set_task("ticket-1842", intent="Refund order 9921", app_name="support")
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("execute_tool lookup_order"):
            pass  # instrumented tool span ends before the user code returns
        ShortChain.set_success(True)
        memory.force_flush()

        root = [s for s in finished(memory) if s.name == "shortchain.task"][0]
        tool = [s for s in finished(memory) if s.name == "execute_tool lookup_order"][0]

        # K13 golden: ONE trace_id; the root still carries success after the
        # child already ended; the child is a child of the root.
        assert root.context.trace_id == tool.context.trace_id
        assert tool.parent.span_id == root.context.span_id
        assert root.attributes["shortchain.success"] is True
        assert root.attributes["shortchain.task_id"] == "ticket-1842"
        assert root.attributes["shortchain.complete"] is True
        # No orphan success span on a different trace.
        assert len(finished(memory)) == 2

    def test_association_reaches_children(self, otel_global_clear):
        memory = otel_global_clear
        ShortChain.set_task(task_id="abc", intent="do the thing")
        tracer = trace.get_tracer("y")
        with tracer.start_as_current_span("tool_z"):
            pass
        ShortChain.end_task(success=True)
        memory.force_flush()
        tool = [s for s in finished(memory) if s.name == "tool_z"][0]
        assert tool.attributes["shortchain.task_id"] == "abc"
        assert tool.attributes["traceloop.association.properties.intent"] == "do the thing"

    def test_end_task_false_is_known_failure(self, otel_global_clear):
        memory = otel_global_clear
        ShortChain.set_task(task_id="fail", intent="x")
        ShortChain.end_task(success=False)
        root = [s for s in finished(memory) if s.name == "shortchain.task"][0]
        assert root.attributes["shortchain.success"] is False

    def test_set_success_without_task_is_noop(self, otel_global_clear):
        memory = otel_global_clear
        n_before = len(finished(memory))
        ShortChain.set_success(False)  # warns, no-op
        assert len(finished(memory)) == n_before

    def test_settle_split_not_created(self, otel_global_clear):
        memory = otel_global_clear
        ShortChain.set_task("one-shot")
        ShortChain.set_success(True)
        memory.force_flush()
        # Exactly one root; nothing else on a different trace.
        roots = [s for s in finished(memory) if s.name == "shortchain.task"]
        assert len(roots) == 1

    def test_set_association_on_open_root(self, otel_global_clear):
        memory = otel_global_clear
        ts.set_association(tenant="acme")
        ShortChain.set_task("x-1", intent="y")
        ShortChain.set_association(region="eu")
        ShortChain.end_task(success=True)
        root = [s for s in finished(memory) if s.name == ROOT][0]
        assert root.attributes.get("shortchain.region") == "eu"