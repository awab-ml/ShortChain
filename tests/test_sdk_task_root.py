"""Tests for the task-root span lifecycle (Section 6.1)."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import set_tracer_provider

from shortchain.runtime import task_span as ts
from shortchain.runtime.association import (
    AssociationInjectionSpanProcessor,
    association_values,
)

ROOT = "shortchain.task"


def _build_env() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(AssociationInjectionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(memory))
    set_tracer_provider(provider)
    return provider, memory





# The global provider can only be set once (OTEL refuses overrides): create
# one module-scoped env and clear the exporter before each test.
_PROVIDER, _MEMORY = _build_env()


@pytest.fixture(autouse=True)
def _clear_env():
    _MEMORY.clear()
    ts.end_task()  # close any leaked task root between tests
    _MEMORY.clear()
    yield
    _MEMORY.clear()


def roots(memory: InMemorySpanExporter) -> list:
    return [s for s in memory.get_finished_spans() if s.name == ROOT]


def children(memory: InMemorySpanExporter) -> list:
    return [s for s in memory.get_finished_spans() if s.name != ROOT]


# ---------------------------------------------------------------------------
# open_task / end_task
# ---------------------------------------------------------------------------


class TestOpenEnd:
    def test_root_ended_with_success_and_same_trace_as_child(self):
        memory = _MEMORY
        ts.open_task("t1", intent="Refund", app_name="support")
        child = trace.get_tracer("t").start_span("execute_tool look")
        child.end()
        ts.end_task(success=True)
        memory.force_flush()

        root = roots(memory)[0]
        child_ = children(memory)[0]
        assert root.attributes["shortchain.success"] is True
        assert root.attributes["shortchain.task_id"] == "t1"
        assert root.attributes["traceloop.span.kind"] == "workflow"
        assert root.attributes["shortchain.complete"] is True
        # K13 golden: one trace_id; the child is parented under the root.
        assert root.context.trace_id == child_.context.trace_id
        assert child_.parent.span_id == root.context.span_id

    def test_end_without_success_ends_but_no_success_attr(self):
        memory = _MEMORY
        ts.open_task("t2")
        ts.end_task()  # success=None
        root = roots(memory)[0]
        assert "shortchain.success" not in root.attributes
        assert root.attributes["shortchain.complete"] is True

    def test_set_success_false_is_known_failure(self):
        memory = _MEMORY
        ts.open_task("t3")
        ts.set_success(False)
        root = roots(memory)[0]
        assert root.attributes["shortchain.success"] is False
        assert root.attributes["traceloop.association.properties.success"] is False

    def test_end_without_task_is_noop(self):
        ts.end_task(success=True)  # warning only, nothing leaks
        assert ts.current_task() is None


# ---------------------------------------------------------------------------
# Nested set_task
# ---------------------------------------------------------------------------


class TestNested:
    def test_nested_set_task_ends_previous_without_success(self):
        memory = _MEMORY
        ts.open_task("outer")
        ts.open_task("inner")
        ts.end_task(success=True)
        finished = memory.get_finished_spans()
        assert len([s for s in finished if s.name == ROOT]) == 2
        outer = [s for s in finished if s.name == ROOT and s.attributes["shortchain.task_id"] == "outer"][0]
        assert "shortchain.success" not in outer.attributes
        inner = [s for s in finished if s.name == ROOT and s.attributes["shortchain.task_id"] == "inner"][0]
        assert inner.attributes["shortchain.success"] is True

    def test_handle_cleared_after_end(self):
        ts.open_task("a")
        ts.open_task("b")
        ts.end_task(success=True)
        assert ts.current_task() is None


# --------------------------------------------------------------------- #
# Association merge-not-replace + injection
# --------------------------------------------------------------------- #


class TestAssociation:
    def test_merge_not_replace(self):
        ts.open_task("t9", intent="x")
        ts.set_association(a="1")
        ts.set_association(b="2")
        ts.set_association(a="3")
        assert association_values() == {"a": "3", "b": "2", "task_id": "t9", "intent": "x"}

    def test_map_written_onto_open_root(self):
        memory = _MEMORY
        ts.open_task("t9", intent="x")
        ts.set_association(other_id="42")
        ts.end_task(success=True)
        attrs = dict(roots(memory)[0].attributes or {})
        assert attrs["shortchain.other_id"] == "42"
        assert attrs["traceloop.association.properties.other_id"] == "42"

    def test_injection_processor_copies_to_children(self):
        memory = _MEMORY
        ts.open_task("t10", intent="hello")
        child = trace.get_tracer("t").start_span("tool_span")
        child.end()
        ts.end_task(success=True)
        child_ = children(memory)[0]
        assert child_.attributes["shortchain.task_id"] == "t10"
        assert child_.attributes["shortchain.intent"] == "hello"
        assert child_.attributes["traceloop.association.properties.task_id"] == "t10"

    def test_association_cleared_after_end(self):
        ts.open_task("t11")
        ts.end_task(success=True)
        assert association_values() == {}

    def test_set_task_children_inherit_via_start_span(self):
        memory = _MEMORY
        ts.open_task("t12", intent="intent", app_name="app")
        child = trace.get_tracer("t").start_span("child_1")
        child.end()
        ts.end_task(success=True)
        child_ = children(memory)[0]
        assert child_.attributes["shortchain.intent"] == "intent"
        assert child_.attributes["shortchain.app_name"] == "app"

    def test_set_association_without_task_is_noop(self):
        ts.set_association(k="v")
        assert association_values() == {}