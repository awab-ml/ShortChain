"""Tests for OTEL span classification and tool-name extraction (Section 2.2)."""

from __future__ import annotations

from shortchain.ingest.otel import (
    OtelSpan,
    classify,
    extract_tool_name,
)


def make_span(name: str, attrs: dict | None = None, span_id: str = "s1") -> OtelSpan:
    return OtelSpan(
        trace_id="a" * 32,
        span_id=span_id,
        parent_span_id=None,
        name=name,
        start_time_unix_nano=1,
        end_time_unix_nano=2,
        attributes=attrs or {},
    )


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


class TestClassify:
    def test_tool_by_operation(self):
        span = make_span(
            "execute_tool lookup_order",
            {"gen_ai.operation.name": "execute_tool", "traceloop.span.kind": "tool"},
        )
        assert classify(span) == "tool"

    def test_tool_by_tool_name_attribute(self):
        span = make_span(
            "random_name",
            {"gen_ai.tool.name": "lookup_order", "traceloop.span.kind": "tool"},
        )
        assert classify(span) == "tool"

    def test_tool_by_name_suffix(self):
        assert classify(make_span("lookup_order.tool")) == "tool"
        assert classify(make_span("execute_tool lookup_order")) == "tool"

    def test_openinference_function(self):
        span = make_span("function.lookup_order", {"openinference.span.kind": "TOOL"})
        assert classify(span) == "tool"

    def test_tool_kind_without_name(self):
        assert classify(make_span("blah", {"traceloop.span.kind": "tool"})) == "tool"

    def test_llm(self):
        span = make_span("chat", {"gen_ai.operation.name": "chat"})
        assert classify(span) == "llm"
        span = make_span("generation.gemini", {"openinference.span.kind": "LLM"})
        assert classify(span) == "llm"

    def test_agent(self):
        span = make_span(
            "agent",
            {"gen_ai.operation.name": "invoke_agent", "traceloop.span.kind": "agent"},
        )
        assert classify(span) == "agent"
        span = make_span("workflow", {"traceloop.span.kind": "workflow"})
        assert classify(span) == "agent"

    def test_task(self):
        span = make_span("execute_task", {"gen_ai.operation.name": "execute_task"})
        assert classify(span) == "task"
        span = make_span("some task", {"traceloop.span.kind": "task"})
        assert classify(span) == "task"

    def test_skip_ops_checked_first(self):
        """Retrievers carry kind=task but must be skipped before task check."""
        span = make_span(
            "vector store",
            {"gen_ai.operation.name": "vector_db_retrieve", "traceloop.span.kind": "task"},
        )
        assert classify(span) == "skip"
        span = make_span(
            "embed",
            {"gen_ai.operation.name": "embeddings", "traceloop.span.kind": "task"},
        )
        assert classify(span) == "skip"
        span = make_span(
            "hand off",
            {"traceloop.span.kind": "handoff"},
        )
        assert classify(span) == "skip"

    def test_skip_kinds(self):
        for kind in ("session", "server"):
            assert classify(make_span("x", {"traceloop.span.kind": kind})) == "skip"

    def test_skip_names(self):
        assert classify(make_span("mcp_tools", {"traceloop.span.kind": "tool"})) == "skip"
        assert classify(make_span("tools/list", {"traceloop.span.kind": "tool"})) == "skip"

    def test_sdk_task_root(self):
        span = make_span(
            "shortchain.task",
            {"shortchain.task_root": True, "traceloop.span.kind": "workflow"},
        )
        assert classify(span) == "root"

    def test_root_by_attribute(self):
        span = make_span("anything", {"shortchain.task_root": "true"})
        assert classify(span) == "root"

    def test_mcp_tools_tool_kind_skipped_by_name(self):
        """Catalog listings are kind=TOOL but must not be tasks."""
        span = make_span("mcp_tools", {"openinference.span.kind": "TOOL"})
        assert classify(span) == "skip"

    def test_other(self):
        assert classify(make_span("http_get")) == "other"


# ---------------------------------------------------------------------------
# extract_tool_name()
# ---------------------------------------------------------------------------


class TestExtractToolName:
    def test_gen_ai_tool_name(self):
        span = make_span(
            "execute_tool lookup_order",
            {"gen_ai.tool.name": "lookup_order", "traceloop.span.kind": "tool"},
        )
        assert extract_tool_name(span) == "lookup_order"

    def test_openinference_tool_name(self):
        span = make_span("function.lookup_order", {"tool.name": "lookup_order"})
        assert extract_tool_name(span) == "lookup_order"

    def test_traceloop_entity_name_for_tool_kind(self):
        span = make_span(
            "add_numbers.tool",
            {"traceloop.span.kind": "tool", "traceloop.entity.name": "add_numbers"},
        )
        assert extract_tool_name(span) == "add_numbers"

    def test_span_name_parse(self):
        assert extract_tool_name(make_span("execute_tool send_email")) == "send_email"
        assert extract_tool_name(make_span("send_email.tool")) == "send_email"
        assert extract_tool_name(make_span("function.send_email")) == "send_email"

    def test_mcp_client_entity_input_tool_name(self):
        span = make_span(
            "tools/call.tool",
            {"traceloop.entity.input.tool_name": "add_numbers", "traceloop.span.kind": "tool"},
        )
        assert extract_tool_name(span) == "add_numbers"

    def test_args_stripped(self):
        span = make_span("execute_tool send_email", {"gen_ai.tool.name": "send_email(a=1)"})
        assert extract_tool_name(span) == "send_email"

    def test_nameless_tool_kind_returns_none(self):
        """The load-bearing emit gate: nameless mcp_tools must be None."""
        span = make_span("mcp_tools", {"openinference.span.kind": "TOOL"})
        assert extract_tool_name(span) is None

    def test_tools_list_returns_none(self):
        span = make_span("tools/list", {"traceloop.span.kind": "tool"})
        assert extract_tool_name(span) is None

    def test_non_string_attr_ignored(self):
        span = make_span("x", {"gen_ai.tool.name": {"nested": 1}})
        assert extract_tool_name(span) is None

    def test_empty_name_returns_none(self):
        span = make_span("no_name_here")
        assert extract_tool_name(span) is None

    def test_shortchain_task_name_not_a_tool(self):
        span = make_span("shortchain.task", {"shortchain.task_root": True})
        assert extract_tool_name(span) is None