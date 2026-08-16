"""Tests for MCP twin dedup and ordering (Section 2.4)."""

from __future__ import annotations

from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    dedupe_tool_spans,
)

TID = "a" * 32


def span(
    name: str,
    span_id: str,
    parent: str | None,
    start: int,
    end: int | None = None,
    attrs: dict | None = None,
) -> OtelSpan:
    return OtelSpan(
        trace_id=TID,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        start_time_unix_nano=start,
        end_time_unix_nano=end if end is not None else start + 10,
        attributes=attrs or {},
    )


def tool_attr(tool_name: str) -> dict:
    return {"gen_ai.operation.name": "execute_tool", "traceloop.span.kind": "tool"}


def by_id(trace: OtelTrace) -> dict[str, OtelSpan]:
    return {s.span_id: s for s in trace.spans}


def tool_candidates(trace: OtelTrace) -> list[OtelSpan]:
    from shortchain.ingest.otel import classify

    return [s for s in trace.spans if classify(s) == "tool"]


def run_dedup(trace: OtelTrace) -> list[OtelSpan]:
    return dedupe_tool_spans(tool_candidates(trace), by_id(trace))


def names(spans: list[OtelSpan]) -> list[str]:
    from shortchain.ingest.otel import extract_tool_name

    return [extract_tool_name(s) for s in spans]


# ---------------------------------------------------------------------------
# mcp_tool_client_server.json: one add_numbers call, two raw tool spans -> one
# ---------------------------------------------------------------------------


class TestMcpClientServerTwin:
    def test_one_call_two_spans_collapse_to_one(self):
        trace = OtelTrace(
            trace_id=TID,
            spans=[
                span("mcp.client.session", "session", None, 0),
                span("mcp.server", "mcp_server", None, 0, attrs={"traceloop.span.kind": "server"}),
                # client twin (nested under session) + server twin (nested under mcp.server)
                span(
                    "tools/call.tool",
                    "client",
                    "session",
                    10,
                    attrs={"traceloop.span.kind": "tool", "traceloop.entity.input.tool_name": "add_numbers"},
                ),
                span(
                    "add_numbers.tool",
                    "server",
                    "mcp_server",
                    12,
                    attrs={
                        "traceloop.span.kind": "tool",
                        "traceloop.entity.name": "add_numbers",
                        "traceloop.entity.output": '{"result": 3}',
                    },
                ),
            ],
        )
        result = run_dedup(trace)
        assert names(result) == ["add_numbers"]
        # server twin is kept because it carries the richer output
        assert result[0].span_id == "server"

    def test_unpaired_client_is_kept(self):
        trace = OtelTrace(
            trace_id=TID,
            spans=[
                span(
                    "tools/call.tool",
                    "client",
                    None,
                    0,
                    attrs={"traceloop.span.kind": "tool", "traceloop.entity.input.tool_name": "add_numbers"},
                ),
            ],
        )
        result = run_dedup(trace)
        assert names(result) == ["add_numbers"]


# ---------------------------------------------------------------------------
# mcp_tool_two_calls.json: two sequential calls, FOUR raw spans → TWO spans
# ---------------------------------------------------------------------------


class TestMcpTwoCalls:
    def make_trace(self) -> OtelTrace:
        return OtelTrace(
            trace_id=TID,
            spans=[
                span("mcp.client.session", "session", None, 0),
                span("mcp.server", "mcp_server", None, 0, attrs={"traceloop.span.kind": "server"}),
                span("tools/call.tool", "c1", "session", 10, attrs={"traceloop.span.kind": "tool", "traceloop.entity.input.tool_name": "add_numbers"}),
                span("add_numbers.tool", "s1", "mcp_server", 12, attrs={"traceloop.span.kind": "tool", "traceloop.entity.name": "add_numbers"}),
                span("tools/call.tool", "c2", "session", 20, attrs={"traceloop.span.kind": "tool", "traceloop.entity.input.tool_name": "add_numbers"}),
                span("add_numbers.tool", "s2", "mcp_server", 22, attrs={"traceloop.span.kind": "tool", "traceloop.entity.name": "add_numbers"}),
            ],
        )

    def test_two_sequential_calls_remain_two(self):
        """Siblings must NOT be collapsed (retries / parallel same-name)."""
        trace = self.make_trace()
        result = run_dedup(trace)
        assert names(result) == ["add_numbers", "add_numbers"]
        assert [s.span_id for s in result] == ["s1", "s2"]


# ---------------------------------------------------------------------------
# langchain_mcp_wrapper.json: wrapper + client + server → one span
# ---------------------------------------------------------------------------


class TestLangchainMcpWrapper:
    def test_three_raw_spans_to_one(self):
        trace = OtelTrace(
            trace_id=TID,
            spans=[
                span("workflow", "workflow", None, 0, attrs={"traceloop.span.kind": "workflow"}),
                span(
                    "execute_tool add_numbers",
                    "wrapper",
                    "workflow",
                    10,
                    attrs=tool_attr("add_numbers"),
                ),
                span("mcp.client.session", "session", "wrapper", 11),
                span(
                    "tools/call.tool",
                    "client",
                    "session",
                    12,
                    attrs={"traceloop.span.kind": "tool", "traceloop.entity.input.tool_name": "add_numbers"},
                ),
                span("mcp.server", "mcp_server", "wrapper", 13, attrs={"traceloop.span.kind": "server"}),
                span(
                    "add_numbers.tool",
                    "server_leaf",
                    "mcp_server",
                    14,
                    attrs={
                        "traceloop.span.kind": "tool",
                        "traceloop.entity.name": "add_numbers",
                        "traceloop.entity.output": '{"result": 3}',
                    },
                ),
            ],
        )
        result = run_dedup(trace)
        assert names(result) == ["add_numbers"]
        assert result[0].span_id == "server_leaf"


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_parallel_tools_ordered_by_start(self):
        trace = OtelTrace(
            trace_id=TID,
            spans=[
                span("a.tool", "a", None, 30, attrs=tool_attrs("b_tool")),
                span("b.tool", "b", None, 10, attrs=tool_attrs("a_tool")),
            ],
        )
        result = run_dedup(trace)
        assert names(result) == ["a_tool", "b_tool"]

    def test_retries_not_collapsed(self):
        trace = OtelTrace(
            trace_id=TID,
            spans=[
                span("search.tool", "first", None, 10, attrs=tool_attrs("search_emails")),
                span("search.tool", "second", None, 20, attrs=tool_attrs("search_emails")),
            ],
        )
        result = run_dedup(trace)
        assert names(result) == ["search_emails", "search_emails"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def tool_attrs(name: str) -> dict:
    return {"gen_ai.operation.name": "execute_tool", "traceloop.span.kind": "tool", "gen_ai.tool.name": name}