"""Tests for OtelTraceProjector orchestration (Section 2.6)."""

from __future__ import annotations

import json

from shortchain.config import ProjectionConfig
from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    OtelTraceProjector,
)

TID = "a" * 32


def make_span(**kw) -> OtelSpan:
    defaults = dict(
        trace_id=TID,
        span_id="s1",
        parent_span_id=None,
        name="unnamed",
        start_time_unix_nano=1,
        end_time_unix_nano=2,
        attributes={},
        resource={},
    )
    defaults.update(kw)
    return OtelSpan(**defaults)


def trace(*spans: OtelSpan, reason: str = "explicit") -> OtelTrace:
    return OtelTrace(trace_id=TID, spans=list(spans), complete_reason=reason)


def task_root(
    success: bool = True,
    intent: str | None = "Refund order 9921",
    task_id: str = "ticket-1842",
) -> OtelSpan:
    attrs: dict = {
        "shortchain.task_root": True,
        "traceloop.span.kind": "workflow",
        "shortchain.task_id": task_id,
        "shortchain.complete": True,
    }
    if intent is not None:
        attrs["shortchain.intent"] = intent
    if success is not None:
        attrs["shortchain.success"] = success
    return make_span(
        name="shortchain.task",
        span_id="root",
        parent_span_id=None,
        start_time_unix_nano=0,
        end_time_unix_nano=100,
        attributes=attrs,
        resource={"service.name": "support-agent"},
    )


def llm(messages: list | None = None, span_id: str = "llm1") -> OtelSpan:
    attrs: dict = {"gen_ai.operation.name": "chat"}
    if messages is not None:
        attrs["gen_ai.input.messages"] = json.dumps(messages)
    return make_span(
        name="chat",
        span_id=span_id,
        parent_span_id="root",
        start_time_unix_nano=10,
        end_time_unix_nano=20,
        attributes=attrs,
    )


def tool(
    name: str,
    tool_name: str,
    span_id: str = "tool1",
    start: int = 30,
    end: int = 40,
    result: str | None = None,
    arguments: str | None = None,
    **extra_attrs,
) -> OtelSpan:
    attrs: dict = {
        "gen_ai.operation.name": "execute_tool",
        "traceloop.span.kind": "tool",
        "gen_ai.tool.name": tool_name,
    }
    if result is not None:
        attrs["gen_ai.tool.call.result"] = result
    if arguments is not None:
        attrs["gen_ai.tool.call.arguments"] = arguments
    attrs.update(extra_attrs)
    return make_span(
        name=name,
        span_id=span_id,
        parent_span_id="root",
        start_time_unix_nano=start,
        end_time_unix_nano=end,
        attributes=attrs,
    )


def project(t: OtelTrace, *, cfg: ProjectionConfig | None = None):
    return OtelTraceProjector(cfg).project(t)


MESSAGES = [
    {"role": "user", "content": "Refund order 9921 and email the customer"},
]


# ---------------------------------------------------------------------------
# Happy path: SDK root + LangChain tool span
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_basic_langchain_trace(self):
        t = trace(
            task_root(),
            llm(messages=MESSAGES),
            tool(
                "execute_tool lookup_order",
                "lookup_order",
                result='{"status": "delivered"}',
                arguments='{"order_id": 9921}',
            ),
        )
        result = project(t)
        assert result.drop_reason is None
        traj = result.trajectory
        assert traj is not None
        assert traj.task_id == "ticket-1842"
        assert traj.intent == "Refund order 9921"
        assert traj.success is True
        assert traj.app_name == "support-agent"
        assert len(traj.spans) == 1
        s = traj.spans[0]
        assert s.action == "lookup_order"
        assert s.observation == '{"status": "delivered"}'
        assert s.metadata["otel.span_id"] == "tool1"
        assert s.metadata["tool_arguments"] == '{"order_id": 9921}'
        assert traj.metadata["success_source"] == "association"
        assert traj.metadata["source"] == "otel_openllmetry"
        assert traj.metadata["otel.trace_id"] == TID

    def test_tools_used_derived(self):
        t = trace(
            task_root(),
            tool("execute_tool a", "tool_a"),
            tool("execute_tool b", "tool_b"),
        )
        traj = project(t).trajectory
        assert traj.tools_used == {"tool_a", "tool_b"}
        assert traj.tool_sequence == ["tool_a", "tool_b"]

    def test_parallel_tools_ordered_by_start(self):
        t = trace(
            task_root(),
            tool("execute_tool b", "tool_b", start=40, end=50, span_id="tb"),
            tool("execute_tool a", "tool_a", start=30, end=35, span_id="ta"),
        )
        traj = project(t).trajectory
        assert traj.tool_sequence == ["tool_a", "tool_b"]

    def test_retries_kept_as_two_spans(self):
        t = trace(
            task_root(),
            tool("execute_tool search", "search_emails", start=10, end=15, span_id="r1"),
            tool("execute_tool search", "search_emails", start=20, end=25, span_id="r2"),
        )
        traj = project(t).trajectory
        assert [s.metadata["otel.span_id"] for s in traj.spans] == ["r1", "r2"]


# ---------------------------------------------------------------------------
# Projector-side quality gates (full quality module lives in ingest/quality)
# ---------------------------------------------------------------------------


class TestQualityGates:
    def test_missing_intent_drop(self):
        t = trace(
            task_root(intent=None),
            llm(messages=[{"role": "assistant", "parts": []}]),
            tool("execute_tool look", "look"),
        )
        result = project(t)
        assert result.trajectory is None
        assert result.drop_reason == "missing_intent"

    def test_missing_intent_tolerated_when_flagged_off(self):
        t = trace(
            task_root(intent=None),
            llm(messages=[{"role": "assistant", "parts": []}]),
            tool("execute_tool look", "look"),
        )
        cfg = ProjectionConfig(require_intent=False)
        traj = project(t, cfg=cfg).trajectory
        assert traj is not None and traj.intent == ""

    def test_success_unknown_drop(self):
        t = trace(llm(messages=MESSAGES), tool("execute_tool look", "look"))
        result = project(t)
        assert result.drop_reason == "success_unknown"

    def test_success_unknown_allowed_when_flagged_off(self):
        t = trace(llm(messages=MESSAGES), tool("execute_tool look", "look"))
        cfg = ProjectionConfig(require_known_success=False)
        traj = project(t, cfg=cfg).trajectory
        assert traj is not None
        assert traj.success is False
        assert traj.metadata["success_source"] == "unknown"

    def test_zero_tool_spans_drop(self):
        t = trace(task_root(), llm(messages=MESSAGES))
        result = project(t)
        assert result.drop_reason == "zero_tool_spans"

    def test_nameless_mcp_tools_do_not_become_spans(self):
        t = trace(
            task_root(),
            make_span(
                name="mcp_tools",
                span_id="t1",
                parent_span_id="root",
                start_time_unix_nano=30,
                end_time_unix_nano=40,
                attributes={"openinference.span.kind": "TOOL"},
            ),
        )
        result = project(t)
        assert result.drop_reason == "zero_tool_spans"

    def test_success_false_is_not_success_unknown(self):
        t = trace(
            task_root(success=False),
            tool("execute_tool look", "look"),
        )
        result = project(t)
        assert result.drop_reason is None
        assert result.trajectory.success is False
        assert result.trajectory.metadata["success_source"] == "association"

    def test_warnings_are_reported(self):
        t = trace(task_root(), tool("execute_tool look", "look"))
        result = project(t)
        assert isinstance(result.warnings, list)


# ---------------------------------------------------------------------------
# drop_tools / metadata
# ---------------------------------------------------------------------------


class TestDropTools:
    def test_drop_tools_filtered(self):
        t = trace(
            task_root(),
            tool("execute_tool keep_me", "keep_me"),
            tool("execute_tool drop_me", "drop_me"),
        )
        cfg = ProjectionConfig(drop_tools=["drop_me"])
        traj = project(t, cfg=cfg).trajectory
        assert traj.tool_sequence == ["keep_me"]


class TestMetadata:
    def test_token_sums(self):
        t = trace(
            task_root(),
            tool(
                "execute_tool look",
                "look",
                **{
                    "gen_ai.usage.input_tokens": 10,
                    "gen_ai.usage.output_tokens": 5,
                    "gen_ai.usage.total_tokens": 15,
                },
            ),
        )
        meta = project(t).trajectory.metadata
        assert meta["tokens.input_sum"] == 10
        assert meta["tokens.output_sum"] == 5
        assert meta["tokens.total_sum"] == 15

    def test_service_name_in_metadata(self):
        t = trace(task_root(), tool("execute_tool look", "look"))
        assert project(t).trajectory.metadata["service.name"] == "support-agent"

    def test_conversation_id_in_metadata(self):
        t = trace(
            task_root(),
            make_span(
                name="chat",
                span_id="l",
                parent_span_id="root",
                start_time_unix_nano=10,
                end_time_unix_nano=20,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.conversation.id": "thread-9",
                },
            ),
            tool("execute_tool look", "look"),
        )
        assert project(t).trajectory.metadata["gen_ai.conversation.id"] == "thread-9"


# ---------------------------------------------------------------------------
# CrewAI fallback wiring
# ---------------------------------------------------------------------------


class TestCrewAIFallback:
    def test_no_tool_spans_uses_llm_fallback(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "parts": [
                    {
                        "id": "c1",
                        "type": "tool_call",
                        "name": "search_emails",
                        "arguments": '{"q": "invoices"}',
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "2 invoices"},
        ]
        t = trace(
            task_root(),
            llm(messages=msgs),
        )
        traj = project(t).trajectory
        assert traj is not None
        assert [s.action for s in traj.spans] == ["search_emails"]
        assert traj.spans[0].observation == "2 invoices"
        assert traj.metadata["projection.fallback"] == "llm_tool_calls"
        assert traj.spans[0].metadata["projection.role"] == "llm_fallback"