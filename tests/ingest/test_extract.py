"""Tests for OTEL field extractors (Section 2.3)."""

from __future__ import annotations

import json

from shortchain.config import ProjectionConfig
from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    _find_root,
    extract_app_name,
    extract_intent,
    extract_success,
    extract_task_id,
    extract_tool_arguments,
    extract_tool_observation,
    extract_tool_thoughts,
    nearest_agent_name,
    resolve_success,
)

TID = "a" * 32


def span(
    name: str,
    attrs: dict | None = None,
    span_id: str = "s1",
    parent: str | None = None,
    start: int = 1,
    end: int = 2,
    resource: dict | None = None,
) -> OtelSpan:
    return OtelSpan(
        trace_id=TID,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        start_time_unix_nano=start,
        end_time_unix_nano=end,
        attributes=attrs or {},
        resource=resource or {},
    )


def trace(*spans: OtelSpan) -> OtelTrace:
    return OtelTrace(trace_id=TID, spans=list(spans))


def task_root(
    task_id: str = "ticket-1",
    intent: str | None = "Refund order 9921",
    success: bool = True,
    app_name: str | None = "support-agent",
) -> OtelSpan:
    attrs: dict = {
        "shortchain.task_root": True,
        "traceloop.span.kind": "workflow",
        "shortchain.complete": True,
    }
    if task_id is not None:
        attrs["shortchain.task_id"] = task_id
        attrs["traceloop.association.properties.task_id"] = task_id
    if intent is not None:
        attrs["shortchain.intent"] = intent
        attrs["traceloop.association.properties.intent"] = intent
    if success is not None:
        attrs["shortchain.success"] = success
        attrs["traceloop.association.properties.success"] = success
    if app_name is not None:
        attrs["shortchain.app_name"] = app_name
        attrs["traceloop.association.properties.app_name"] = app_name
    return span("shortchain.task", attrs, span_id="root", start=0, end=100)


def llm_span(
    messages: list,
    output_messages: list | None = None,
    span_id: str = "llm1",
    parent: str = "root",
    start: int = 10,
    end: int = 20,
) -> OtelSpan:
    attrs: dict = {"gen_ai.operation.name": "chat"}
    if messages:
        attrs["gen_ai.input.messages"] = json.dumps(messages)
    if output_messages is not None:
        attrs["gen_ai.output.messages"] = json.dumps(output_messages)
    return span("chat", attrs, span_id=span_id, parent=parent, start=start, end=end)


def tool_span(
    name: str,
    tool_name: str,
    span_id: str = "tool1",
    parent: str = "root",
    start: int = 30,
    end: int = 40,
    result: str | None = None,
    arguments: str | None = None,
    entity_output: str | None = None,
    entity_input: str | None = None,
    input_value: str | None = None,
    output_value: str | None = None,
    agent_name: str | None = None,
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
    if entity_output is not None:
        attrs["traceloop.entity.output"] = entity_output
    if entity_input is not None:
        attrs["traceloop.entity.input"] = entity_input
    if input_value is not None:
        attrs["input.value"] = input_value
    if output_value is not None:
        attrs["output.value"] = output_value
    if agent_name:
        attrs["gen_ai.agent.name"] = agent_name
    return span(name, attrs, span_id=span_id, parent=parent, start=start, end=end)


# ---------------------------------------------------------------------------
# _find_root
# ---------------------------------------------------------------------------


class TestFindRoot:
    def test_prefers_sdk_task_root(self):
        t = trace(
            span("workflow", {"traceloop.span.kind": "workflow"}, span_id="w", start=0),
            task_root(),
        )
        assert _find_root(t).span_id == "root"

    def test_outermost_agent_span(self):
        inner = span(
            "agent", {"traceloop.span.kind": "agent"}, span_id="inner", parent="outer"
        )
        outer = span(
            "agent", {"traceloop.span.kind": "agent"}, span_id="outer", start=0
        )
        assert _find_root(trace(inner, outer)).span_id == "outer"

    def test_empty_parent_fallback(self):
        t = trace(
            span("chat", {"gen_ai.operation.name": "chat"}, span_id="c", parent="p"),
            span("http", {}, span_id="p", start=5),
        )
        assert _find_root(t).span_id == "p"

    def test_empty_trace(self):
        assert _find_root(trace()) is None


# ---------------------------------------------------------------------------
# task_id
# ---------------------------------------------------------------------------


class TestTaskId:
    def test_association_first(self):
        t = trace(task_root(task_id="t-42"))
        assert extract_task_id(t, ProjectionConfig(), _find_root(t)) == "t-42"

    def test_conversation_id(self):
        t = trace(
            span("chat", {"gen_ai.operation.name": "chat", "gen_ai.conversation.id": "thread-7"})
        )
        assert extract_task_id(t, ProjectionConfig(), None) == "thread-7"

    def test_falls_back_to_trace_id(self):
        t = trace(span("chat", {"gen_ai.operation.name": "chat"}))
        assert extract_task_id(t, ProjectionConfig(), None) == TID

    def test_gen_ai_task_id_requires_opt_in(self):
        root = span(
            "workflow",
            {"traceloop.span.kind": "workflow", "gen_ai.task.id": "run-9"},
            span_id="w",
        )
        t = trace(root)
        assert extract_task_id(t, ProjectionConfig(), root) == TID
        cfg = ProjectionConfig(accept_gen_ai_task_id=True)
        assert extract_task_id(t, cfg, root) == "run-9"


# ---------------------------------------------------------------------------
# intent
# ---------------------------------------------------------------------------


class TestIntent:
    def test_association_first(self):
        t = trace(task_root(intent="Please refund"))
        assert extract_intent(t, ProjectionConfig(), _find_root(t)) == (
            "Please refund",
            "association",
        )

    def test_gen_ai_task_input_unwrap(self):
        root = span(
            "workflow",
            {
                "traceloop.span.kind": "workflow",
                "gen_ai.task.input": '{"inputs": "Refund order 9921"}',
            },
            span_id="w",
        )
        t = trace(root)
        assert extract_intent(t, ProjectionConfig(), root) == (
            "Refund order 9921",
            "task_input",
        )

    def test_first_user_message(self):
        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "Refund order 9921"},
        ]
        t = trace(task_root(success=True, intent=None), llm_span(messages))
        assert extract_intent(t, ProjectionConfig(), _find_root(t)) == (
            "Refund order 9921",
            "user_message",
        )

    def test_first_user_message_parts_style(self):
        messages = [
            {"role": "user", "parts": [{"type": "text", "content": "Book a flight"}]},
        ]
        t = trace(task_root(success=True, intent=None), llm_span(messages))
        assert extract_intent(t, ProjectionConfig(), _find_root(t)) == (
            "Book a flight",
            "user_message",
        )

    def test_last_user_before_tools_strategy(self):
        messages = [
            {"role": "user", "content": "You are a long system preamble. " * 100},
            {"role": "user", "content": "The real task: send payment"},
            {
                "role": "assistant",
                "parts": [
                    {"type": "tool_call", "id": "call_1", "name": "pay", "arguments": "{}"}
                ],
            },
        ]
        cfg = ProjectionConfig(intent_strategy="last_user_before_tools")
        t = trace(task_root(success=True, intent=None), llm_span(messages))
        assert extract_intent(t, cfg, _find_root(t)) == (
            "The real task: send payment",
            "last_user_before_tools",
        )

    def test_missing_intent(self):
        t = trace(llm_span([{"role": "assistant", "parts": []}]))
        assert extract_intent(t, ProjectionConfig(), None) == ("", "")


# ---------------------------------------------------------------------------
# success
# ---------------------------------------------------------------------------


class TestSuccess:
    def test_extract_success_bool(self):
        assert extract_success(task_root(success=True)) is True
        assert extract_success(task_root(success=False)) is False

    def test_extract_success_absent(self):
        assert extract_success(span("chat", {"gen_ai.operation.name": "chat"})) is None

    def test_extract_success_string_form(self):
        s = span("x", {"shortchain.success": "false"})
        assert extract_success(s) is False
        s = span("x", {"traceloop.association.properties.success": 1})
        assert extract_success(s) is True

    def test_resolve_association(self):
        t = trace(task_root(success=True))
        assert resolve_success(t, ProjectionConfig(), _find_root(t)) == (True, "association")

    def test_resolve_false_is_present_value(self):
        t = trace(task_root(success=False))
        assert resolve_success(t, ProjectionConfig(), _find_root(t)) == (False, "association")

    def test_resolve_unknown(self):
        t = trace(span("x", {"traceloop.span.kind": "workflow"}))
        assert resolve_success(t, ProjectionConfig(), None) == (False, "unknown")

    def test_task_status_requires_opt_in(self):
        root = span(
            "workflow",
            {"traceloop.span.kind": "workflow", "gen_ai.task.status": "failure"},
            span_id="w",
        )
        t = trace(root)
        assert resolve_success(t, ProjectionConfig(), root) == (False, "unknown")
        cfg = ProjectionConfig(accept_task_status=True)
        assert resolve_success(t, cfg, root) == (False, "task_status")

    def test_success_tools_heuristic(self):
        t = trace(tool_span("done.tool", "done", start=1, end=2))
        cfg = ProjectionConfig(success_tools=["done"])
        assert resolve_success(t, cfg, None) == (True, "heuristic")
        cfg2 = ProjectionConfig(success_tools=["other"])
        assert resolve_success(t, cfg2, None) == (False, "heuristic")


# ---------------------------------------------------------------------------
# app_name / agent_name
# ---------------------------------------------------------------------------


class TestAppName:
    def test_association(self):
        t = trace(task_root(app_name="billing"))
        assert extract_app_name(t, _find_root(t)) == "billing"

    def test_resource_service_name(self):
        root = span(
            "workflow",
            {"traceloop.span.kind": "workflow"},
            span_id="w",
            resource={"service.name": "my-service"},
        )
        assert extract_app_name(trace(root), root) == "my-service"

    def test_workflow_name_fallback(self):
        root = span(
            "workflow",
            {"traceloop.span.kind": "workflow", "traceloop.workflow.name": "OrdersAgent"},
            span_id="w",
        )
        assert extract_app_name(trace(root), root) == "OrdersAgent"

    def test_empty(self):
        assert extract_app_name(trace(span("x", {})), None) == ""


class TestAgentName:
    def test_inherits_from_ancestor(self):
        agent = span(
            "agent",
            {"traceloop.span.kind": "agent", "gen_ai.agent.name": "SupportAgent"},
            span_id="root",
        )
        tool = tool_span("tool.tool", "tool", span_id="t", parent="root")
        by_id = {s.span_id: s for s in (agent, tool)}
        assert nearest_agent_name(tool, by_id) == "SupportAgent"

    def test_own_agent_name(self):
        tool = tool_span("tool.tool", "tool", agent_name="DirectAgent")
        assert nearest_agent_name(tool, {"t": tool}) == "DirectAgent"

    def test_empty(self):
        tool = tool_span("tool.tool", "tool")
        assert nearest_agent_name(tool, {}) == ""


# ---------------------------------------------------------------------------
# arguments / observation / thoughts
# ---------------------------------------------------------------------------


class TestToolFields:
    def test_arguments_json_blob(self):
        t = tool_span("t.tool", "t", arguments='{"order_id": 9921}')
        assert "9921" in extract_tool_arguments(t)

    def test_arguments_entity_input(self):
        t = tool_span("t.tool", "t", entity_input='{"arguments": "a=1"}')
        assert extract_tool_arguments(t) == "a=1"

    def test_arguments_openinference(self):
        t = tool_span("function.t", "t", input_value='{"q": "x"}')
        assert extract_tool_arguments(t) == '{"q": "x"}'

    def test_arguments_empty(self):
        assert extract_tool_arguments(tool_span("t.tool", "t")) == ""

    def test_observation_priority(self):
        t = tool_span("t.tool", "t", result="gen_ai result", entity_output="entity out")
        assert extract_tool_observation(t) == "gen_ai result"

    def test_observation_entity_output(self):
        t = tool_span("t.tool", "t", entity_output='{"result": 3}')
        assert extract_tool_observation(t) == '{"result": 3}'

    def test_observation_openinference(self):
        t = tool_span("function.t", "t", output_value="out")
        assert extract_tool_observation(t) == "out"

    def test_observation_truncated(self):
        t = tool_span("t.tool", "t", result="x" * 5000)
        assert len(extract_tool_observation(t, max_chars=2000)) == 2000

    def test_observation_empty(self):
        assert extract_tool_observation(tool_span("t.tool", "t")) == ""

    def test_thoughts_from_preceding_llm(self):
        out = [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "I should look first"},
                    {"type": "tool_call", "id": "c1", "name": "look", "arguments": "{}"},
                ],
            },
        ]
        t = trace(
            task_root(),
            llm_span([{"role": "user", "content": "hi"}], output_messages=out),
            tool_span("look.tool", "look"),
        )
        assert extract_tool_thoughts(t, t.spans[-1]) == "I should look first"

    def test_thoughts_empty(self):
        t = trace(tool_span("look.tool", "look"))
        assert extract_tool_thoughts(t, t.spans[0]) == ""

    def test_thoughts_completion_fallback(self):
        llm = span(
            "chat",
            {"gen_ai.operation.name": "chat", "gen_ai.completion": "Hmm"},
            span_id="l",
            start=10,
            end=20,
        )
        tool = tool_span("look.tool", "look", start=30, end=40)
        assert extract_tool_thoughts(trace(llm, tool), tool) == "Hmm"