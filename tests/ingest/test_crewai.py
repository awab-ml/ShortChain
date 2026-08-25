"""Tests for the CrewAI / raw-OpenAI LLM message-walk fallback (Section 2.5)."""

from __future__ import annotations

import json

from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    extract_llm_tool_calls,
)

TID = "a" * 32


def span(
    name: str,
    span_id: str,
    start: int,
    end: int,
    attrs: dict,
) -> OtelSpan:
    return OtelSpan(
        trace_id=TID,
        span_id=span_id,
        parent_span_id=None,
        name=name,
        start_time_unix_nano=start,
        end_time_unix_nano=end,
        attributes=attrs,
    )


def llm_span(span_id: str, start: int, end: int, messages: list, attrs: dict | None = None) -> OtelSpan:
    a = {"gen_ai.operation.name": "chat", "gen_ai.input.messages": json.dumps(messages)}
    a.update(attrs or {})
    return span("chat", span_id, start, end, a)


def tool_call(call_id: str, name: str, arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "tool_call", "name": name, "arguments": arguments}


def trace(*spans: OtelSpan) -> OtelTrace:
    return OtelTrace(trace_id=TID, spans=list(spans))


# ---------------------------------------------------------------------------
# Two-turn fixture: same call_id in both inputs → ONE decision
# ---------------------------------------------------------------------------


class TestContractTwoTurns:
    def test_same_call_id_in_second_input_yields_one_span(self):
        first_turn = [
            {"role": "user", "content": "Find invoices"},
            {
                "role": "assistant",
                "parts": [tool_call("call_1", "search_emails", '{"q": "invoices"}')],
            },
        ]
        second_turn = [
            {"role": "user", "content": "Find invoices"},
            {
                "role": "assistant",
                "parts": [tool_call("call_1", "search_emails", '{"q": "invoices"}')],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2 invoices"},
        ]
        t = trace(
            llm_span("llm1", 10, 20, first_turn),
            llm_span("llm2", 30, 40, second_turn),
        )
        decisions = extract_llm_tool_calls(t)
        assert len(decisions) == 1
        name, args, obs = decisions[0]
        assert name == "search_emails"
        assert "invoices" in args
        assert obs == "2 invoices"

    def test_latest_span_snapshot_used(self):
        """A call only on the FIRST span's input is not re-emitted.

        The latest LLM span is the conversation snapshot (HALO last-span);
        it is NOT unioned with earlier spans' tool calls. This is what
        prevents the historical-call duplication bug.
        """
        first_only = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "parts": [tool_call("c9", "first_tool")]},
        ]
        later = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "parts": []},
        ]
        t = trace(
            llm_span("llm1", 10, 20, first_only),
            llm_span("llm2", 30, 40, later),
        )
        assert extract_llm_tool_calls(t) == []


class TestIdDedup:
    def test_same_id_twice_in_one_snapshot(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "parts": [tool_call("a", "x_tool")]},
            {"role": "assistant", "parts": [tool_call("a", "x_tool")]},
        ]
        t = trace(llm_span("l", 10, 20, msgs))
        assert len(extract_llm_tool_calls(t)) == 1

    def test_distinct_ids_kept(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "parts": [
                    tool_call("a", "x_tool"),
                    tool_call("b", "y_tool"),
                ],
            },
        ]
        t = trace(llm_span("s", 10, 20, msgs))
        decisions = extract_llm_tool_calls(t)
        assert [d[0] for d in decisions] == ["x_tool", "y_tool"]

    def test_unpaired_call_gets_empty_observation(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "parts": [tool_call("c9", "alone_tool")]},
        ]
        t = trace(llm_span("s", 10, 20, msgs))
        assert extract_llm_tool_calls(t) == [("alone_tool", "{}", "")]

    def test_missing_id_keys_on_name_args_ordinal(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "parts": [
                    tool_call("", "dup_tool", '{"n": 1}'),
                    tool_call("", "dup_tool", '{"n": 1}'),
                    tool_call("", "dup_tool", '{"n": 2}'),
                ],
            },
        ]
        t = trace(llm_span("s", 10, 20, msgs))
        decisions = extract_llm_tool_calls(t)
        assert len(decisions) == 2  # two identical calls collapse, distinct args stay


class TestResponsePairing:
    def test_response_paired_by_id(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "parts": [tool_call("r1", "get_data")]},
            {"role": "tool", "tool_call_id": "r1", "content": "payload"},
        ]
        t = trace(llm_span("s", 10, 20, msgs))
        assert extract_llm_tool_calls(t) == [("get_data", "{}", "payload")]

    def test_llm_with_no_input_messages(self):
        t = trace(llm_span("s", 10, 20, []))
        assert extract_llm_tool_calls(t) == []

    def test_no_llm_spans(self):
        t = OtelTrace(trace_id=TID, spans=[])
        assert extract_llm_tool_calls(t) == []


class TestOutputMerge:
    def test_output_tool_calls_merged_by_id(self):
        msgs = [{"role": "user", "content": "hi"}]
        output = [
            {"role": "assistant", "parts": [tool_call("out1", "fresh_tool")]},
        ]
        t = trace(
            llm_span("s", 10, 20, msgs, attrs={"gen_ai.output.messages": json.dumps(output)})
        )
        decisions = extract_llm_tool_calls(t)
        assert [d[0] for d in decisions] == ["fresh_tool"]

    def test_output_call_already_in_input_not_duplicated(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "parts": [tool_call("d1", "seen_tool")]},
        ]
        output = [
            {"role": "assistant", "parts": [tool_call("d1", "seen_tool")]},
        ]
        t = trace(
            llm_span(
                "s",
                10,
                20,
                msgs,
                attrs={"gen_ai.output.messages": json.dumps(output)},
            )
        )
        assert len(extract_llm_tool_calls(t)) == 1