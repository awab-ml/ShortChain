"""Tests for tool-catalog extraction and merge (Section 3.2)."""

from __future__ import annotations

import json

from shortchain.ingest.otel import OtelSpan, OtelTrace
from shortchain.runtime.catalog import (
    build_catalog,
    load_catalog_file,
    merge_catalog,
)

TID = "a" * 32


def make_span(
    span_id: str,
    name: str,
    attrs: dict | None = None,
    start: int = 1,
    end: int = 2,
) -> OtelSpan:
    return OtelSpan(
        trace_id=TID,
        span_id=span_id,
        parent_span_id=None,
        name=name,
        start_time_unix_nano=start,
        end_time_unix_nano=end,
        attributes=attrs or {},
    )


def make_trace(*spans: OtelSpan) -> OtelTrace:
    return OtelTrace(trace_id=TID, spans=list(spans))


def llm_span(span_id: str, attrs: dict) -> OtelSpan:
    attrs.setdefault("gen_ai.operation.name", "chat")
    return make_span(span_id, "chat", attrs)


def tool_span(span_id: str, tool_name: str, attrs: dict | None = None) -> OtelSpan:
    a = {
        "gen_ai.operation.name": "execute_tool",
        "traceloop.span.kind": "tool",
        "gen_ai.tool.name": tool_name,
    }
    a.update(attrs or {})
    return make_span(span_id, f"execute_tool {tool_name}", a)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestDefinitions:
    def test_gen_ai_tool_definitions_json_array(self):
        t = make_trace(
            llm_span(
                "l1",
                {
                    "gen_ai.tool.definitions": json.dumps(
                        [
                            {"name": "lookup_order", "description": "Look up an order", "parameters": {}},
                            {"name": "refund_order", "description": "Refund an order"},
                        ]
                    )
                },
            )
        )
        catalog = build_catalog([t])
        assert catalog["lookup_order"] == "Look up an order"
        assert catalog["refund_order"] == "Refund an order"

    def test_definitions_as_list_not_string(self):
        trace = make_trace(
            llm_span(
                "l1",
                {
                    "gen_ai.tool.definitions": [
                        {"name": "search_emails", "description": "Search email"}
                    ]
                },
            )
        )
        assert build_catalog([trace])["search_emails"] == "Search email"

    def test_legacy_indexed_definitions(self):
        trace = make_trace(
            llm_span(
                "l1",
                {
                    "gen_ai.tool.definitions.0.name": "tool_a",
                    "gen_ai.tool.definitions.0.description": "A tool",
                    "gen_ai.tool.definitions.1.name": "tool_b",
                    "gen_ai.tool.definitions.1.description": "B tool",
                },
            )
        )
        catalog = build_catalog([trace])
        assert catalog == {"tool_a": "A tool", "tool_b": "B tool"}

    def test_legacy_llm_request_functions(self):
        trace = make_trace(
            llm_span(
                "l1",
                {
                    "llm.request.functions.0.name": "fn_a",
                    "llm.request.functions.0.description": "Fn A",
                },
            )
        )
        assert build_catalog([trace])["fn_a"] == "Fn A"


class TestToolSpanDescriptions:
    def test_agno_tool_description(self):
        trace = make_trace(tool_span("t1", "search_catalog", {"tool.description": "Search catalog"}))
        assert build_catalog([trace])["search_catalog"] == "Search catalog"

    def test_gen_ai_tool_description(self):
        trace = make_trace(
            tool_span("t1", "lookup", {"gen_ai.tool.description": "Lookup tool"})
        )
        assert build_catalog([trace])["lookup"] == "Lookup tool"

    def test_called_tool_without_description_gets_empty(self):
        trace = make_trace(tool_span("t1", "plain_call"))
        catalog = build_catalog([trace])
        assert catalog["plain_call"] == ""


class TestMCPListing:
    def test_mcp_tools_listed_names(self):
        trace = make_trace(
            make_span("m1", "mcp_tools", {"mcp.tools.listed": json.dumps(["add_numbers", "sub_numbers"])})
        )
        catalog = build_catalog([trace])
        assert catalog == {"add_numbers": "", "sub_numbers": ""}

    def test_tools_list_output(self):
        trace = make_trace(
            make_span(
                "tl1",
                "tools/list",
                {
                    "traceloop.entity.output": json.dumps(
                        [{"name": "add_numbers", "description": "Add numbers"}]
                    )
                },
            )
        )
        assert build_catalog([trace])["add_numbers"] == "Add numbers"


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------


class TestMerge:
    def test_user_wins(self):
        base = {"tool_a": "from traces", "tool_b": "B"}
        user = {"tool_a": "from user"}
        merged = merge_catalog(base, user)
        assert merged["tool_a"] == "from user"
        assert merged["tool_b"] == "B"

    def test_sorted_keys(self):
        merged = merge_catalog({"b": "B"}, {"a": "A"})
        assert list(merged) == ["a", "b"]

    def test_first_source_wins_within_traces(self):
        t1 = make_trace(
            llm_span("l1", {"gen_ai.tool.definitions": json.dumps([{"name": "x", "description": "first"}])})
        )
        t2 = make_trace(
            llm_span("l1", {"gen_ai.tool.definitions": json.dumps([{"name": "x", "description": "second"}])})
        )
        assert build_catalog([t1, t2])["x"] == "first"


class TestLoadCatalogFile:
    def test_loads_json(self, tmp_path):
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps({"a": "A", "b": "B"}))
        assert load_catalog_file(str(path)) == {"a": "A", "b": "B"}