"""Tests for HALO/OpenInference AppWorld trace ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shortchain.adapters.halo import (
    app_of,
    build_trajectory_from_rows,
    catalog_app_index,
    is_control_tool,
    load_appworld_traces,
    reconstruct_catalog,
)


def _msg(role: str, content: str, tool_calls=None) -> dict:
    m = {"role": role, "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m


def _tool_call(name: str, args: str = "{}") -> dict:
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": args}}


def _llm_span(messages: list[dict]) -> dict:
    return {
        "trace_id": "trace_abc",
        "span_id": "llm",
        "name": "generation.gemini",
        "kind": "SPAN_KIND_CLIENT",
        "attributes": {
            "openinference.span.kind": "LLM",
            "llm.input_messages": json.dumps(messages),
        },
    }


def _tool_span(top: str, tool: str, out: str = "{}") -> dict:
    return {
        "trace_id": "trace_abc",
        "span_id": f"t_{tool}",
        "name": f"function.{tool}",
        "kind": "SPAN_KIND_INTERNAL",
        "attributes": {
            "openinference.span.kind": "TOOL",
            "tool.name": tool,
            "mcp.tools.listed": json.dumps(
                ["supervisor__complete_task", "spotify__login", "spotify__search_songs",
                 "phone__search_contacts", "venmo__login"]
            ),
            "input.value": top,
            "output.value": out,
        },
    }


@pytest.fixture
def halo_trace(tmp_path: Path):
    """One trace: profile preamble (which contains the word 'Real Task'), a
    tutorial task, and a real task — mirroring the real export shape."""
    profile = (
        "My name is: Glenn Burton.\n"
        "You will be given a task instruction and a list of functions in the standard format. "
        "You will complete the real task instruction completely autonomously..."
    )
    messages = [
        _msg("system", "You are a supervisor agent."),
        _msg("user", profile),
        _msg("assistant", "Sounds good!"),
        _msg("user", "# Tutorial Task Instruction 1\nHow many songs are in my queue?"),
        _msg("assistant", "", tool_calls=[_tool_call("spotify__show_song_queue")]),
        _msg("tool", "{\"songs\": 2}"),
        _msg("user", "# Real Task Instruction\nTerminate my Venmo account then list contacts."),
        _msg("assistant", "", tool_calls=[_tool_call("venmo__login"), _tool_call("phone__search_contacts")]),
        _msg("tool", "{\"login\": true}"),
        _msg("tool", "{\"contacts\": []}"),
        _msg("assistant", ""),
        _msg("assistant", "", tool_calls=[_tool_call("supervisor__complete_task")]),
        _msg("tool", "{\"msg\": \"complete\"}"),
    ]
    rows = [
        _llm_span(messages),
        _tool_span("{}", "spotify__show_song_queue"),
        _tool_span("{}", "venmo__login"),
        _tool_span("{}", "phone__search_contacts"),
        _tool_span("{}", "supervisor__complete_task"),
    ]
    p = tmp_path / "trace.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


class TestHaloBuild:
    def test_fixture_like_real_data(self, halo_trace):
        trajs = load_appworld_traces(halo_trace)
        assert len(trajs) == 1
        t = trajs[0]
        # intent = the REAL task text, not the profile or the tutorial.
        assert t.intent.startswith("Terminate my Venmo")
        assert "My name is" not in t.intent
        # supervisor excluded; tutorial queue tool excluded.
        assert "supervisor__complete_task" not in t.tools_used
        assert "spotify__show_song_queue" not in t.tools_used
        assert t.tools_used == {"venmo__login", "phone__search_contacts"}
        assert t.app_name == "venmo"
        assert t.success is True  # exported flat list ends with complete_task
        assert t.metadata["apps"] == ["phone", "venmo"]

    def test_catalog_reconstruction(self, halo_trace):
        cat = reconstruct_catalog(halo_trace)
        assert "supervisor__complete_task" not in cat
        assert "spotify__login" in cat and "venmo__login" in cat
        idx = catalog_app_index(cat)
        assert "venmo" in idx and "spotify" in idx

    def test_helpers(self):
        assert app_of("phone__search_contacts") == "phone"
        assert is_control_tool("supervisor__complete_task") is True
        assert is_control_tool("spotify__login") is False

    def test_no_real_task_falls_back_to_last_user(self, tmp_path):
        messages = [
            _msg("system", "sys"),
            _msg("user", "Simple instruction without header"),
            _msg("assistant", "", tool_calls=[_tool_call("venmo__login")]),
            _msg("tool", "{}"),
        ]
        rows = [
            _llm_span(messages),
            _tool_span("{}", "venmo__login"),
            _tool_span("{}", "supervisor__complete_task"),
        ]
        t = build_trajectory_from_rows(rows)
        assert t is not None
        assert t.intent == "Simple instruction without header"
        assert t.tools_used == {"venmo__login"}
        assert t.success is True
