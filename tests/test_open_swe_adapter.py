"""Unit tests for the Open-SWE-Traces adapter logic.

Tests the conversion functions in scripts/fetch_hf_traces.py to verify
that raw HuggingFace records are correctly mapped to ShortChain Trajectory
and Span objects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path so we can import fetch_hf_traces
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fetch_hf_traces import (
    convert_trajectory,
    _extract_tool_name,
    _extract_tool_name_from_call,
)
from shortchain.ingest.schema import Span, Trajectory


# ---------------------------------------------------------------------------
# Fixtures — realistic Open-SWE-Traces records
# ---------------------------------------------------------------------------

@pytest.fixture
def swe_agent_record() -> dict:
    """A minimal record mimicking SWE-agent style with tool_calls."""
    return {
        "instance_id": "django__django-15814",
        "repo": "django/django",
        "license": "BSD-3-Clause",
        "language": "Python",
        "trajectory_id": "traj_001",
        "trajectory": [
            {"role": "system", "content": "You are an autonomous agent..."},
            {"role": "user", "content": "Fix the QuerySet.only() method..."},
            {
                "role": "assistant",
                "content": "I need to understand the issue first.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "find . -name models.py"}'
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "./django/db/models/query.py\n./tests/model_fields/models.py",
            },
            {
                "role": "assistant",
                "content": "Let me look at the query.py file.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": '{"command": "view", "path": "django/db/models/query.py"}'
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "class QuerySet:\n    def only(self, *fields):\n        ...",
            },
            {
                "role": "assistant",
                "content": "Now I'll apply the fix.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": '{"command": "str_replace", "old_str": "..."}'
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "The file has been edited.",
            },
        ],
        "tools": [
            json.dumps({"function": {"name": "bash", "description": "Run a bash command"}}),
            json.dumps({"function": {"name": "str_replace_editor", "description": "Edit files"}}),
        ],
        "resolved": 1,
        "metadata": {
            "category": "bug-fix",
            "reference_patch": {
                "patch": "diff --git a/...",
                "num_modified_files": 1,
            },
        },
    }


@pytest.fixture
def openhands_record() -> dict:
    """A record mimicking OpenHands-style with XML function calls."""
    return {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "license": "BSD-3-Clause",
        "language": "Python",
        "trajectory_id": "traj_002",
        "trajectory": [
            {"role": "user", "content": "Fix the compound model inversion..."},
            {
                "role": "assistant",
                "content": "Let me search for the relevant code. <function=bash>{\"command\": \"grep -r 'inverse' astropy/modeling/\"}</function>",
            },
            {
                "role": "tool",
                "content": "astropy/modeling/core.py:    def inverse(self):",
            },
            {
                "role": "assistant",
                "content": "Now viewing the file. <function=view_file>{\"path\": \"astropy/modeling/core.py\"}</function>",
            },
            {
                "role": "tool",
                "content": "class CompoundModel:\n    ...",
            },
        ],
        "tools": [],
        "resolved": 0,
        "metadata": {"category": "feature-request"},
    }


@pytest.fixture
def empty_trajectory_record() -> dict:
    """A record with no trajectory messages."""
    return {
        "instance_id": "empty__empty-001",
        "repo": "empty/empty",
        "trajectory": [],
        "tools": [],
        "resolved": -1,
        "metadata": {},
    }


@pytest.fixture
def no_user_message_record() -> dict:
    """A record with trajectory but no user message (no intent)."""
    return {
        "instance_id": "no_intent__test-001",
        "repo": "test/test",
        "trajectory": [
            {"role": "system", "content": "System prompt..."},
            {
                "role": "assistant",
                "content": "Doing something.",
                "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "output"},
        ],
        "tools": [],
        "resolved": 1,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Tests — _extract_tool_name
# ---------------------------------------------------------------------------

class TestExtractToolName:
    """Test tool name extraction from message content."""

    def test_xml_function_call(self) -> None:
        content = 'Let me run this. <function=bash>{"command": "ls -la"}</function>'
        assert _extract_tool_name(content) == "bash"

    def test_json_name_field(self) -> None:
        content = 'Using tool: {"name": "str_replace_editor", "args": {}}'
        assert _extract_tool_name(content) == "str_replace_editor"

    def test_no_tool_in_content(self) -> None:
        content = "I need to think about this problem carefully."
        assert _extract_tool_name(content) is None

    def test_empty_content(self) -> None:
        assert _extract_tool_name("") is None

    def test_none_content(self) -> None:
        assert _extract_tool_name(None) is None


# ---------------------------------------------------------------------------
# Tests — _extract_tool_name_from_call
# ---------------------------------------------------------------------------

class TestExtractToolNameFromCall:
    """Test tool name extraction from structured tool_calls."""

    def test_openai_format(self) -> None:
        tool_calls = [{"function": {"name": "bash", "arguments": "{}"}}]
        assert _extract_tool_name_from_call(tool_calls) == "bash"

    def test_direct_name(self) -> None:
        tool_calls = [{"name": "view_file", "args": {}}]
        assert _extract_tool_name_from_call(tool_calls) == "view_file"

    def test_empty_list(self) -> None:
        assert _extract_tool_name_from_call([]) is None

    def test_none(self) -> None:
        assert _extract_tool_name_from_call(None) is None


# ---------------------------------------------------------------------------
# Tests — convert_trajectory
# ---------------------------------------------------------------------------

class TestConvertTrajectory:
    """Test full trajectory conversion."""

    def test_swe_agent_conversion(self, swe_agent_record: dict) -> None:
        result = convert_trajectory(swe_agent_record)
        assert result is not None
        assert result["task_id"] == "django__django-15814"
        assert result["app_name"] == "django/django"
        assert result["success"] is True

    def test_swe_agent_has_spans(self, swe_agent_record: dict) -> None:
        result = convert_trajectory(swe_agent_record)
        assert result is not None
        assert len(result["spans"]) == 3
        actions = [s["action"] for s in result["spans"]]
        assert "bash" in actions
        assert "str_replace_editor" in actions

    def test_swe_agent_intent(self, swe_agent_record: dict) -> None:
        result = convert_trajectory(swe_agent_record)
        assert result is not None
        assert "QuerySet.only()" in result["intent"]

    def test_swe_agent_observations(self, swe_agent_record: dict) -> None:
        result = convert_trajectory(swe_agent_record)
        assert result is not None
        # First tool call should have observation
        assert result["spans"][0]["observation"] is not None
        assert "models.py" in result["spans"][0]["observation"]

    def test_swe_agent_thoughts(self, swe_agent_record: dict) -> None:
        result = convert_trajectory(swe_agent_record)
        assert result is not None
        assert result["spans"][0]["thoughts"] is not None
        assert "understand" in result["spans"][0]["thoughts"].lower()

    def test_swe_agent_tools_used(self, swe_agent_record: dict) -> None:
        result = convert_trajectory(swe_agent_record)
        assert result is not None
        assert set(result["tools_used"]) == {"bash", "str_replace_editor"}

    def test_swe_agent_metadata(self, swe_agent_record: dict) -> None:
        result = convert_trajectory(swe_agent_record)
        assert result is not None
        assert result["metadata"]["repo"] == "django/django"
        assert result["metadata"]["language"] == "Python"
        assert result["metadata"]["category"] == "bug-fix"

    def test_swe_agent_available_tools(self, swe_agent_record: dict) -> None:
        result = convert_trajectory(swe_agent_record)
        assert result is not None
        assert "bash" in result["metadata"]["available_tools"]
        assert "str_replace_editor" in result["metadata"]["available_tools"]

    def test_openhands_conversion(self, openhands_record: dict) -> None:
        result = convert_trajectory(openhands_record)
        assert result is not None
        assert result["task_id"] == "astropy__astropy-12907"
        assert result["app_name"] == "astropy/astropy"
        assert result["success"] is False

    def test_openhands_xml_tool_extraction(self, openhands_record: dict) -> None:
        result = convert_trajectory(openhands_record)
        assert result is not None
        actions = [s["action"] for s in result["spans"]]
        assert "bash" in actions
        assert "view_file" in actions

    def test_empty_trajectory_returns_none(self, empty_trajectory_record: dict) -> None:
        result = convert_trajectory(empty_trajectory_record)
        assert result is None

    def test_no_user_message_returns_none(self, no_user_message_record: dict) -> None:
        """Trajectories without user intent should be skipped."""
        result = convert_trajectory(no_user_message_record)
        assert result is None

    def test_resolved_minus_one_is_failure(self) -> None:
        record = {
            "instance_id": "test-001",
            "repo": "test/repo",
            "trajectory": [
                {"role": "user", "content": "Fix the bug."},
                {"role": "assistant", "content": "On it.",
                 "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]},
                {"role": "tool", "content": "done"},
            ],
            "resolved": -1,
            "tools": [],
            "metadata": {},
        }
        result = convert_trajectory(record)
        assert result is not None
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Tests — roundtrip to ShortChain schema
# ---------------------------------------------------------------------------

class TestSchemaRoundtrip:
    """Verify converted records parse into ShortChain Trajectory objects."""

    def test_converted_parses_to_trajectory(self, swe_agent_record: dict) -> None:
        converted = convert_trajectory(swe_agent_record)
        assert converted is not None
        traj = Trajectory(**converted)
        assert traj.task_id == "django__django-15814"
        assert traj.app_name == "django/django"
        assert traj.success is True
        assert traj.n_spans == 3

    def test_tools_used_derived(self, swe_agent_record: dict) -> None:
        converted = convert_trajectory(swe_agent_record)
        assert converted is not None
        traj = Trajectory(**converted)
        assert "bash" in traj.tools_used
        assert "str_replace_editor" in traj.tools_used

    def test_span_tool_names(self, swe_agent_record: dict) -> None:
        converted = convert_trajectory(swe_agent_record)
        assert converted is not None
        traj = Trajectory(**converted)
        tool_seq = traj.tool_sequence
        assert "bash" in tool_seq
        assert "str_replace_editor" in tool_seq

    def test_openhands_parses_to_trajectory(self, openhands_record: dict) -> None:
        converted = convert_trajectory(openhands_record)
        assert converted is not None
        traj = Trajectory(**converted)
        assert traj.task_id == "astropy__astropy-12907"
        assert traj.success is False
        assert traj.n_spans >= 1

    def test_intent_truncation(self) -> None:
        """Long intents should be truncated to 2000 chars."""
        record = {
            "instance_id": "long-intent-001",
            "repo": "test/repo",
            "trajectory": [
                {"role": "user", "content": "x" * 5000},
                {"role": "assistant", "content": "OK",
                 "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]},
                {"role": "tool", "content": "done"},
            ],
            "resolved": 1,
            "tools": [],
            "metadata": {},
        }
        result = convert_trajectory(record)
        assert result is not None
        assert len(result["intent"]) == 2000

    def test_multi_part_content(self) -> None:
        """Handle list-type content in user messages."""
        record = {
            "instance_id": "multi-part-001",
            "repo": "test/repo",
            "trajectory": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Fix this bug:"},
                        {"type": "text", "text": "The test fails."},
                    ],
                },
                {"role": "assistant", "content": "Looking into it.",
                 "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]},
                {"role": "tool", "content": "output"},
            ],
            "resolved": 1,
            "tools": [],
            "metadata": {},
        }
        result = convert_trajectory(record)
        assert result is not None
        assert "Fix this bug:" in result["intent"]
        assert "test fails" in result["intent"]
