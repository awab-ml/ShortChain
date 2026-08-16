"""Golden end-to-end projection tests over tests/fixtures/otel/*.json (2.8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from shortchain.config import ProjectionConfig
from shortchain.ingest.otel import (
    OtelTrace,
    OtelTraceProjector,
    OtelTrajectoryLoader,
)

FIXTURES = Path(__file__).parent / "fixtures" / "otel"


def load_trace(name: str) -> OtelTrace:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"fixture missing: {name}")
    return OtelTrace.model_validate_json(path.read_text())


def project_fixture(name: str, cfg: ProjectionConfig | None = None):
    return OtelTraceProjector(cfg).project(load_trace(name))


# ---------------------------------------------------------------------------
# Framework-specific fixtures
# ---------------------------------------------------------------------------


class TestLangChain:
    def test_basic_trace(self):
        result = project_fixture("langchain_tool.json")
        assert result.drop_reason is None
        traj = result.trajectory
        assert traj.task_id == "ticket-1842"
        assert traj.intent == "Refund order 9921 and email the customer"
        assert traj.success is True
        assert traj.app_name == "support-agent"
        assert [s.action for s in traj.spans] == ["lookup_order"]
        s = traj.spans[0]
        assert s.observation == '{"status": "delivered"}'
        assert s.metadata["tool_arguments"] == '{"order_id": 9921}'
        assert traj.metadata["success_source"] == "association"
        assert traj.metadata["projection.fallback"] == "none"
        assert traj.metadata["tokens.input_sum"] == 150


class TestOpenAIAgents:
    def test_trace(self):
        result = project_fixture("openai_agents_tool.json")
        assert result.drop_reason is None
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["lookup_order"]
        assert traj.spans[0].agent_name == "SupportAgent"


class TestAgno:
    def test_trace(self):
        result = project_fixture("agno_tool.json")
        assert result.drop_reason is None
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["search_catalog"]
        s = traj.spans[0]
        assert s.observation == '{"hits": 2}'
        assert "laptop" in s.metadata["tool_arguments"]


# ---------------------------------------------------------------------------
# MCP dedup fixtures
# ---------------------------------------------------------------------------


class TestMCP:
    def test_single_server_tool(self):
        result = project_fixture("mcp_tool.json")
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["add_numbers"]
        assert traj.spans[0].observation == '{"result": 3}'

    def test_client_server_twins_collapse(self):
        result = project_fixture("mcp_tool_client_server.json")
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["add_numbers"]
        assert len(traj.spans) == 1
        assert result.stats["n_tool"] == 2  # 2 raw tool spans in, 1 span out

    def test_two_calls_are_two_decisions(self):
        result = project_fixture("mcp_tool_two_calls.json")
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["add_numbers", "add_numbers"]
        assert [s.observation for s in traj.spans] == ['{"result": 3}', '{"result": 4}']

    def test_langchain_wrapper_collapses(self):
        result = project_fixture("langchain_mcp_wrapper.json")
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["add_numbers"]
        assert len(traj.spans) == 1


# ---------------------------------------------------------------------------
# OpenInference fixtures
# ---------------------------------------------------------------------------


class TestOpenInference:
    def test_function_tool(self):
        result = project_fixture("openinference_function.json")
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["check_status"]
        s = traj.spans[0]
        assert s.observation == '{"status": "shipped"}'
        assert "9921" in s.metadata["tool_arguments"]

    def test_mcp_tools_nameless_never_decisions(self):
        result = project_fixture("openinference_mcp_tools.json")
        assert result.drop_reason is None
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["refund_order"]
        assert all(s.action != "mcp_tools" for s in traj.spans)


# ---------------------------------------------------------------------------
# CrewAI fallback fixture
# ---------------------------------------------------------------------------


class TestCrewAIFallback:
    def test_two_turns_one_decision(self):
        result = project_fixture("crewai_llm_fallback.json")
        assert result.drop_reason is None
        traj = result.trajectory
        assert [s.action for s in traj.spans] == ["search_emails"]
        assert traj.spans[0].observation == "3 unpaid invoices"
        assert traj.metadata["projection.fallback"] == "llm_tool_calls"


# ---------------------------------------------------------------------------
# Association success fixtures
# ---------------------------------------------------------------------------


class TestAssociation:
    def test_success_true(self):
        result = project_fixture("association_success.json")
        assert result.drop_reason is None
        assert result.trajectory.success is True
        assert result.trajectory.metadata["success_source"] == "association"

    def test_success_false_is_known_failure(self):
        """Golden contract: 'false' must NOT be treated as unknown."""
        result = project_fixture("association_success_false.json")
        assert result.drop_reason is None
        assert result.trajectory.success is False
        assert result.trajectory.metadata["success_source"] == "association"


# ---------------------------------------------------------------------------
# End-to-end via OtelTrajectoryLoader over the whole fixture directory
# ---------------------------------------------------------------------------


class TestLoaderGolden:
    def test_all_fixtures_load_without_error(self):
        loader = OtelTrajectoryLoader()
        trajs = loader.load(FIXTURES)
        # Every fixture projects (no unhandled crashes); some are assertively
        # checked above.
        assert len(trajs) >= 10

    def test_directory_of_traces_uses_projection(self):
        t = project_fixture("association_success.json").trajectory
        assert t is not None