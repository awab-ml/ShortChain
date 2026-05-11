"""Tests for ToolBenchLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from tabagent.ingest.toolbench_catalog import ToolBenchCatalog
from tabagent.ingest.toolbench_loader import (
    ToolBenchLoader,
    parse_assistant_message,
    _determine_success,
    _extract_tool_names_from_system,
    _clean_user_message,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "toolbench" / "fixtures"
SAMPLE_PREPROCESSED = FIXTURES / "sample_preprocessed.json"
SAMPLE_TOOLENV = FIXTURES / "sample_toolenv"


# ------------------------------------------------------------------
# Unit tests for parsing helpers
# ------------------------------------------------------------------


class TestParseAssistantMessage:
    """Tests for parse_assistant_message()."""

    def test_standard_format(self) -> None:
        msg = (
            "\nThought: I need weather data.\n"
            "Action: get_weather_for_open_weather\n"
            'Action Input: {\n  "location": "London"\n}\n'
        )
        thought, action, action_input = parse_assistant_message(msg)
        assert thought == "I need weather data."
        assert action == "get_weather_for_open_weather"
        assert '"location"' in action_input

    def test_finish_action(self) -> None:
        msg = (
            "\nThought: I have the answer.\n"
            "Action: Finish\n"
            'Action Input: {\n  "return_type": "give_answer",\n'
            '  "final_answer": "The weather is 15C"\n}\n'
        )
        thought, action, action_input = parse_assistant_message(msg)
        assert action == "Finish"
        assert "give_answer" in action_input

    def test_empty_message(self) -> None:
        thought, action, action_input = parse_assistant_message("")
        assert thought == ""
        assert action == ""
        assert action_input == ""


class TestDetermineSuccess:
    """Tests for _determine_success()."""

    def test_give_answer_is_success(self) -> None:
        convs = [
            {"from": "user", "value": "Do something"},
            {"from": "assistant", "value": 'Action: Finish\nAction Input: {"return_type": "give_answer", "final_answer": "Done"}'},
        ]
        assert _determine_success(convs) is True

    def test_give_up_is_failure(self) -> None:
        convs = [
            {"from": "user", "value": "Do something"},
            {"from": "assistant", "value": 'Action: Finish\nAction Input: {"return_type": "give_up_and_restart"}'},
        ]
        assert _determine_success(convs) is False

    def test_no_finish_defaults_true(self) -> None:
        convs = [
            {"from": "user", "value": "Do something"},
            {"from": "assistant", "value": "Thought: Just thinking.\nAction: some_api\nAction Input: {}"},
        ]
        assert _determine_success(convs) is True


class TestExtractToolNames:
    """Tests for _extract_tool_names_from_system()."""

    def test_single_tool(self) -> None:
        system = (
            "Some preamble.\n"
            "You have access of the following tools:\n"
            "1.open_weather: Provides weather data.\n\n"
            "Specifically, you have access to the following APIs:"
        )
        tools = _extract_tool_names_from_system(system)
        assert tools == ["open_weather"]

    def test_multiple_tools(self) -> None:
        system = (
            "Some preamble.\n"
            "You have access of the following tools:\n"
            "1.open_weather: Weather data.\n"
            "2.stock_tracker: Stock data.\n\n"
            "Specifically..."
        )
        tools = _extract_tool_names_from_system(system)
        assert tools == ["open_weather", "stock_tracker"]

    def test_no_tools(self) -> None:
        system = "No tools section here."
        tools = _extract_tool_names_from_system(system)
        assert tools == []


class TestCleanUserMessage:
    """Tests for _clean_user_message()."""

    def test_strips_begin(self) -> None:
        msg = "\nCheck the weather in London.\nBegin!\n"
        assert _clean_user_message(msg) == "Check the weather in London."

    def test_no_begin(self) -> None:
        msg = "Just a plain message."
        assert _clean_user_message(msg) == "Just a plain message."


# ------------------------------------------------------------------
# Integration tests with fixture data
# ------------------------------------------------------------------


class TestToolBenchLoader:
    """Tests for ToolBenchLoader with fixture data."""

    @pytest.fixture()
    def catalog(self) -> ToolBenchCatalog:
        return ToolBenchCatalog.from_toolenv(SAMPLE_TOOLENV)

    @pytest.fixture()
    def loader(self, catalog: ToolBenchCatalog) -> ToolBenchLoader:
        return ToolBenchLoader(catalog=catalog, success_only=False)

    @pytest.fixture()
    def success_loader(self, catalog: ToolBenchCatalog) -> ToolBenchLoader:
        return ToolBenchLoader(catalog=catalog, success_only=True)

    def test_load_all(self, loader: ToolBenchLoader) -> None:
        """Should load all 5 fixture instances."""
        trajs = loader.load(SAMPLE_PREPROCESSED)
        assert len(trajs) == 5

    def test_load_success_only(self, success_loader: ToolBenchLoader) -> None:
        """Should filter out the failed instance (fixture_005)."""
        trajs = success_loader.load(SAMPLE_PREPROCESSED)
        assert len(trajs) == 4
        assert all(t.success for t in trajs)

    def test_trajectory_has_intent(self, loader: ToolBenchLoader) -> None:
        """Each trajectory should have a non-empty intent."""
        trajs = loader.load(SAMPLE_PREPROCESSED)
        for t in trajs:
            assert t.intent, f"Empty intent for {t.task_id}"
            assert "Begin!" not in t.intent  # Should be stripped

    def test_trajectory_has_steps(self, loader: ToolBenchLoader) -> None:
        """Successful trajectories should have at least 1 step."""
        trajs = loader.load(SAMPLE_PREPROCESSED)
        for t in trajs:
            if t.success:
                assert len(t.steps) > 0, f"No steps for {t.task_id}"

    def test_trajectory_tools_used(self, loader: ToolBenchLoader) -> None:
        """tools_used should contain resolved API keys."""
        trajs = loader.load(SAMPLE_PREPROCESSED)
        # First fixture uses open_weather APIs
        weather_traj = trajs[0]
        assert len(weather_traj.tools_used) > 0

    def test_trajectory_app_name(self, loader: ToolBenchLoader) -> None:
        """app_name should be resolved from catalog categories."""
        trajs = loader.load(SAMPLE_PREPROCESSED)
        # First fixture is Weather category
        assert trajs[0].app_name in ("Weather", "open_weather")

    def test_trajectory_success_detection(self, loader: ToolBenchLoader) -> None:
        """Last fixture should be marked as failed."""
        trajs = loader.load(SAMPLE_PREPROCESSED)
        # fixture_005 ends with give_up_and_restart
        failed = [t for t in trajs if not t.success]
        assert len(failed) == 1

    def test_filter_g1(self, loader: ToolBenchLoader) -> None:
        """load_with_filter(scenario='G1') should return only single-tool trajs."""
        trajs = loader.load_with_filter(SAMPLE_PREPROCESSED, scenario="G1")
        # Fixtures 1,2,3,5 are single-tool; 4 is multi-tool
        for t in trajs:
            assert t.metadata.get("n_system_tools") == 1

    def test_metadata_contains_system_tools(self, loader: ToolBenchLoader) -> None:
        """Metadata should contain system tool information."""
        trajs = loader.load(SAMPLE_PREPROCESSED)
        for t in trajs:
            assert "n_system_tools" in t.metadata
            assert "system_tool_names" in t.metadata

    def test_steps_have_thoughts(self, loader: ToolBenchLoader) -> None:
        """Steps should contain thought text from assistant messages."""
        trajs = loader.load(SAMPLE_PREPROCESSED)
        weather = trajs[0]
        has_thoughts = any(s.thoughts for s in weather.steps)
        assert has_thoughts

    def test_integration_with_dataset_builder(self, loader: ToolBenchLoader) -> None:
        """Trajectories should work with DatasetBuilder.build()."""
        from tabagent.dataset.builder import DatasetBuilder

        trajs = loader.load_with_filter(SAMPLE_PREPROCESSED, scenario="G1")
        # Filter to successful only
        trajs = [t for t in trajs if t.success]
        assert len(trajs) > 0

        builder = DatasetBuilder()
        df = builder.build(trajs)

        assert len(df) > 0
        assert "label" in df.columns
        assert "tool_name" in df.columns
        assert df["label"].sum() > 0  # Has positive examples
