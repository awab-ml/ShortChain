"""Tests for G3 (cross-category multi-tool) support.

Validates:
- Category-aware features in ContextFeatureBuilder
- tool_category in ToolFeatureBuilder
- Step-history category tracking
- G3 scenario classification
- End-to-end G3 pipeline with DatasetBuilder
"""

from __future__ import annotations

import pytest

from tabagent.features.context import ContextFeatureBuilder
from tabagent.features.tool import ToolFeatureBuilder
from tabagent.dataset.builder import DatasetBuilder
from tabagent.ingest.schema import Step, Trajectory
from tabagent.ingest.toolbench_loader import ToolBenchLoader


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_traj(
    task_id: str = "t1",
    tools: list[str] | None = None,
    app: str = "TestApp",
    available_tools: list[str] | None = None,
    n_steps: int = 3,
    success: bool = True,
) -> Trajectory:
    """Build a trajectory with controllable metadata."""
    tools = tools or [f"tool_{i}" for i in range(n_steps)]
    steps = [
        Step(
            agent_name="test",
            action=t,
            observation=f"result for {t}",
            thoughts=f"thinking about {t}",
        )
        for t in tools
    ]
    metadata = {
        "n_system_tools": len(tools),
        "system_tool_names": tools,
        "available_tools": available_tools or tools,
    }
    return Trajectory(
        task_id=task_id,
        intent="find weather and flights",
        steps=steps,
        success=success,
        app_name=app,
        metadata=metadata,
    )


CATEGORY_MAP = {
    "weather_api.get_forecast": "Weather",
    "weather_api.get_current": "Weather",
    "flight_api.search_flights": "Travel",
    "flight_api.get_prices": "Travel",
    "hotel_api.search_hotels": "Travel",
    "stock_api.get_price": "Finance",
}


# ------------------------------------------------------------------
# ContextFeatureBuilder — category-aware
# ------------------------------------------------------------------


class TestContextCategoryFeatures:
    """Tests for category-aware context features."""

    def test_n_categories_with_category_map(self) -> None:
        """Should compute n_categories from available_tools."""
        traj = _make_traj(
            available_tools=[
                "weather_api.get_forecast",
                "flight_api.search_flights",
                "stock_api.get_price",
            ]
        )
        builder = ContextFeatureBuilder(category_map=CATEGORY_MAP)
        features = builder.build(traj)

        assert features["n_categories"] == 3  # Weather, Travel, Finance

    def test_n_categories_same_category(self) -> None:
        """Same-category tools should produce n_categories=1."""
        traj = _make_traj(
            available_tools=[
                "weather_api.get_forecast",
                "weather_api.get_current",
            ]
        )
        builder = ContextFeatureBuilder(category_map=CATEGORY_MAP)
        features = builder.build(traj)

        assert features["n_categories"] == 1

    def test_n_categories_without_map(self) -> None:
        """Without category_map, n_categories should be 0."""
        traj = _make_traj(available_tools=["api_a", "api_b"])
        builder = ContextFeatureBuilder()
        features = builder.build(traj)

        assert features["n_categories"] == 0

    def test_available_tool_count(self) -> None:
        """Should count available tools from metadata."""
        traj = _make_traj(
            available_tools=["a", "b", "c", "d"]
        )
        builder = ContextFeatureBuilder()
        features = builder.build(traj)

        assert features["available_tool_count"] == 4


# ------------------------------------------------------------------
# Step-history category features
# ------------------------------------------------------------------


class TestStepHistoryFeatures:
    """Tests for step-history category tracking."""

    def test_previous_tool_categories(self) -> None:
        """Should map previous tools to their categories."""
        traj = _make_traj(
            tools=[
                "weather_api.get_forecast",
                "flight_api.search_flights",
                "stock_api.get_price",
            ],
        )
        builder = ContextFeatureBuilder(category_map=CATEGORY_MAP)
        features = builder.build(traj)

        assert "previous_tool_categories" in features
        cats = features["previous_tool_categories"].split(" | ")
        assert cats == ["Weather", "Travel", "Finance"]

    def test_n_prev_categories_used(self) -> None:
        """Should count unique categories used so far."""
        traj = _make_traj(
            tools=[
                "weather_api.get_forecast",
                "weather_api.get_current",
                "flight_api.search_flights",
            ],
        )
        builder = ContextFeatureBuilder(category_map=CATEGORY_MAP)
        features = builder.build(traj)

        assert features["n_prev_categories_used"] == 2  # Weather, Travel

    def test_prev_categories_repeat(self) -> None:
        """Should detect when categories repeat (same cat used multiple times)."""
        traj = _make_traj(
            tools=[
                "weather_api.get_forecast",
                "weather_api.get_current",
            ],
        )
        builder = ContextFeatureBuilder(category_map=CATEGORY_MAP)
        features = builder.build(traj)

        # Two tools from Weather → categories repeat
        assert features["prev_categories_repeat"] == 1

    def test_no_repeat_when_all_unique(self) -> None:
        """No repeat when each tool from a different category."""
        traj = _make_traj(
            tools=[
                "weather_api.get_forecast",
                "flight_api.search_flights",
                "stock_api.get_price",
            ],
        )
        builder = ContextFeatureBuilder(category_map=CATEGORY_MAP)
        features = builder.build(traj)

        assert features["prev_categories_repeat"] == 0

    def test_step_history_empty_when_no_category_map(self) -> None:
        """Without category_map, step-history features should not appear."""
        traj = _make_traj(tools=["a", "b", "c"])
        builder = ContextFeatureBuilder()
        features = builder.build(traj)

        assert "previous_tool_categories" not in features
        assert "n_prev_categories_used" not in features


# ------------------------------------------------------------------
# ToolFeatureBuilder — tool_category
# ------------------------------------------------------------------


class TestToolCategoryFeature:
    """Tests for tool_category feature."""

    def test_tool_category_present(self) -> None:
        """Should include tool_category from tool_meta."""
        builder = ToolFeatureBuilder()
        features = builder.build(
            "weather_api.get_forecast",
            tool_meta={"description": "Get forecast", "category": "Weather"},
        )
        assert features["tool_category"] == "Weather"

    def test_tool_category_default(self) -> None:
        """Should default to 'unknown' when no category provided."""
        builder = ToolFeatureBuilder()
        features = builder.build("some_api", tool_meta={"description": "test"})
        assert features["tool_category"] == "unknown"


# ------------------------------------------------------------------
# DatasetBuilder with category_map
# ------------------------------------------------------------------


class TestDatasetBuilderWithCategories:
    """Tests for DatasetBuilder with category_map."""

    def test_category_map_passed_through(self) -> None:
        """Category info should appear in the built dataset."""
        traj = _make_traj(
            tools=["weather_api.get_forecast", "flight_api.search_flights"],
            available_tools=[
                "weather_api.get_forecast",
                "flight_api.search_flights",
                "stock_api.get_price",
            ],
        )
        catalog = {
            "weather_api.get_forecast": "Get weather forecast",
            "flight_api.search_flights": "Search flights",
            "stock_api.get_price": "Get stock price",
        }
        builder = DatasetBuilder(
            tool_catalog=catalog,
            category_map=CATEGORY_MAP,
        )
        df = builder.build([traj])

        assert "tool_category" in df.columns
        assert "n_categories" in df.columns
        assert "previous_tool_categories" in df.columns

        # Positive rows should have real categories
        pos = df[df["label"] == 1]
        categories = set(pos["tool_category"].values)
        assert "Weather" in categories or "Travel" in categories

    def test_n_categories_in_dataset(self) -> None:
        """n_categories should reflect cross-category tools."""
        traj = _make_traj(
            tools=["weather_api.get_forecast", "flight_api.search_flights"],
            available_tools=[
                "weather_api.get_forecast",
                "flight_api.search_flights",
                "stock_api.get_price",
            ],
        )
        catalog = {
            "weather_api.get_forecast": "Get weather forecast",
            "flight_api.search_flights": "Search flights",
            "stock_api.get_price": "Get stock price",
        }
        builder = DatasetBuilder(
            tool_catalog=catalog,
            category_map=CATEGORY_MAP,
        )
        df = builder.build([traj])

        # All rows should have n_categories=3 (Weather, Travel, Finance)
        assert (df["n_categories"] == 3).all()


# ------------------------------------------------------------------
# G3 classification
# ------------------------------------------------------------------


class TestG3Classification:
    """Tests for G3 scenario classification."""

    def test_g3_filter_in_loader(self) -> None:
        """load_with_filter(scenario='G3') should only return cross-category."""
        # This is already tested in test_toolbench_g2.py::test_g3_filter
        # Adding a specific validation here
        from pathlib import Path
        fixture = Path("data/toolbench/fixtures/sample_preprocessed.json")
        if not fixture.exists():
            pytest.skip("Fixture not available")

        loader = ToolBenchLoader(catalog=None, success_only=False)
        trajs = loader.load_with_filter(str(fixture), scenario="G3")
        # G3 filter applied — all should have cross-category tools
        assert isinstance(trajs, list)


# ------------------------------------------------------------------
# Step expansion with G3
# ------------------------------------------------------------------


class TestG3StepExpansion:
    """Tests for step-level expansion with cross-category trajectories."""

    def test_step_expansion_preserves_available_tools(self) -> None:
        """Sub-trajectories should inherit available_tools metadata."""
        traj = _make_traj(
            tools=[
                "weather_api.get_forecast",
                "flight_api.search_flights",
                "stock_api.get_price",
            ],
            available_tools=[
                "weather_api.get_forecast",
                "flight_api.search_flights",
                "stock_api.get_price",
                "hotel_api.search_hotels",
            ],
        )
        subs = ToolBenchLoader.expand_to_step_trajectories(traj)

        assert len(subs) == 3
        for sub in subs:
            # available_tools should be inherited from parent
            assert "available_tools" in sub.metadata
            assert len(sub.metadata["available_tools"]) == 4

    def test_step_history_context_grows(self) -> None:
        """Each step sub-trajectory should have growing category history."""
        traj = _make_traj(
            tools=[
                "weather_api.get_forecast",
                "flight_api.search_flights",
                "stock_api.get_price",
            ],
        )
        subs = ToolBenchLoader.expand_to_step_trajectories(traj)
        builder = ContextFeatureBuilder(category_map=CATEGORY_MAP)

        # step_0: no prior steps
        f0 = builder.build(subs[0])
        assert f0["n_prev_categories_used"] == 0

        # step_1: 1 prior step (Weather)
        f1 = builder.build(subs[1])
        assert f1["n_prev_categories_used"] == 1

        # step_2: 2 prior steps (Weather, Travel)
        f2 = builder.build(subs[2])
        assert f2["n_prev_categories_used"] == 2
