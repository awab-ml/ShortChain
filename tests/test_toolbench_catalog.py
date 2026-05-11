"""Tests for ToolBenchCatalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from tabagent.ingest.toolbench_catalog import (
    ToolBenchCatalog,
    _normalise_api_name,
    _normalise_tool_name,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "toolbench" / "fixtures"
SAMPLE_TOOLENV = FIXTURES / "sample_toolenv"


class TestToolBenchCatalog:
    """Tests for ToolBenchCatalog.from_toolenv()."""

    @pytest.fixture()
    def catalog(self) -> ToolBenchCatalog:
        return ToolBenchCatalog.from_toolenv(SAMPLE_TOOLENV)

    def test_from_toolenv_loads(self, catalog: ToolBenchCatalog) -> None:
        """Catalog should load all API entries from fixture toolenv."""
        # 3 tools: open_weather(2 APIs), stock_tracker(3 APIs), live_scores(2 APIs)
        assert len(catalog.catalog) == 7

    def test_api_level_granularity(self, catalog: ToolBenchCatalog) -> None:
        """Catalog keys should be in 'tool.api' format."""
        for key in catalog.catalog:
            assert "." in key, f"Key '{key}' missing dot separator"

    def test_known_api_keys(self, catalog: ToolBenchCatalog) -> None:
        """Expected API keys should be present."""
        assert "open_weather.get_current_weather" in catalog.catalog
        assert "open_weather.get_forecast" in catalog.catalog
        assert "stock_tracker.get_stock_price" in catalog.catalog
        assert "live_scores.get_live_matches" in catalog.catalog

    def test_descriptions_not_empty(self, catalog: ToolBenchCatalog) -> None:
        """Every API entry should have a non-empty description."""
        for key, desc in catalog.catalog.items():
            assert desc, f"Empty description for {key}"

    def test_descriptions_contain_tool_info(self, catalog: ToolBenchCatalog) -> None:
        """Descriptions should include '(Tool: ...)' suffix."""
        desc = catalog.catalog["open_weather.get_current_weather"]
        assert "(Tool:" in desc

    def test_category_map(self, catalog: ToolBenchCatalog) -> None:
        """Every API key should map to a category."""
        assert len(catalog.category_map) == len(catalog.catalog)
        assert catalog.get_category("open_weather.get_current_weather") == "Weather"
        assert catalog.get_category("stock_tracker.get_stock_price") == "Finance"
        assert catalog.get_category("live_scores.get_live_matches") == "Sports"

    def test_tool_to_apis(self, catalog: ToolBenchCatalog) -> None:
        """tool_to_apis should map tool names to their API keys."""
        assert "open_weather" in catalog.tool_to_apis
        assert len(catalog.tool_to_apis["open_weather"]) == 2
        assert "stock_tracker" in catalog.tool_to_apis
        assert len(catalog.tool_to_apis["stock_tracker"]) == 3

    def test_summary(self, catalog: ToolBenchCatalog) -> None:
        """Summary should contain correct counts."""
        s = catalog.summary()
        assert s["n_apis"] == 7
        assert s["n_tools"] == 3
        assert s["n_categories"] == 3

    def test_resolve_action_for_format(self, catalog: ToolBenchCatalog) -> None:
        """resolve_action should handle '{api}_for_{tool}' format."""
        result = catalog.resolve_action("get_current_weather_for_open_weather")
        assert result == "open_weather.get_current_weather"

    def test_resolve_action_direct(self, catalog: ToolBenchCatalog) -> None:
        """resolve_action should handle direct 'tool.api' format."""
        result = catalog.resolve_action("open_weather.get_forecast")
        assert result == "open_weather.get_forecast"

    def test_resolve_action_finish(self, catalog: ToolBenchCatalog) -> None:
        """resolve_action should return None for 'Finish'."""
        assert catalog.resolve_action("Finish") is None

    def test_resolve_action_unknown(self, catalog: ToolBenchCatalog) -> None:
        """resolve_action should return None for unknown actions."""
        assert catalog.resolve_action("totally_fake_api") is None

    def test_missing_directory_raises(self) -> None:
        """from_toolenv should raise if directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ToolBenchCatalog.from_toolenv("/nonexistent/path")


class TestNormalisationHelpers:
    """Tests for internal normalisation functions."""

    def test_normalise_tool_name_standardized(self) -> None:
        result = _normalise_tool_name({"standardized_name": "My Tool"})
        assert result == "my_tool"

    def test_normalise_tool_name_fallback(self) -> None:
        result = _normalise_tool_name({"tool_name": "Weather API"})
        assert result == "weather_api"

    def test_normalise_api_name(self) -> None:
        assert _normalise_api_name("Get Current Weather") == "get_current_weather"
