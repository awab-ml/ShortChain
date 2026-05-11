"""ToolBench tool environment parser and catalog builder.

Walks the ``data/toolenv/tools/{Category}/`` directory tree produced by
the ToolBench project and builds a flat API-level tool catalog that
TabAgent can consume for feature engineering and negative sampling.

Each API endpoint becomes a separate entry in the catalog, keyed as
``"{standardized_tool_name}.{api_name}"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tabagent.utils.logging import get_logger

log = get_logger(__name__)


class ToolBenchCatalog:
    """Parse ToolBench tool environment into a TabAgent-compatible catalog.

    Parameters
    ----------
    catalog
        ``api_key → description`` mapping where *api_key* is
        ``"{tool_name}.{api_name}"`` (API-level granularity).
    category_map
        ``api_key → category_name`` mapping.
    tool_to_apis
        ``tool_name → [api_key, ...]`` mapping.
    """

    def __init__(
        self,
        catalog: dict[str, str],
        category_map: dict[str, str],
        tool_to_apis: dict[str, list[str]],
    ) -> None:
        self._catalog = catalog
        self._category_map = category_map
        self._tool_to_apis = tool_to_apis

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_toolenv(cls, tools_dir: str | Path) -> "ToolBenchCatalog":
        """Build catalog by walking ``data/toolenv/tools/{Category}/``.

        For API-level granularity, each entry in ``api_list`` becomes a
        separate tool in the catalog::

            key:   "{standardized_tool_name}.{api_name}"
            value: "{api_description} (Tool: {tool_description})"

        Parameters
        ----------
        tools_dir
            Path to the ``data/toolenv/tools/`` directory.

        Returns
        -------
        ToolBenchCatalog
        """
        tools_dir = Path(tools_dir)
        if not tools_dir.is_dir():
            raise FileNotFoundError(f"Tool environment directory not found: {tools_dir}")

        catalog: dict[str, str] = {}
        category_map: dict[str, str] = {}
        tool_to_apis: dict[str, list[str]] = {}

        categories = sorted(
            d.name for d in tools_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        for category in categories:
            cat_dir = tools_dir / category
            json_files = sorted(cat_dir.glob("*.json"))

            for json_file in json_files:
                try:
                    tool_data = _read_tool_json(json_file)
                except Exception as exc:
                    log.warning(f"Skipping {json_file}: {exc}")
                    continue

                tool_name = _normalise_tool_name(tool_data)
                tool_desc = tool_data.get("tool_description", "")
                api_list = tool_data.get("api_list", [])

                api_keys: list[str] = []
                for api in api_list:
                    api_name = api.get("name", "")
                    if not api_name:
                        continue

                    api_key = f"{tool_name}.{_normalise_api_name(api_name)}"
                    api_desc = api.get("description", "")

                    # Combine API + tool description for richer features
                    full_desc = api_desc
                    if tool_desc:
                        full_desc = f"{api_desc} (Tool: {tool_desc})"

                    catalog[api_key] = full_desc
                    category_map[api_key] = category
                    api_keys.append(api_key)

                if api_keys:
                    tool_to_apis[tool_name] = api_keys

        log.info(
            f"Parsed ToolBench catalog: [bold]{len(catalog)}[/bold] APIs "
            f"from [bold]{len(tool_to_apis)}[/bold] tools "
            f"across [bold]{len(categories)}[/bold] categories"
        )
        return cls(catalog, category_map, tool_to_apis)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def catalog(self) -> dict[str, str]:
        """``api_key → description`` mapping."""
        return self._catalog

    @property
    def category_map(self) -> dict[str, str]:
        """``api_key → category_name`` mapping."""
        return self._category_map

    @property
    def tool_to_apis(self) -> dict[str, list[str]]:
        """``tool_name → [api_key, ...]`` mapping."""
        return self._tool_to_apis

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_category(self, api_key: str) -> str:
        """Return the category for an API key, or ``""`` if unknown."""
        return self._category_map.get(api_key, "")

    def get_description(self, api_key: str) -> str:
        """Return the description for an API key, or ``""`` if unknown."""
        return self._catalog.get(api_key, "")

    def resolve_action(self, action: str) -> str | None:
        """Attempt to resolve a ToolBench action string to a catalog key.

        ToolBench actions use the format ``{api_name}_for_{tool_name}``
        (e.g. ``"get_weather_for_open_weather"``).  This method maps them
        back to catalog keys (``"open_weather.get_weather"``).

        Parameters
        ----------
        action
            Raw action string from a ToolBench conversation.

        Returns
        -------
        str or None
            The matching catalog key, or ``None`` if not found.
        """
        if not action or action == "Finish":
            return None

        # Try direct match first (already in "tool.api" format)
        if action in self._catalog:
            return action

        # Parse "{api_name}_for_{tool_name}" format
        if "_for_" in action:
            parts = action.rsplit("_for_", 1)
            if len(parts) == 2:
                api_name, tool_name = parts
                key = f"{tool_name}.{api_name}"
                if key in self._catalog:
                    return key
                # Try normalised forms
                key_norm = f"{_normalise_api_name(tool_name)}.{_normalise_api_name(api_name)}"
                if key_norm in self._catalog:
                    return key_norm

        # Brute-force search (slow, last resort)
        action_lower = action.lower().replace(" ", "_")
        for key in self._catalog:
            if key.lower().endswith(f".{action_lower}"):
                return key

        return None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return summary statistics about the catalog."""
        categories = set(self._category_map.values())
        cat_counts = {}
        for cat in categories:
            cat_counts[cat] = sum(1 for c in self._category_map.values() if c == cat)

        top_categories = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "n_apis": len(self._catalog),
            "n_tools": len(self._tool_to_apis),
            "n_categories": len(categories),
            "top_categories": top_categories,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_tool_json(path: Path) -> dict[str, Any]:
    """Read and return a single tool JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalise_tool_name(tool_data: dict[str, Any]) -> str:
    """Extract and normalise the tool name from tool JSON data.

    Prefers ``standardized_name``, falls back to ``tool_name``.
    """
    name = tool_data.get("standardized_name", "")
    if not name:
        name = tool_data.get("tool_name", "")
    # Normalise: lowercase, replace spaces with underscores
    return name.lower().replace(" ", "_").strip()


def _normalise_api_name(name: str) -> str:
    """Normalise an API name: lowercase, strip whitespace."""
    return name.lower().replace(" ", "_").strip()
