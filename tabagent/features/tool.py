"""Tool / candidate feature builder for TabAgent.

Builds features for a candidate tool, optionally enriched with
corpus-level statistics (frequency, co-occurrence) and context
(app-matching).
"""

from __future__ import annotations

from typing import Any

from tabagent.features.stats import CorpusStats
from tabagent.utils.logging import get_logger

log = get_logger(__name__)


class ToolFeatureBuilder:
    """Build features for a candidate tool.

    Parameters
    ----------
    corpus_stats
        Optional precomputed corpus statistics for frequency / co-usage
        features.
    """

    def __init__(self, corpus_stats: CorpusStats | None = None) -> None:
        self.corpus_stats = corpus_stats

    def build(
        self,
        tool_name: str,
        tool_meta: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build features for a single candidate tool.

        Parameters
        ----------
        tool_name
            The name of the candidate tool.
        tool_meta
            Optional metadata dict (e.g. ``{"description": "..."}``).
        context
            Optional context dict (from ``ContextFeatureBuilder.build``)
            used to compute cross-features like ``tool_app_match``.

        Returns
        -------
        dict[str, Any]
            Feature dictionary for this tool.
        """
        meta = tool_meta or {}
        features: dict[str, Any] = {}

        # --- Core tool features ---
        features["tool_name"] = tool_name
        features["tool_description"] = meta.get("description", "")
        features["tool_name_length"] = len(tool_name)
        features["has_description"] = bool(meta.get("description"))

        # --- Context cross-features ---
        if context:
            features["tool_app_match"] = self._app_match(tool_name, context)

        # --- Corpus-derived features ---
        if self.corpus_stats:
            features["tool_frequency"] = self.corpus_stats.get_tool_freq(tool_name)
            # Co-occurrence with tools already used in context
            if context and context.get("previous_tools"):
                features["tool_co_occurrence"] = self._co_occurrence_score(
                    tool_name, context["previous_tools"]
                )
            else:
                features["tool_co_occurrence"] = 0.0

        return features

    def _app_match(self, tool_name: str, context: dict[str, Any]) -> int:
        """Check if tool belongs to the same app as the context."""
        app_name = context.get("app_name", "")
        if not app_name or not self.corpus_stats:
            return 0
        app_tools = self.corpus_stats.get_same_app_tools(app_name)
        return 1 if tool_name in app_tools else 0

    def _co_occurrence_score(self, tool_name: str, previous_tools: str) -> float:
        """Compute a co-occurrence score between tool and previously used tools."""
        if not self.corpus_stats:
            return 0.0
        prev = [t.strip() for t in previous_tools.split("|") if t.strip()]
        if not prev:
            return 0.0
        co_occ = self.corpus_stats.get_co_occurring_tools(tool_name)
        total = sum(co_occ.get(p, 0) for p in prev)
        return total / len(prev)
