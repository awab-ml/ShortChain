"""Context feature builder for ShortChain.

Extracts state-aware context features from a trajectory, optionally
at a specific span index.  Designed for trajectory-level use now
(``span_index=None``) with a clean migration path to span-level
feature extraction in future phases.
"""

from __future__ import annotations

from typing import Any

from shortchain.features.stats import CorpusStats
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class ContextFeatureBuilder:
    """Build context features from a trajectory.

    Parameters
    ----------
    corpus_stats
        Optional precomputed corpus statistics for dependency features.
    include_state
        Whether to include state-aware features (span_index, last_action, etc.).
    include_dependencies
        Whether to include dependency features (tool_diversity, etc.).
    """

    def __init__(
        self,
        corpus_stats: CorpusStats | None = None,
        include_state: bool = True,
        include_dependencies: bool = True,
    ) -> None:
        self.corpus_stats = corpus_stats
        self.include_state = include_state
        self.include_dependencies = include_dependencies

    def build(self, traj: Any, span_index: int | None = None) -> dict[str, Any]:
        """Extract context features from a trajectory.

        Parameters
        ----------
        traj
            A ``Trajectory`` object from ``shortchain.ingest.schema``.
        span_index
            If ``None`` (default), extracts trajectory-level features
            summarising the whole run.  If provided, extracts features
            as of that span (for future span-level decision modelling).

        Returns
        -------
        dict[str, Any]
            Feature dictionary ready for DataFrame construction.
        """
        features: dict[str, Any] = {}

        # --- Core features (always present) ---
        features["task_id"] = traj.task_id
        features["intent"] = traj.intent
        features["app_name"] = traj.app_name

        if span_index is None:
            # Trajectory-level (prior to execution): no previous tools have executed yet
            features["n_spans"] = 0
            features["previous_tools"] = ""
            features["last_thought"] = ""
        else:
            # Span-level: summarise up to span_index
            spans_so_far = traj.spans[: span_index + 1]
            tool_seq = [s.tool_name for s in spans_so_far if s.tool_name]
            features["n_spans"] = len(spans_so_far)
            features["previous_tools"] = " | ".join(tool_seq)
            last = spans_so_far[-1] if spans_so_far else None
            features["last_thought"] = (last.thoughts or "") if last else ""

        # --- State features ---
        if self.include_state:
            features.update(self._state_features(traj, span_index))

        # --- Dependency features ---
        if self.include_dependencies:
            features.update(self._dependency_features(traj, span_index))

        return features


    #include_state:
       # Whether to include state-aware features (span_index, last_action, etc.)
    def _state_features(self, traj: Any, span_index: int | None) -> dict[str, Any]:
        """Extract state-aware features."""
        features: dict[str, Any] = {}

        if span_index is None:
            # Trajectory-level state (prior to execution): no steps taken yet
            features["span_index"] = 0
            features["last_action"] = ""
            features["last_observation"] = ""
            features["unique_tools_so_far"] = 0
            features["history_summary"] = ""
        else:
            spans_so_far = traj.spans[: span_index + 1]
            features["span_index"] = span_index
            tool_seq = [s.tool_name for s in spans_so_far if s.tool_name]
            features["last_action"] = tool_seq[-1] if tool_seq else ""
            last_span = spans_so_far[-1] if spans_so_far else None
            features["last_observation"] = (
                (last_span.observation or "")[:200] if last_span else ""
            )
            features["unique_tools_so_far"] = len(set(tool_seq))
            features["history_summary"] = self._summarise_history(spans_so_far)

        return features

     #include_dependencies
         #Whether to include dependency features (tool_diversity, etc.).
    def _dependency_features(
        self, traj: Any, span_index: int | None
    ) -> dict[str, Any]:
        """Extract dependency / co-usage features."""
        features: dict[str, Any] = {}

        if span_index is None:
            n_tools = len(traj.tools_used)
            n_spans = traj.n_spans
        else:
            tool_seq = [
                s.tool_name for s in traj.spans[: span_index + 1] if s.tool_name
            ]
            n_tools = len(set(tool_seq))
            n_spans = span_index + 1

        features["tool_diversity"] = n_tools / max(n_spans, 1)

        if self.corpus_stats:
            features["app_tool_count"] = (
                self.corpus_stats.app_tool_count.get(traj.app_name, 0)
            )
        else:
            features["app_tool_count"] = 0

        return features

    @staticmethod
    def _summarise_history(spans: list) -> str:
        """Create a compact text summary of span history."""
        parts = []
        for i, span in enumerate(spans):
            tool = span.tool_name or "think"
            obs_snippet = (span.observation or "")[:50]
            parts.append(f"{tool}→{obs_snippet}")
        return " | ".join(parts[-5:])  # Last 5 spans max
