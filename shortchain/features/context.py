"""Context feature builder for ShortChain.

Extracts state-aware context features from a trajectory, optionally
at a specific step index.  Designed for trajectory-level use now
(``step_index=None``) with a clean migration path to step-level
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
        Whether to include state-aware features (step_index, last_action, etc.).
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

    def build(self, traj: Any, step_index: int | None = None) -> dict[str, Any]:
        """Extract context features from a trajectory.

        Parameters
        ----------
        traj
            A ``Trajectory`` object from ``shortchain.ingest.schema``.
        step_index
            If ``None`` (default), extracts trajectory-level features
            summarising the whole run.  If provided, extracts features
            as of that step (for future step-level decision modelling).

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

        if step_index is None:
            # Trajectory-level: summarise the whole run
            features["n_steps"] = traj.n_steps
            features["previous_tools"] = " | ".join(traj.tool_sequence)
            features["last_thought"] = traj.last_thought or ""
        else:
            # Step-level: summarise up to step_index
            steps_so_far = traj.steps[: step_index + 1]
            tool_seq = [s.tool_name for s in steps_so_far if s.tool_name]
            features["n_steps"] = len(steps_so_far)
            features["previous_tools"] = " | ".join(tool_seq)
            last = steps_so_far[-1] if steps_so_far else None
            features["last_thought"] = (last.thoughts or "") if last else ""

        # --- State features ---
        if self.include_state:
            features.update(self._state_features(traj, step_index))

        # --- Dependency features ---
        if self.include_dependencies:
            features.update(self._dependency_features(traj, step_index))

        return features

    def _state_features(self, traj: Any, step_index: int | None) -> dict[str, Any]:
        """Extract state-aware features."""
        features: dict[str, Any] = {}

        if step_index is None:
            # Trajectory-level state
            features["step_index"] = traj.n_steps  # final position
            seq = traj.tool_sequence
            features["last_action"] = seq[-1] if seq else ""
            # Last observation from final step
            last_step = traj.steps[-1] if traj.steps else None
            features["last_observation"] = (
                (last_step.observation or "")[:200] if last_step else ""
            )
            features["unique_tools_so_far"] = len(traj.tools_used)
            # History summary: compact representation
            features["history_summary"] = self._summarise_history(traj.steps)
        else:
            steps_so_far = traj.steps[: step_index + 1]
            features["step_index"] = step_index
            tool_seq = [s.tool_name for s in steps_so_far if s.tool_name]
            features["last_action"] = tool_seq[-1] if tool_seq else ""
            last_step = steps_so_far[-1] if steps_so_far else None
            features["last_observation"] = (
                (last_step.observation or "")[:200] if last_step else ""
            )
            features["unique_tools_so_far"] = len(set(tool_seq))
            features["history_summary"] = self._summarise_history(steps_so_far)

        return features

    def _dependency_features(
        self, traj: Any, step_index: int | None
    ) -> dict[str, Any]:
        """Extract dependency / co-usage features."""
        features: dict[str, Any] = {}

        if step_index is None:
            n_tools = len(traj.tools_used)
            n_steps = traj.n_steps
        else:
            tool_seq = [
                s.tool_name for s in traj.steps[: step_index + 1] if s.tool_name
            ]
            n_tools = len(set(tool_seq))
            n_steps = step_index + 1

        features["tool_diversity"] = n_tools / max(n_steps, 1)

        if self.corpus_stats:
            features["app_tool_count"] = (
                self.corpus_stats.app_tool_count.get(traj.app_name, 0)
            )
        else:
            features["app_tool_count"] = 0

        return features

    @staticmethod
    def _summarise_history(steps: list) -> str:
        """Create a compact text summary of step history."""
        parts = []
        for i, step in enumerate(steps):
            tool = step.tool_name or "think"
            obs_snippet = (step.observation or "")[:50]
            parts.append(f"{tool}→{obs_snippet}")
        return " | ".join(parts[-5:])  # Last 5 steps max
