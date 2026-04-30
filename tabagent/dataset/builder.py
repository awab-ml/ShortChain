"""Build (context, tool, label) training pairs from trajectories.

Implements the **pointwise reduction** strategy from the TabAgent paper:
for each trajectory, create positive pairs for tools actually used and
negative pairs for sampled tools from the catalog that were *not* used.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from tabagent.config import DatasetConfig
from tabagent.ingest.schema import Trajectory
from tabagent.utils.logging import get_logger

log = get_logger(__name__)


class DatasetBuilder:
    """Transform trajectories into a training DataFrame.

    Parameters
    ----------
    config
        Dataset construction settings.
    tool_catalog
        Optional explicit tool catalog ``{tool_name: description}``.
        If ``None``, the catalog is derived from all tools seen in the
        provided trajectories.
    """

    def __init__(
        self,
        config: DatasetConfig | None = None,
        tool_catalog: dict[str, str] | None = None,
    ) -> None:
        self.config = config or DatasetConfig()
        self._explicit_catalog = tool_catalog

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, trajectories: list[Trajectory]) -> pd.DataFrame:
        """Build the training dataset from a list of trajectories.

        Returns
        -------
        pd.DataFrame
            Columns: ``task_id, intent, app_name, n_steps, previous_tools,
            last_thought, tool_name, tool_description, label``
        """
        catalog = self._resolve_catalog(trajectories)
        log.info(f"Tool catalog size: [bold]{len(catalog)}[/bold]")

        rows: list[dict[str, Any]] = []
        for traj in trajectories:
            rows.extend(self._build_pairs(traj, catalog))

        df = pd.DataFrame(rows)
        pos = int(df["label"].sum())
        neg = len(df) - pos
        log.info(
            f"Built dataset: [bold green]{len(df)}[/bold green] rows "
            f"({pos} positive, {neg} negative, ratio ≈ {neg / max(pos, 1):.1f}:1)"
        )
        return df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_catalog(
        self, trajectories: list[Trajectory]
    ) -> dict[str, str]:
        """Build or validate the tool catalog."""
        if self._explicit_catalog:
            return self._explicit_catalog

        # Derive catalog from trajectories
        tool_counts: Counter[str] = Counter()
        for traj in trajectories:
            for tool in traj.tools_used:
                tool_counts[tool] += 1

        catalog = {tool: "" for tool in sorted(tool_counts)}
        log.info(
            f"Derived catalog from trajectories: {len(catalog)} unique tools "
            f"(top-5: {tool_counts.most_common(5)})"
        )
        return catalog

    def _build_pairs(
        self,
        traj: Trajectory,
        catalog: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Create positive + negative (context, tool, label) rows for one trajectory."""
        context = self._extract_context(traj)
        rows: list[dict[str, Any]] = []

        # Positive pairs: tools actually used
        for tool_name in traj.tools_used:
            row = {
                **context,
                "tool_name": tool_name,
                "tool_description": catalog.get(tool_name, ""),
                "label": 1,
            }
            rows.append(row)

        # Negative pairs: tools NOT used
        negative_pool = [t for t in catalog if t not in traj.tools_used]
        n_negatives = min(
            len(negative_pool),
            len(traj.tools_used) * self.config.negative_ratio,
        )

        if negative_pool and n_negatives > 0:
            sampled_negatives = random.sample(negative_pool, n_negatives)
            for tool_name in sampled_negatives:
                row = {
                    **context,
                    "tool_name": tool_name,
                    "tool_description": catalog.get(tool_name, ""),
                    "label": 0,
                }
                rows.append(row)

        return rows

    def _extract_context(self, traj: Trajectory) -> dict[str, Any]:
        """Extract context features from a trajectory."""
        return {
            "task_id": traj.task_id,
            "intent": traj.intent,
            "app_name": traj.app_name,
            "n_steps": traj.n_steps,
            "previous_tools": " | ".join(traj.tool_sequence),
            "last_thought": traj.last_thought or "",
        }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_dataset(
    trajectories: list[Trajectory],
    config: DatasetConfig | None = None,
    tool_catalog: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a training dataset from trajectories (convenience wrapper)."""
    builder = DatasetBuilder(config=config, tool_catalog=tool_catalog)
    return builder.build(trajectories)
