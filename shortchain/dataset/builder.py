"""Build (context, tool, label) training pairs from trajectories.

Implements the **pointwise reduction** strategy from the ShortChain methodology:
for each trajectory, create positive pairs for tools actually used and
negative pairs for sampled tools from the catalog that were *not* used.

Phase 2 upgrades:
- Uses ``ContextFeatureBuilder`` for richer, state-aware context features.
- Uses ``ToolFeatureBuilder`` for corpus-enriched tool features.
- Delegates negative sampling to pluggable ``NegativeSampler`` strategies.
- Precomputes ``CorpusStats`` from the training trajectories.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from shortchain.config import (
    DatasetConfig,
    FeaturesConfig,
    NegativeSamplingConfig,
)
from shortchain.dataset.negatives import NegativeSampler, create_sampler
from shortchain.features.context import ContextFeatureBuilder
from shortchain.features.stats import CorpusStats
from shortchain.features.tool import ToolFeatureBuilder
from shortchain.ingest.schema import Trajectory
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class DatasetBuilder:
    """Transform trajectories into a training DataFrame.

    Parameters
    ----------
    config
        Dataset construction settings.
    features_config
        Feature pipeline settings (controls state / dependency features).
    negatives_config
        Negative sampling strategy settings.
    tool_catalog
        Optional explicit tool catalog ``{tool_name: description}``.
        If ``None``, the catalog is derived from all tools seen in the
        provided trajectories.
    """

    def __init__(
        self,
        config: DatasetConfig | None = None,
        features_config: FeaturesConfig | None = None,
        negatives_config: NegativeSamplingConfig | None = None,
        tool_catalog: dict[str, str] | None = None,
    ) -> None:
        self.config = config or DatasetConfig()
        self.features_config = features_config or FeaturesConfig()
        self.negatives_config = negatives_config or NegativeSamplingConfig()
        self._explicit_catalog = tool_catalog
        self._corpus_stats: CorpusStats | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, trajectories: list[Trajectory]) -> pd.DataFrame:
        """Build the training dataset from a list of trajectories.

        Returns
        -------
        pd.DataFrame
            Columns include context features, tool features, and ``label``.
        """
        catalog = self._resolve_catalog(trajectories)
        log.info(f"Tool catalog size: [bold]{len(catalog)}[/bold]")

        # Compute corpus stats for feature builders and samplers
        self._corpus_stats = CorpusStats.from_trajectories(trajectories)

        # Initialise feature builders
        context_builder = ContextFeatureBuilder(
            corpus_stats=self._corpus_stats,
            include_state=self.features_config.include_state_features,
            include_dependencies=self.features_config.include_dependency_features,
        )
        tool_builder = ToolFeatureBuilder(corpus_stats=self._corpus_stats)

        # Initialise negative sampler
        sampler = create_sampler(
            config=self.negatives_config,
            catalog=catalog,
            corpus_stats=self._corpus_stats,
        )

        rows: list[dict[str, Any]] = []
        for traj in trajectories:
            rows.extend(
                self._build_pairs(traj, catalog, context_builder, tool_builder, sampler)
            )

        df = pd.DataFrame(rows)
        pos = int(df["label"].sum())
        neg = len(df) - pos
        log.info(
            f"Built dataset: [bold green]{len(df)}[/bold green] rows "
            f"({pos} positive, {neg} negative, ratio ≈ {neg / max(pos, 1):.1f}:1)"
        )
        return df

    @property
    def corpus_stats(self) -> CorpusStats | None:
        """Access the computed corpus statistics (available after ``build()``)."""
        return self._corpus_stats

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
        context_builder: ContextFeatureBuilder,
        tool_builder: ToolFeatureBuilder,
        sampler: NegativeSampler,
    ) -> list[dict[str, Any]]:
        """Create positive + negative (context, tool, label) rows for one trajectory."""
        context = context_builder.build(traj, span_index=None)
        rows: list[dict[str, Any]] = []

        # Positive pairs: tools actually used
        for tool_name in traj.tools_used:
            tool_features = tool_builder.build(
                tool_name,
                tool_meta={"description": catalog.get(tool_name, "")},
                context=context,
            )
            row = {**context, **tool_features, "label": 1}
            rows.append(row)

        # Negative pairs via sampler
        n_negatives = len(traj.tools_used) * self.config.negative_ratio
        negative_tools = sampler.sample(
            positive_tools=traj.tools_used,
            app_name=traj.app_name,
            n=n_negatives,
        )
        for tool_name in negative_tools:
            tool_features = tool_builder.build(
                tool_name,
                tool_meta={"description": catalog.get(tool_name, "")},
                context=context,
            )
            row = {**context, **tool_features, "label": 0}
            rows.append(row)

        return rows


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_dataset(
    trajectories: list[Trajectory],
    config: DatasetConfig | None = None,
    features_config: FeaturesConfig | None = None,
    negatives_config: NegativeSamplingConfig | None = None,
    tool_catalog: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a training dataset from trajectories (convenience wrapper)."""
    builder = DatasetBuilder(
        config=config,
        features_config=features_config,
        negatives_config=negatives_config,
        tool_catalog=tool_catalog,
    )
    return builder.build(trajectories)
