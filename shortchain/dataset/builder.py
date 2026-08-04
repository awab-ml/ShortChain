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

from collections import Counter
from typing import Any

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
        corpus_stats: CorpusStats | None = None,
    ) -> None:
        self.config = config or DatasetConfig()
        self.features_config = features_config or FeaturesConfig()
        self.negatives_config = negatives_config or NegativeSamplingConfig()
        self._explicit_catalog = tool_catalog
        # If provided, ``corpus_stats`` are pinned (frozen train statistics) and
        # ``build()``/``build_candidates()`` will NOT recompute them. This is the
        # mechanism that prevents evaluation features from leaking test-set
        # statistics (tool_frequency / co_occurrence / app_tool_count) into the
        # scored rows.
        self._corpus_stats: CorpusStats | None = corpus_stats

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
        # Freeze corpus stats on the FIRST build only; later calls (e.g. on an
        # evaluation set) keep using the training statistics, never recomputing
        # them from the incoming trajectories.
        if self._corpus_stats is None:
            self._corpus_stats = CorpusStats.from_trajectories(trajectories)
        catalog = self._resolve_catalog(trajectories)
        log.info(f"Tool catalog size: [bold]{len(catalog)}[/bold]")

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

        # --- Leakage guard ---
        if "previous_tools" in df.columns:
            negatives = df[df["label"] == 0]
            if len(negatives) > 0:
                leaked = negatives.apply(
                    lambda r: (
                        bool(r.get("previous_tools"))
                        and str(r["tool_name"]) in str(r["previous_tools"])
                    ),
                    axis=1,
                )
                if leaked.any():
                    raise ValueError(
                        f"TARGET LEAKAGE: {leaked.sum()} negative samples "
                        f"contain tool_name in previous_tools"
                    )

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

    def build_candidates(
        self,
        traj: Trajectory,
        candidates: list[dict[str, Any]],
        relevant_tools: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build pointwise (context, tool, label) rows for one trajectory against an
        explicit candidate set.

        This is the faithful evaluation path: the candidate pool is supplied by
        the caller (e.g. the query's ``api_list``) rather than generated by the
        negative sampler. No negatives are sampled here — every candidate is
        labelled 1 if it is in ``relevant_tools`` and 0 otherwise.

        Parameters
        ----------
        traj
            The trajectory (or lightweight proxy) holding ``task_id``, ``intent``
            and ``app_name`` for the context.
        candidates
            Explicit candidate tools, each ``{"tool_name": str, "tool_description": str}``.
        relevant_tools
            Set of candidate ``tool_name`` values that are ground-truth relevant
            (label = 1); everything else is label = 0.

        Returns
        -------
        list[dict]
            Pointwise rows using the SAME context/tool feature builders as the
            training path, so their columns/schema match the training DataFrame.
        """
        if self._corpus_stats is None:
            raise ValueError(
                "DatasetBuilder must have frozen corpus statistics before "
                "build_candidates() (pass corpus_stats= derived from the TRAIN "
                "set). Recomputing stats from evaluation data would leak test "
                "answers into the features."
            )

        relevant = {t for t in (relevant_tools or set()) if t}
        context_builder = ContextFeatureBuilder(
            corpus_stats=self._corpus_stats,
            include_state=self.features_config.include_state_features,
            include_dependencies=self.features_config.include_dependency_features,
        )
        tool_builder = ToolFeatureBuilder(corpus_stats=self._corpus_stats)
        context = context_builder.build(traj, span_index=None)

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for cand in candidates:
            tool_name = str(cand.get("tool_name", "")).strip()
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            desc = str(cand.get("tool_description") or "")
            rows.append(
                self._make_row(
                    context=context,
                    tool_name=tool_name,
                    description=desc,
                    label=1 if tool_name in relevant else 0,
                    tool_builder=tool_builder,
                )
            )
        return rows

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
            rows.append(
                self._make_row(
                    context=context,
                    tool_name=tool_name,
                    description=catalog.get(tool_name, ""),
                    label=1,
                    tool_builder=tool_builder,
                )
            )

        # Negative pairs via sampler
        n_negatives = len(traj.tools_used) * self.config.negative_ratio
        negative_tools = sampler.sample(
            positive_tools=traj.tools_used,
            app_name=traj.app_name,
            n=n_negatives,
        )
        for tool_name in negative_tools:
            rows.append(
                self._make_row(
                    context=context,
                    tool_name=tool_name,
                    description=catalog.get(tool_name, ""),
                    label=0,
                    tool_builder=tool_builder,
                )
            )

        return rows

    @staticmethod
    def _make_row(
        context: dict[str, Any],
        tool_name: str,
        description: str,
        label: int,
        tool_builder: ToolFeatureBuilder,
    ) -> dict[str, Any]:
        """Build a single pointwise row from a context + candidate tool features."""
        tool_features = tool_builder.build(
            tool_name,
            tool_meta={"description": description},
            context=context,
        )
        return {**context, **tool_features, "label": label}


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
