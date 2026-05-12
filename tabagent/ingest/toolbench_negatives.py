"""Failure-informed hard negative extraction for ToolBench G2.

Extracts tool choices from **failed** ToolBench trajectories and uses
them as hard negatives when training on successful trajectories.
Failed trajectories contain API calls that the model tried but that
ultimately didn't lead to task completion — these are excellent hard
negatives because they represent plausible-but-wrong tool choices.

Usage (hybrid strategy C)::

    # Load all trajectories (success_only=False)
    loader = ToolBenchLoader(catalog=catalog, success_only=False)
    all_trajs = loader.load_with_filter(path, scenario="G2")

    # Split into success / failure
    success = [t for t in all_trajs if t.success]
    failed  = [t for t in all_trajs if not t.success]

    # Extract failure negatives
    extractor = FailureNegativeExtractor(catalog=catalog)
    augmented_df = extractor.augment_negatives(
        train_df, failed_trajectories=failed, ratio=0.3,
    )
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import pandas as pd

from tabagent.features.context import ContextFeatureBuilder
from tabagent.features.stats import CorpusStats
from tabagent.features.tool import ToolFeatureBuilder
from tabagent.ingest.schema import Trajectory
from tabagent.utils.logging import get_logger

log = get_logger(__name__)


class FailureNegativeExtractor:
    """Extract hard negatives from failed ToolBench trajectories.

    Parameters
    ----------
    catalog
        Tool catalog ``{api_key: description}``.
    ratio
        Fraction of total negatives to source from failures (rest come
        from standard negative sampling). Default ``0.3`` (30%).
    random_state
        Seed for reproducibility.
    """

    def __init__(
        self,
        catalog: dict[str, str] | None = None,
        ratio: float = 0.3,
        random_state: int | None = 42,
    ) -> None:
        self.catalog = catalog or {}
        self.ratio = ratio
        self._rng = random.Random(random_state)

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_failed_tools(
        self,
        failed_trajectories: list[Trajectory],
    ) -> dict[str, set[str]]:
        """Map categories to tools that appeared in failed trajectories.

        Parameters
        ----------
        failed_trajectories
            Trajectories where ``success == False``.

        Returns
        -------
        dict[str, set[str]]
            ``{app_name → {tool_names that failed}}``
        """
        category_failures: dict[str, set[str]] = defaultdict(set)

        for traj in failed_trajectories:
            if traj.success:
                continue
            for tool in traj.tools_used:
                category_failures[traj.app_name].add(tool)

        log.info(
            f"Extracted failure negatives from {len(failed_trajectories)} "
            f"failed trajectories across {len(category_failures)} categories"
        )
        return dict(category_failures)

    def extract_intent_failures(
        self,
        failed_trajectories: list[Trajectory],
    ) -> list[tuple[str, str, set[str]]]:
        """Extract (intent, app_name, failed_tools) tuples.

        Each entry represents a failed attempt: the user's intent,
        the category, and which tools were tried but didn't work.

        Parameters
        ----------
        failed_trajectories
            Trajectories where ``success == False``.

        Returns
        -------
        list[tuple[str, str, set[str]]]
            ``[(intent, app_name, {failed_tools}), ...]``
        """
        results = []
        for traj in failed_trajectories:
            if traj.success or not traj.tools_used:
                continue
            results.append((traj.intent, traj.app_name, set(traj.tools_used)))
        return results

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def augment_negatives(
        self,
        train_df: pd.DataFrame,
        failed_trajectories: list[Trajectory],
        corpus_stats: CorpusStats | None = None,
    ) -> pd.DataFrame:
        """Add failure-informed hard negatives to a training DataFrame.

        For each unique ``(task_id, app_name)`` group in the training set,
        finds tools that failed for tasks in the same category and adds
        them as negative rows.

        The number of failure negatives added is controlled by
        ``self.ratio`` — approximately ``ratio * current_negatives``
        new rows are added.

        Parameters
        ----------
        train_df
            Training DataFrame from ``DatasetBuilder.build()``.
        failed_trajectories
            Failed trajectories (``success == False``).
        corpus_stats
            Optional corpus stats for feature building.

        Returns
        -------
        pd.DataFrame
            Augmented DataFrame with additional failure negatives.
        """
        if not failed_trajectories or self.ratio <= 0:
            return train_df

        # Build category → failed_tools mapping
        category_failures = self.extract_failed_tools(failed_trajectories)
        if not category_failures:
            log.info("No failure negatives found")
            return train_df

        # Count current negatives
        n_existing_neg = int((train_df["label"] == 0).sum())
        n_failure_neg = int(n_existing_neg * self.ratio)

        if n_failure_neg == 0:
            return train_df

        # Build feature builders
        context_builder = ContextFeatureBuilder(
            corpus_stats=corpus_stats,
            include_state=True,
            include_dependencies=True,
        )
        tool_builder = ToolFeatureBuilder(corpus_stats=corpus_stats)

        # Collect (app_name, task_id) groups from training data
        groups = train_df.groupby(["app_name", "task_id"]).first().reset_index()

        failure_rows: list[dict[str, Any]] = []
        budget = n_failure_neg

        for _, group_row in groups.iterrows():
            if budget <= 0:
                break

            app_name = group_row["app_name"]
            task_id = group_row["task_id"]

            # Get failed tools for this category
            failed_tools = category_failures.get(app_name, set())
            if not failed_tools:
                continue

            # Get positive tools for this task (to avoid adding them as negatives)
            task_mask = train_df["task_id"] == task_id
            task_positives = set(
                train_df.loc[task_mask & (train_df["label"] == 1), "tool_name"]
            )

            # Filter: only tools NOT already in this task's positives or negatives
            existing_tools = set(train_df.loc[task_mask, "tool_name"])
            candidate_negatives = list(failed_tools - existing_tools - task_positives)

            if not candidate_negatives:
                continue

            # Sample up to a few failure negatives per group
            n_pick = min(len(candidate_negatives), max(1, budget // max(len(groups), 1)))
            picked = self._rng.sample(
                candidate_negatives, min(n_pick, len(candidate_negatives))
            )

            # Build feature rows using the task's context
            context = {
                col: group_row[col]
                for col in train_df.columns
                if col not in ("label", "tool_name")
                and col in group_row.index
            }

            for tool_name in picked:
                tool_features = tool_builder.build(
                    tool_name,
                    tool_meta={"description": self.catalog.get(tool_name, "")},
                    context=context,
                )
                row = {**context, **tool_features, "label": 0}
                failure_rows.append(row)
                budget -= 1

                if budget <= 0:
                    break

        if failure_rows:
            failure_df = pd.DataFrame(failure_rows)
            # Align columns
            for col in train_df.columns:
                if col not in failure_df.columns:
                    failure_df[col] = 0
            failure_df = failure_df[train_df.columns]

            result = pd.concat([train_df, failure_df], ignore_index=True)
            log.info(
                f"Added [bold]{len(failure_rows)}[/bold] failure negatives "
                f"({len(failure_rows) / max(n_existing_neg, 1):.1%} of existing negatives)"
            )
            return result

        return train_df
