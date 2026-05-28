"""Single-pass inference engine.

Loads a trained classifier, scores candidate tools against a context,
and returns a ranked shortlist.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from shortchain.head.classifier import ShortChainClassifier
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class InferenceEngine:
    """Run inference with a trained ShortChain classifier.

    Parameters
    ----------
    model_path
        Path to a saved ``.pkl`` model file.
    top_k
        Default number of tools to return per task.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        classifier: ShortChainClassifier | None = None,
        top_k: int = 7,
    ) -> None:
        if classifier is not None:
            self.classifier = classifier
        elif model_path is not None:
            self.classifier = ShortChainClassifier.load(model_path)
        else:
            raise ValueError("Provide either model_path or classifier")
        self.top_k = top_k

    def predict(
        self,
        context: dict[str, Any],
        candidates: list[dict[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Score candidate tools against a single context.

        Parameters
        ----------
        context
            Dict with keys: ``intent``, ``app_name``, ``n_steps``,
            ``previous_tools``, ``last_thought``.
        candidates
            List of dicts with keys: ``tool_name``, ``tool_description``.
        top_k
            Override the default shortlist size.

        Returns
        -------
        list[tuple[str, float]]
            ``(tool_name, confidence)`` sorted by descending confidence.
        """
        k = top_k or self.top_k

        # Build a DataFrame: one row per candidate
        rows = []
        for cand in candidates:
            row = {**context, **cand}
            row.setdefault("task_id", "inference")
            rows.append(row)

        df = pd.DataFrame(rows)

        start = time.perf_counter()
        scores = self.classifier.predict_proba(df)
        latency_ms = (time.perf_counter() - start) * 1000

        # Rank and return top-K
        ranked = sorted(
            zip(df["tool_name"].values, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        shortlist = [(str(name), float(score)) for name, score in ranked[:k]]

        log.info(
            f"Inference: {len(candidates)} candidates → top-{k} in {latency_ms:.1f}ms"
        )
        return shortlist

    def predict_batch(
        self,
        df: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Score candidates for multiple tasks at once.

        Parameters
        ----------
        df
            DataFrame with ``task_id``, context columns, and candidate
            columns.
        top_k
            Number of tools to return per task.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            Mapping from ``task_id`` to ranked shortlist.
        """
        k = top_k or self.top_k

        start = time.perf_counter()
        scores = self.classifier.predict_proba(df)
        latency_ms = (time.perf_counter() - start) * 1000

        results: dict[str, list[tuple[str, float]]] = {}
        for task_id in df["task_id"].unique():
            mask = df["task_id"] == task_id
            task_tools = df.loc[mask, "tool_name"].values
            task_scores = scores[mask]
            ranked = sorted(
                zip(task_tools, task_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            results[str(task_id)] = [
                (str(t), float(s)) for t, s in ranked[:k]
            ]

        n_tasks = len(results)
        log.info(
            f"Batch inference: {n_tasks} tasks, {len(df)} total candidates "
            f"in {latency_ms:.1f}ms ({latency_ms / max(n_tasks, 1):.1f}ms/task)"
        )
        return results
