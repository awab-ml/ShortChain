"""Evaluation metrics for tool shortlisting.

Implements the head-matched metrics from the ShortChain methodology:
- R-precision (P@R): adapts cutoff to the task's relevant-set size
- Recall@k: fixed-budget recovery at k ∈ {3, 5, 7, 9}
- Standard classification metrics: accuracy, precision, recall, F1, AUC
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def r_precision(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    task_ids: np.ndarray,
    tool_names: np.ndarray,
) -> float:
    """Compute macro-averaged R-precision (P@R).

    For each task ``t``, let ``R_t = |G(t)|`` be the number of relevant
    tools.  P@R retrieves the top-``R_t`` candidates and measures
    precision.

    .. math::

        P@R = \\frac{1}{|T|} \\sum_{t \\in T}
              \\frac{|S_{R_t}(t) \\cap G(t)|}{|G(t)|}

    Parameters
    ----------
    y_true
        Binary labels (1 = relevant).
    y_scores
        Predicted scores / probabilities.
    task_ids
        Task ID for each sample.
    tool_names
        Tool name for each sample.

    Returns
    -------
    float
        Macro-averaged R-precision.
    """
    unique_tasks = np.unique(task_ids)
    precisions = []

    for tid in unique_tasks:
        mask = task_ids == tid
        true = y_true[mask]
        scores = y_scores[mask]

        r = int(true.sum())  # |G(t)|
        if r == 0:
            continue

        # Get top-R predictions
        top_r_idx = np.argsort(scores)[::-1][:r]
        hits = true[top_r_idx].sum()
        precisions.append(hits / r)

    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    task_ids: np.ndarray,
    k: int,
) -> float:
    """Compute macro-averaged Recall@k.

    .. math::

        R@k = \\frac{1}{|T|} \\sum_{t \\in T}
              \\frac{|S_k(t) \\cap G(t)|}{|G(t)|}

    Parameters
    ----------
    y_true
        Binary labels.
    y_scores
        Predicted scores.
    task_ids
        Task ID per sample.
    k
        Fixed budget (number of candidates retrieved).

    Returns
    -------
    float
        Macro-averaged Recall@k.
    """
    unique_tasks = np.unique(task_ids)
    recalls = []

    for tid in unique_tasks:
        mask = task_ids == tid
        true = y_true[mask]
        scores = y_scores[mask]

        r = int(true.sum())
        if r == 0:
            continue

        top_k_idx = np.argsort(scores)[::-1][:k]
        hits = true[top_k_idx].sum()
        recalls.append(hits / r)

    return float(np.mean(recalls)) if recalls else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    X_val: pd.DataFrame | None = None,
    k_values: list[int] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Parameters
    ----------
    y_true
        Ground-truth binary labels.
    y_proba
        Predicted probabilities for the positive class.
    X_val
        Validation DataFrame (needs ``task_id`` and ``tool_name``
        columns for ranking metrics).
    k_values
        List of k values for Recall@k.
    threshold
        Decision threshold for binary predictions.

    Returns
    -------
    dict[str, float]
        Dictionary of metric name → value.
    """
    k_values = k_values or [3, 5, 7, 9]

    y_pred = (y_proba >= threshold).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # AUC (requires both classes present)
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))

    # Ranking metrics (require task_id grouping)
    if X_val is not None and "task_id" in X_val.columns:
        task_ids = X_val["task_id"].values
        tool_names = X_val["tool_name"].values if "tool_name" in X_val.columns else None

        metrics["r_precision"] = r_precision(
            y_true, y_proba, task_ids, tool_names
        )

        for k in k_values:
            metrics[f"recall_at_{k}"] = recall_at_k(y_true, y_proba, task_ids, k)

    return metrics


def format_metrics(metrics: dict[str, float], indent: int = 2) -> str:
    """Pretty-format a metrics dictionary for display."""
    lines = []
    for key, value in sorted(metrics.items()):
        lines.append(f"{' ' * indent}{key:>20s}: {value:.4f}")
    return "\n".join(lines)
