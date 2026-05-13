"""Confidence threshold tuning for tool selection.

Sweeps decision thresholds on validation data and selects the one
that maximizes a target metric (default: F1).  This addresses the
precision/recall imbalance where the default threshold of 0.5 is
too conservative — rejecting correct tools that have moderate but
not high confidence.

Usage::

    tuner = ThresholdTuner(target_metric="f1")
    best_t, sweep = tuner.find_optimal(y_true, y_proba, task_ids)
    print(f"Best threshold: {best_t}")
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from tabagent.utils.logging import get_logger

log = get_logger(__name__)


class ThresholdTuner:
    """Find the optimal confidence threshold for tool selection.

    Parameters
    ----------
    thresholds
        List of thresholds to evaluate.  Defaults to
        ``[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]``.
    target_metric
        Metric to maximize: ``"f1"``, ``"recall"``, or ``"precision"``.
    """

    DEFAULT_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    def __init__(
        self,
        thresholds: list[float] | None = None,
        target_metric: str = "f1",
    ) -> None:
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.target_metric = target_metric

    def find_optimal(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> tuple[float, dict[float, dict[str, float]]]:
        """Sweep thresholds and return the best one.

        Parameters
        ----------
        y_true
            Ground-truth binary labels.
        y_proba
            Predicted probabilities for the positive class.

        Returns
        -------
        tuple[float, dict]
            ``(best_threshold, {threshold: {metric: value, ...}, ...})``
        """
        results: dict[float, dict[str, float]] = {}

        for t in sorted(self.thresholds):
            y_pred = (y_proba >= t).astype(int)

            # Skip if all same class (degenerate)
            if len(np.unique(y_pred)) < 2:
                results[t] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
                continue

            results[t] = {
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }

        if not results:
            return 0.5, {}

        best_t = max(results, key=lambda t: results[t].get(self.target_metric, 0.0))

        log.info(
            f"Threshold sweep: best={best_t:.2f} "
            f"(F1={results[best_t]['f1']:.4f}, "
            f"P={results[best_t]['precision']:.4f}, "
            f"R={results[best_t]['recall']:.4f})"
        )

        return best_t, results
