"""Probability calibration metrics and scalers for ShortChain.

The classifier's raw XGBoost scores are not guaranteed to be well calibrated
(i.e. a score of 0.8 does not necessarily mean the prediction is correct 80%
of the time). Calibration adjusts per-decision confidence so that a
confidence threshold becomes a meaningful, actionable "defer to LLM" rule.

Leak-safe usage: the calibrator must be fit only on TRAIN-task scores (e.g.
group-aware out-of-fold scores) and applied to test-task scores. Fitting it
on test data would leak — that is a hard invariant here.

How it is fitted (cross-fold, group-aware)
------------------------------------------
In the validation harness each calibration point is an *out-of-fold* decision
from a fold model trained without that task. Every fold's calibrator is fit on
the OTHER folds' points and applied to its own held-out fold — so no task ever
contributes to the calibrator that will score it. Because out-of-fold scores
carry real variance (unlike in-sample train predictions), the fit is
non-degenerate even when the underlying model memorises training tasks.

Choice of method
----------------
Platt (logistic on log-odds) is the default: it is robust with small OOF
samples. Isotonic is a monotone alternative when more calibration data is
available.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Weighted binning ECE: mean |acc(bin) - conf(bin)| * frac(bin)."""
    conf, acc, counts, _ = reliability_data(y_true, y_proba, n_bins)
    if counts.sum() == 0:
        return 0.0
    return float(np.sum(counts * np.abs(acc - conf)) / counts.sum())


def reliability_data(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(bin_conf, bin_acc, bin_counts, bin_edges)``."""
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, y_proba, side="right") - 1, 0, n_bins - 1)
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            bin_conf[b] = float(np.mean(y_proba[mask]))
            bin_acc[b] = float(np.mean(y_true[mask]))
    return bin_conf, bin_acc, counts, edges


class PlattCalibrator:
    """Platt scaling: logistic regression on the log-odds of the scores."""

    def __init__(self, max_iter: int = 1000) -> None:
        self.max_iter = max_iter
        self._model: LogisticRegression | None = None

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(y_proba / (1 - y_proba)).reshape(-1, 1)
        self._model = LogisticRegression(max_iter=self.max_iter, C=10_000.0)
        self._model.fit(logits, np.asarray(y_true, dtype=int))
        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PlattCalibrator.fit() must be called first")
        y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(y_proba / (1 - y_proba)).reshape(-1, 1)
        p = self._model.predict_proba(logits)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)


class IsotonicCalibrator:
    """Isotonic regression (monotone) mapping of scores to probabilities."""

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(np.asarray(y_proba, dtype=float), np.asarray(y_true, dtype=float))
        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator.fit() must be called first")
        return np.clip(self._model.predict(np.asarray(y_proba, dtype=float)), 1e-6, 1 - 1e-6)


def create_calibrator(method: str = "platt"):
    """Factory: ``platt`` (recommended, robust for small data) or ``isotonic``."""
    method = method.lower()
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method: {method!r} (platt | isotonic)")
