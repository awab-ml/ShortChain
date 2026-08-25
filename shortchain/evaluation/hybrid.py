"""Selective prediction & LLM-fallback (hybrid) evaluation for ShortChain.

Deployment pattern (complementary deployment):
- ShortChain makes the decision (its top-k shortlist) when its calibrated
  top-k confidence is high.
- When confidence is below a threshold, the decision is *deferred* to an LLM
  shortlister (the generative component being replaced).

Everything here is a pure function of per-task arrays so it can be wrapped in
the existing paired-bootstrap / CI machinery. Thresholds are swept and
reported as a risk-coverage curve; they are never tuned on the test set.

Terminology
-----------
- A *decision* is a binary outcome: whether the produced shortlist was
  correct (e.g. top-R exactly relevant at task level, next-tool correct at
  span level). Its raw/calibrated probability is the *confidence*.
- *Coverage* = fraction of decisions handled locally (kept).
- *Risk* = 1 - accuracy (share of incorrect decisions).
- The **risk-coverage curve** sweeps the deferral threshold: as confidence
  must be higher to keep a decision, coverage falls and (usually) risk on the
  kept set falls. Integrating risk over coverage summarises the trade-off.
- **Coverage at parity risk** asks: at what coverage can we stay at or above
  a target accuracy (e.g. the LLM-only accuracy) — the honest answer to "how
  much can we afford to keep local?".
"""

from __future__ import annotations

import numpy as np


def topk_confidence_and_recall(
    scores_by_task: dict[str, np.ndarray],
    labels_by_task: dict[str, np.ndarray],
    k: int,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, int]]:
    """Per-task ``(max top-k probability, Recall@k)``.

    - ``confidence`` = the highest predicted probability inside the model's
      top-``k`` ranked candidates (the user-selected confidence definition).
    - ``recall@k`` = 1 if any of the top-``k`` candidates is relevant.
    Ties are broken deterministically by tool id to pick a stable top-k.
    """
    out: dict[str, tuple[float, int]] = {}
    for task_id, scores in scores_by_task.items():
        labels = labels_by_task[task_id]
        # stable top-k: sort by (-score, index) to break ties deterministically
        order = np.lexsort((np.arange(len(scores)), -scores))
        topk = order[:k]
        conf = float(np.max(scores[topk])) if len(topk) else 0.0
        rk = int(int(labels[topk].sum()) > 0)
        out[task_id] = (conf, rk)
    return out


def coverage_risk_curve(
    confidences: np.ndarray,
    outcomes: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Local-only selective curve: coverage vs risk on the *kept* decisions.

    Returns dict with ``tau``, ``coverage`` (fraction kept), ``risk``
    (1 - accuracy among kept); empty kept bins default risk to 1.0.
    """
    tau = []
    coverage = []
    risk = []
    for t in thresholds:
        kept = confidences >= t
        c = float(kept.mean())
        r = 1.0
        if kept.any():
            r = 1.0 - float(outcomes[kept].mean())
        tau.append(float(t))
        coverage.append(c)
        risk.append(r)
    return {"tau": np.asarray(tau), "coverage": np.asarray(coverage),
            "risk": np.asarray(risk)}


def hybrid_curve(
    confidences: np.ndarray,
    local_outcomes: np.ndarray,
    deferred_outcomes: np.ndarray,
    thresholds: np.ndarray,
    local_cost: float = 1.0,
    deferred_cost: float = 100.0,
    local_latency: float = 1.0,
    deferred_latency: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Hybrid (local + defer-to-LLM) risk–coverage, cost and latency.

    ``deferred_outcomes`` are the per-decision correctness values of the LLM
    baseline on every decision (used only for the decisions actually deferred).
    Costs/latencies are relative to ``local_cost/local_latency`` = 1.
    """
    n = len(confidences)
    tau = []
    coverage = []
    hybrid_risk = []
    norm_cost = []
    norm_latency = []
    for t in thresholds:
        deferred = confidences < t
        kept = ~deferred
        acc = np.zeros(n, dtype=float)
        acc[kept] = local_outcomes[kept]
        acc[deferred] = deferred_outcomes[deferred]
        hybrid_risk.append(1.0 - float(acc.mean()))
        cov = float(kept.mean())
        cost = (cov * local_cost + (1 - cov) * deferred_cost) / n * n  # per-decision avg
        cost = cov * local_cost + (1 - cov) * deferred_cost
        lat = cov * local_latency + (1 - cov) * deferred_latency
        tau.append(float(t))
        coverage.append(cov)
        norm_cost.append(cost)
        norm_latency.append(lat)
    return {"tau": np.asarray(tau), "coverage": np.asarray(coverage),
            "hybrid_risk": np.asarray(hybrid_risk),
            "norm_cost": np.asarray(norm_cost),
            "norm_latency": np.asarray(norm_latency)}


def coverage_at_target_risk(
    curve: dict[str, np.ndarray],
    target_risk: float,
    risk_key: str = "hybrid_risk",
) -> tuple[float, float]:
    """Largest coverage whose risk is at most ``target_risk``.

    Returns ``(coverage, tau)`` (NaN coverage if no point meets the target).
    """
    risk = curve[risk_key]
    cov = curve["coverage"]
    tau = curve["tau"]
    ok = risk <= target_risk
    if not ok.any():
        return float("nan"), float("nan")
    idx = int(np.argmax(cov * ok))
    return float(cov[idx]), float(tau[idx])


def area_under_risk_coverage(
    curve: dict[str, np.ndarray], risk_key: str = "risk"
) -> float:
    """Integrate risk over coverage (lower is better)."""
    cov = np.asarray(curve["coverage"])
    risk = np.asarray(curve[risk_key])
    order = np.argsort(cov)
    return float(np.trapezoid(risk[order], cov[order]))
