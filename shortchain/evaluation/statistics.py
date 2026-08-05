"""Resampling-based statistics for ShortChain benchmark results.

Mirrors the TabAgent (Levy et al. 2026) evaluation protocol:

- Metrics are computed per task, then macro-averaged.
- 95% confidence intervals are obtained via **paired bootstrap** resampling
  over tasks (task alignment is preserved across methods).
- Pairwise contrasts use the bootstrap distribution of the macro-mean
  difference, significant when the bias-corrected CI excludes 0.
- Family-wise error is controlled **within each metric** with the
  Holm-Bonferroni procedure.
"""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    task_scores: dict[str, float],
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the macro-mean of a per-task score mapping.

    Parameters
    ----------
    task_scores
        ``{task_id: score}`` for every task (missing tasks are skipped).
    n_boot
        Number of bootstrap resamples over tasks.
    seed
        RNG seed for reproducibility.
    ci
        Confidence level.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, lower, upper)``.
    """
    ids = np.array(sorted(task_scores.keys()))
    if len(ids) == 0:
        return 0.0, 0.0, 0.0
    values = np.array([task_scores[i] for i in ids], dtype=float)
    mean = float(values.mean())

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(values)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = values[idx].mean()

    alpha = 1.0 - ci
    lo = float(np.percentile(boot, 100 * alpha / 2.0))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2.0)))
    return mean, lo, hi


def paired_bootstrap_samples(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    n_boot: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Return bootstrapped macro-mean differences ``(a - b)`` over tasks.

    Shared tasks are paired across the two methods before resampling.
    """
    shared = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not shared:
        return np.zeros(n_boot)
    d = np.array([scores_a[k] - scores_b[k] for k in shared], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(d)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = d[idx].mean()
    return boot


def paired_bootstrap_p_and_ci(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float, float]:
    """Two-sided paired-bootstrap p-value + CI for ``mean(a - b)``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(mean_delta, lower, upper, p_value)`` where ``p_value`` is the
        two-sided probability of the observed difference direction under cate.
    """
    shared = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not shared:
        return 0.0, 0.0, 0.0, 1.0
    d = np.array([scores_a[k] - scores_b[k] for k in shared], dtype=float)
    mean_d = float(d.mean())

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(d)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = d[idx].mean()

    alpha = 1.0 - ci
    lo = float(np.percentile(boot, 100 * alpha / 2.0))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2.0)))
    # Two-sided p = 2 * min(P(delta <= 0), P(delta >= 0)).
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return mean_d, lo, hi, p


def paired_bootstrap_delta(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of ``a - b`` over tasks (paired, per task).

    Returns
    -------
    tuple[float, float, float]
        ``(mean_delta, lower, upper)``.
    """
    mean_d, lo, hi, _ = paired_bootstrap_p_and_ci(
        scores_a, scores_b, n_boot=n_boot, seed=seed, ci=ci
    )
    return mean_d, lo, hi


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction over an ordered list of p-values.

    Parameters
    ----------
    p_values
        Raw two-sided p-values for ``m`` comparisons.
    alpha
        Family-wise error rate.

    Returns
    -------
    list[bool]
        Whether each null hypothesis is rejected.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected: set[int] = set()
    for j, i in enumerate(order, start=1):
        if p_values[i] <= alpha / (m - j + 1):
            rejected.add(i)
        else:
            break
    return [i in rejected for i in range(m)]
