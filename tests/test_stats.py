"""Tests for resampling-based statistics (bootstrap CI + Holm)."""

from __future__ import annotations

import pytest

from shortchain.evaluation.statistics import (
    bootstrap_mean_ci,
    holm_bonferroni,
    paired_bootstrap_delta,
)


class TestBootstrapMeanCI:
    def test_known_mean(self):
        scores = {f"t{i}": 1.0 if i % 2 == 0 else 0.0 for i in range(100)}
        mean, lo, hi = bootstrap_mean_ci(scores, n_boot=1000, seed=7)
        assert mean == pytest.approx(0.5)
        assert lo <= mean <= hi

    def test_deterministic(self):
        scores = {f"t{i}": float(i % 3) for i in range(60)}
        a = bootstrap_mean_ci(scores, n_boot=500, seed=1)
        b = bootstrap_mean_ci(scores, n_boot=500, seed=1)
        assert a == b

    def test_empty(self):
        assert bootstrap_mean_ci({}) == (0.0, 0.0, 0.0)


class TestPairedBootstrapDelta:
    def test_a_better_than_b(self):
        a = {f"t{i}": 0.7 for i in range(80)}
        b = {f"t{i}": 0.3 for i in range(80)}
        mean, lo, hi = paired_bootstrap_delta(a, b, n_boot=1000, seed=3)
        assert mean == pytest.approx(0.4)
        assert lo > 0 and hi > 0  # CI excludes 0 -> significant

    def test_ties(self):
        a = {f"t{i}": 0.5 for i in range(50)}
        b = {f"t{i}": 0.5 for i in range(50)}
        mean, lo, hi = paired_bootstrap_delta(a, b, n_boot=500, seed=4)
        assert mean == pytest.approx(0.0)
        assert lo <= 0 <= hi


class TestHolmBonferroni:
    def test_simple(self):
        # two very small p-values -> reject both; one big -> not rejected
        rej = holm_bonferroni([0.001, 0.002, 0.9], alpha=0.05)
        assert rej == [True, True, False]

    def test_threshold(self):
        # with m=2: reject p <= 0.025 (first) and p <= 0.05 (second)
        rej = holm_bonferroni([0.024, 0.049], alpha=0.05)
        assert rej == [True, True]


class TestPairedPAndHolm:
    def test_p_small_when_difference_real(self):
        from shortchain.evaluation.statistics import paired_bootstrap_p_and_ci
        a = {f"t{i}": 0.8 for i in range(100)}
        b = {f"t{i}": 0.2 for i in range(100)}
        _, lo, hi, p = paired_bootstrap_p_and_ci(a, b, n_boot=1000, seed=5)
        assert lo > 0 and hi > 0
        assert p < 0.001

    def test_p_large_when_equal(self):
        from shortchain.evaluation.statistics import paired_bootstrap_p_and_ci
        a = {f"t{i}": 0.5 for i in range(60)}
        b = {f"t{i}": 0.5 for i in range(60)}
        _, lo, hi, p = paired_bootstrap_p_and_ci(a, b, n_boot=1000, seed=6)
        assert lo <= 0 <= hi
        assert p > 0.05
