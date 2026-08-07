"""Tests for selective prediction / LLM-fallback (hybrid) evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from shortchain.evaluation.hybrid import (
    area_under_risk_coverage,
    coverage_at_target_risk,
    coverage_risk_curve,
    hybrid_curve,
    topk_confidence_and_recall,
)


class TestTopKConfidence:
    def test_confidence_and_recall(self):
        scores = {"t1": np.array([0.9, 0.7, 0.2]), "t2": np.array([0.4, 0.8, 0.6])}
        labels = {"t1": np.array([1, 0, 0]), "t2": np.array([0, 0, 1])}
        out = topk_confidence_and_recall(scores, labels, k=2)
        # t1: top-2 = indices 0,1 -> conf=0.9, has relevant -> rk=1
        assert out["t1"][0] == pytest.approx(0.9) and out["t1"][1] == 1
        # t2: top-2 = indices 1,2 -> conf=0.8, relevant at 2 -> rk=1
        assert out["t2"][0] == pytest.approx(0.8) and out["t2"][1] == 1

    def test_confidence_excludes_low_scored_relevant(self):
        # relevant tool is ranked 3rd (outside top-2) -> conf from top-2, rk=0
        scores = {"t": np.array([0.9, 0.8, 0.7])}
        labels = {"t": np.array([0, 0, 1])}
        conf, rk = topk_confidence_and_recall(scores, labels, k=2)["t"]
        assert conf == pytest.approx(0.9) and rk == 0


class TestCoverageRisk:
    def test_perfect_high_conf_correct(self):
        conf = np.array([0.95, 0.9, 0.85, 0.6, 0.5])
        out = np.array([1, 1, 1, 1, 1])  # all correct
        curve = coverage_risk_curve(conf, out, np.array([0.0, 0.5, 0.9, 0.95]))
        # coverage decreases as threshold rises; risk stays 0
        assert curve["coverage"][0] == 1.0
        assert curve["coverage"][1] == 1.0
        assert curve["coverage"][2] == 0.4
        assert curve["risk"][2] == 0.0

    def test_discards_bad_low_conf(self):
        conf = np.array([0.95, 0.1])
        out = np.array([1, 0])
        curve = coverage_risk_curve(conf, out, np.array([0.5]))
        # keep only the high-conf correct decision
        assert curve["coverage"][0] == 0.5
        assert curve["risk"][0] == 0.0


class TestHybrid:
    def test_deferring_to_perfect_llm_reduces_risk(self):
        conf = np.array([0.95, 0.2, 0.3])
        local = np.array([1, 0, 0])
        llm = np.array([1, 1, 1])  # perfect LLM
        taus = np.array([0.99, 0.5, 0.0])
        curve = hybrid_curve(conf, local, llm, taus)
        # defer all (tau above every confidence) -> hybrid risk 0
        assert curve["hybrid_risk"][0] == 0.0
        # defer low-confidence only -> risk improves vs local-only (0.667)
        assert curve["hybrid_risk"][1] < 0.667
        # cost rises as we defer more
        assert curve["norm_cost"][0] > curve["norm_cost"][2]

    def test_coverage_at_target_risk(self):
        curve = {
            "coverage": np.array([1.0, 0.8, 0.5]),
            "hybrid_risk": np.array([0.5, 0.2, 0.0]),
            "tau": np.array([0.0, 0.5, 0.9]),
        }
        cov, tau = coverage_at_target_risk(curve, target_risk=0.25)
        assert cov == pytest.approx(0.8) and tau == pytest.approx(0.5)
        cov2, _ = coverage_at_target_risk(curve, target_risk=0.01)
        assert cov2 == pytest.approx(0.5)  # only the zero-risk point qualifies
        cov3, _ = coverage_at_target_risk(curve, target_risk=-0.01)
        assert np.isnan(cov3)

    def test_area(self):
        curve = {"coverage": np.array([0.0, 1.0]), "risk": np.array([1.0, 0.0])}
        a = area_under_risk_coverage(curve)
        assert a == pytest.approx(0.5)
