"""Tests for the dense-retrieval (E5) baseline glue (no model download)."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_validation_mod", _ROOT / "scripts" / "run_validation.py"
)
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)


class _FakeEncoder:
    """Deterministic stand-in for SentenceTransformer.encode()."""

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        vecs = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            v = np.array(list(digest[:8]), dtype=np.float32)
            if normalize_embeddings:
                v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)


def test_dsr_score_task_matches_manual_dot():
    encoder = _FakeEncoder()
    names = ["beta__tool", "alpha__tool"]
    docs = [f"{n} | desc" for n in names]
    embs = encoder.encode([f"passage: {d}" for d in docs], normalize_embeddings=True)
    mat = embs.astype(np.float32)
    name_to_row = {n: i for i, n in enumerate(names)}

    scores = rv.dsr_score_task(encoder, mat, name_to_row, "please do beta",
                               [{"tool_name": "beta__tool"}, {"tool_name": "alpha__tool"}])

    # expected: recompute query dot
    q = _FakeEncoder().encode(["query: please do beta"], normalize_embeddings=True)[0]
    expected = [float(np.dot(q, mat[name_to_row["beta__tool"]])),
                float(np.dot(q, mat[name_to_row["alpha__tool"]]))]
    np.testing.assert_allclose(scores, expected, atol=1e-5)

    # unknown candidate -> 0.0
    scores2 = rv.dsr_score_task(encoder, mat, name_to_row, "q",
                                [{"tool_name": "unknown"}, {"tool_name": "beta__tool"}])
    assert scores2[0] == 0.0 and scores2[1] > 0.0


def test_timed_returns_elapsed():
    def fn(x):
        return x * 2
    result, ms = rv._timed(fn, 3)
    assert result == 6 and ms >= 0.0


def test_span_step_bucket_r1():
    scores = {"t:0": 0.0, "t:1": 1.0, "t:2": 0.0, "t:5": 1.0, "t:6": 1.0,
              "t:12": 1.0, "t:3": 0.0}
    out = rv._span_step_bucket_r1(scores)
    assert out["step_0"]["r1"] == 0.0 and out["step_0"]["n"] == 1
    assert out["steps_1_2"]["r1"] == pytest.approx(0.5) and out["steps_1_2"]["n"] == 2
    assert out["steps_3_9"]["r1"] == round(2 / 3, 4) and out["steps_3_9"]["n"] == 3
    assert out["steps_10+"]["r1"] == 1.0 and out["steps_10+"]["n"] == 1
