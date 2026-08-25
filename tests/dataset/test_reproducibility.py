"""Determinism regression: dataset construction must not depend on
process-level hash seed (PYTHONHASHSEED), so reproduction is exact.

The same corpus is built in two fresh subprocesses with different hash seeds
and the resulting feature digests must match.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_TRAJS = """
import json
trajs = json.loads(r'''{trajs}''')
"""


def _build_script() -> str:
    return """
import hashlib
import shortchain
from shortchain.config import DatasetConfig, NegativeSamplingConfig
from shortchain.dataset.builder import DatasetBuilder
from shortchain.ingest.schema import Trajectory, Span

trajs = [
    Trajectory(task_id="t1", intent="email john", app_name="gmail", spans=[
        Span(action="search_contacts"), Span(action="send_email")]),
    Trajectory(task_id="t2", intent="reply email", app_name="gmail", spans=[
        Span(action="search_contacts"), Span(action="reply_to_email")]),
    Trajectory(task_id="t3", intent="play song", app_name="spotify", spans=[
        Span(action="search_tracks"), Span(action="play_tracks"), Span(action="create_playlist")]),
    Trajectory(task_id="t4", intent="make playlist", app_name="spotify", spans=[
        Span(action="search_tracks"), Span(action="create_playlist")]),
    Trajectory(task_id="t5", intent="order item", app_name="amazon", spans=[
        Span(action="search_products"), Span(action="add_to_cart"), Span(action="place_order")]),
    Trajectory(task_id="t6", intent="contact support", app_name="gmail", spans=[
        Span(action="send_email"), Span(action="search_emails")]),
]
cfg = NegativeSamplingConfig(
    strategy="hard", random_state=7,
    same_app_weight=0.5, co_usage_weight=0.5, similarity_weight=0.0,
)
df = DatasetBuilder(
    config=DatasetConfig(negative_ratio=3), negatives_config=cfg
).build(trajs)
cols = ["task_id", "tool_name", "label", "tool_frequency",
        "tool_co_occurrence", "app_name"]
mat = df[cols].sort_values(cols).to_string()
print(hashlib.sha256(mat.encode()).hexdigest())
"""


@pytest.mark.parametrize("seed_a,seed_b", [("1", "2"), ("7", "12345")])
def test_dataset_build_is_hash_seed_stable(seed_a: str, seed_b: str):
    root = Path(__file__).resolve().parent.parent.parent
    env_a = dict(os.environ, PYTHONHASHSEED=seed_a)
    env_b = dict(os.environ, PYTHONHASHSEED=seed_b)
    code = _build_script()
    out_a = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env_a,
        capture_output=True, text=True, timeout=120,
    )
    out_b = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env_b,
        capture_output=True, text=True, timeout=120,
    )
    assert out_a.returncode == 0, out_a.stderr
    assert out_b.returncode == 0, out_b.stderr
    digest_a = out_a.stdout.strip().splitlines()[-1].strip()
    digest_b = out_b.stdout.strip().splitlines()[-1].strip()
    assert digest_a == digest_b, (
        "Dataset construction depends on PYTHONHASHSEED (reproducibility bug)"
    )
