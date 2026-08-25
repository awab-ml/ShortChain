"""Tests for scripts/build_dataset.py --catalog (Section 7.1)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_runtime_trajectories(path: Path) -> Path:
    """A runtime-format JSONL: projected record shape, no tools_used key."""
    records = [
        {
            "task_id": "t1",
            "intent": "Refund order 9921",
            "app_name": "support-agent",
            "success": True,
            "spans": [
                {
                    "action": "lookup_order",
                    "observation": '{"status": "delivered"}',
                    "thoughts": "",
                    "metadata": {"tool_arguments": "{}", "projection.role": "tool"},
                },
                {
                    "action": "refund_order",
                    "observation": "ok",
                    "thoughts": "",
                    "metadata": {"tool_arguments": "{}", "projection.role": "tool"},
                },
            ],
            "metadata": {
                "source": "otel_openllmetry",
                "otel.trace_id": "a" * 32,
                "success_source": "association",
            },
        },
        {
            "task_id": "t2",
            "intent": "Book a flight",
            "app_name": "support-agent",
            "success": True,
            "spans": [
                {
                    "action": "search_flights",
                    "observation": "3 results",
                    "thoughts": "",
                    "metadata": {"tool_arguments": "{}", "projection.role": "tool"},
                }
            ],
            "metadata": {
                "source": "otel_openllmetry",
                "otel.trace_id": "b" * 32,
                "success_source": "association",
            },
        },
    ]
    path = path / "trajectories.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def run_script(trajectories: Path, *extra: str, tmp: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent.parent / "scripts" / "build_dataset.py"),
        "--trajectories",
        str(trajectories),
        "--output",
        str(tmp / "dataset"),
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestBuildDatasetWithCatalog:
    def test_end_to_end_with_derived_catalog(self, tmp_path: Path):
        traj = write_runtime_trajectories(tmp_path)
        result = run_script(traj, tmp=tmp_path)
        assert result.returncode == 0, result.stderr
        train = pd.read_csv(tmp_path / "dataset" / "train.csv")
        test = pd.read_csv(tmp_path / "dataset" / "test.csv")
        assert len(train) + len(test) > 0
        # Tools from the runtime trajectories are present as candidates.
        assert {"lookup_order", "refund_order"} <= set(train["tool_name"])

    def test_catalog_flag_supplies_descriptions(self, tmp_path: Path):
        traj = write_runtime_trajectories(tmp_path)
        catalog = {
            "lookup_order": "Look up an order by id",
            "refund_order": "Refund a delivered order",
            "search_flights": "Search available flights",
        }
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))
        result = run_script(traj, "--catalog", str(catalog_path), tmp=tmp_path)
        assert result.returncode == 0, result.stderr
        train = pd.read_csv(tmp_path / "dataset" / "train.csv")
        row = train[train["tool_name"] == "lookup_order"].iloc[0]
        assert row["tool_description"] == "Look up an order by id"

    def test_catalog_braods_candidate_pool(self, tmp_path: Path):
        """Catalog-only tools become negatives (they are not used)."""
        traj = write_runtime_trajectories(tmp_path)
        catalog = {
            "lookup_order": "Look up an order",
            "refund_order": "Refund",
            "search_flights": "Search flights",
            "cancel_flight": "Cancel a flight",  # never used in any trajectory
        }
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))
        result = run_script(traj, "--catalog", str(catalog_path), "--no-split", tmp=tmp_path)
        assert result.returncode == 0, result.stderr
        df = pd.read_csv(tmp_path / "dataset" / "full_dataset.csv")
        # The unused tool appears in the candidate pool with labels 0/1.
        assert "cancel_flight" in set(df["tool_name"])
        assert set(df[df["tool_name"] == "cancel_flight"]["label"]) == {0}