#!/usr/bin/env python3
"""Data preparation script for ToolBench Phase G1 benchmark.

Converts raw ToolBench G1 answer files and query metadata into ShortChain-compatible
JSONL files:
- data/toolbench/g1_train.jsonl
- data/toolbench/g1_test.jsonl
- data/toolbench/g1_catalog.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.progress import track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ToolBench Phase G1 dataset for ShortChain benchmark.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/toolbench",
        help="Path to toolbench data root directory.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit number of train records for fast testing.",
    )
    return parser.parse_args()


def extract_spans_and_tools(rec: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    """Walk tree structure of a ToolBench answer record to extract spans and tool names."""
    spans: list[dict[str, Any]] = []
    tools_used: set[str] = set()

    def walk(node: dict[str, Any], current_thought: str | None = None) -> None:
        if not isinstance(node, dict):
            return
        ntype = node.get("node_type")
        desc = node.get("description", "")
        thought = current_thought

        if ntype == "Thought":
            thought = desc
        elif ntype == "Action":
            if desc and desc != "Finish":
                tools_used.add(desc)
                obs = ""
                for c in node.get("children", []):
                    if isinstance(c, dict) and c.get("node_type") == "Action Input":
                        obs = str(c.get("observation", ""))
                        break
                spans.append({
                    "action": desc,
                    "thoughts": thought or "",
                    "observation": obs[:500],
                })

        for c in node.get("children", []):
            if isinstance(c, dict):
                walk(c, thought)

    if "tree" in rec and isinstance(rec["tree"], dict) and "tree" in rec["tree"]:
        walk(rec["tree"]["tree"])

    return spans, tools_used


def main() -> None:
    args = parse_args()
    base_dir = Path(args.data_dir)

    g1_query_file = base_dir / "data" / "instruction" / "G1_query.json"
    g1_answer_dir = base_dir / "data" / "answer" / "G1_answer"
    test_ids_file = base_dir / "data" / "test_query_ids" / "G1_instruction.json"
    catalog_file = base_dir / "catalog.json"

    print("Checking input paths:")
    print(f"  G1 queries: {g1_query_file}")
    print(f"  G1 answers: {g1_answer_dir}")
    print(f"  Test IDs:   {test_ids_file}")
    print(f"  Catalog:    {catalog_file}")

    # 1. Load G1 query metadata
    with open(g1_query_file, "r") as f:
        g1_queries = json.load(f)
    g1_query_map = {str(q["query_id"]): q for q in g1_queries}
    print(f"Loaded {len(g1_query_map)} G1 query metadata records.")

    # 2. Load G1 instruction test query IDs (200 queries)
    with open(test_ids_file, "r") as f:
        test_ids_dict = json.load(f)
    test_qids = set(str(k) for k in test_ids_dict.keys())
    print(f"Loaded {len(test_qids)} test query IDs.")

    # 3. Process answer files
    answer_files = sorted([f for f in g1_answer_dir.iterdir() if f.name.endswith(".json")])
    print(f"Found {len(answer_files)} G1 answer files.")

    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for fpath in track(answer_files, description="Processing answer trajectories..."):
        qid = fpath.name.split("_")[0]
        try:
            with open(fpath, "r") as f:
                rec = json.load(f)
        except Exception as exc:
            print(f"Warning: Failed to load {fpath}: {exc}")
            continue

        win = bool(rec.get("win", False))
        ag = rec.get("answer_generation", {})
        query_text = str(ag.get("query", ""))

        qinfo = g1_query_map.get(qid, {})
        if not query_text and qinfo:
            query_text = str(qinfo.get("query", ""))

        app_name = ""
        if qinfo and qinfo.get("api_list"):
            app_name = str(qinfo["api_list"][0].get("category_name", ""))

        spans, tools_used = extract_spans_and_tools(rec)

        record = {
            "task_id": qid,
            "intent": query_text,
            "app_name": app_name,
            "success": win,
            "spans": spans,
            "tools_used": sorted(tools_used),
        }

        if qid in test_qids:
            test_records.append(record)
        else:
            train_records.append(record)

    print(f"Total processed: {len(train_records)} train, {len(test_records)} test")

    if args.max_records and len(train_records) > args.max_records:
        print(f"Limiting training records to {args.max_records}")
        train_records = train_records[: args.max_records]

    # Write output JSONL files
    out_train = base_dir / "g1_train.jsonl"
    out_test = base_dir / "g1_test.jsonl"

    print(f"Writing train dataset to {out_train}...")
    with open(out_train, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    print(f"Writing test dataset to {out_test}...")
    with open(out_test, "w") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    # Link catalog
    out_catalog = base_dir / "g1_catalog.json"
    if catalog_file.exists():
        print(f"Ensuring catalog at {out_catalog}...")
        with open(catalog_file, "r") as f:
            cat_data = json.load(f)
        with open(out_catalog, "w") as f:
            json.dump(cat_data, f)

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
