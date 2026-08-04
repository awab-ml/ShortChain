#!/usr/bin/env python3
"""Audit ToolBench train/eval splits for ShortChain compliance.

Checks:
1. Task-ID and query (intent) disjointness between train and eval.
2. Tool-level unseenness: for the G1/G2 unseen subsets, require that eval tasks'
   *relevant* tools are absent from the training corpus (the ToolBench "unseen
   tools" setting), reported per subset with a configurable floor.
3. Candidate-pool integrity: every eval task must have a non-empty ``api_list``
   candidate pool and at least one relevant API inside it.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shortchain.ingest.schema import Trajectory
from shortchain.integrations.toolbench import ToolBenchAdapter
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


def _tasks_to_trajectories(tasks: list[dict]) -> list[Trajectory]:
    """Cheap Trajectory proxies (task_id + query) for disjointness auditing."""
    return [
        Trajectory(task_id=t["task_id"], intent=t["query"], app_name=t["app_name"], spans=[])
        for t in tasks
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ToolBench splits for ShortChain compliance.")
    parser.add_argument("--data-dir", type=str, default="data/data (1).zip", help="Directory or zip containing ToolBench data.")
    parser.add_argument("--train-size", type=int, default=10000, help="Number of training trajectories to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Reservoir-sampling seed for the training split.")
    parser.add_argument("--subsets", type=str, default="G1_tool,G1_category,G2_category", help="Comma-separated eval subsets.")
    parser.add_argument("--min-unseen", type=int, default=400, help="Minimum accepted strictly-unseen eval tasks.")
    parser.add_argument("--output", type=str, default="data/toolbench_split_audit.json", help="Path to save audit report.")
    args = parser.parse_args()

    adapter = ToolBenchAdapter()
    data_path = Path(args.data_dir)
    if not data_path.exists():
        log.error(f"Data path does not exist: {data_path}")
        sys.exit(1)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    log.info(f"Loading {args.train_size} training trajectories (reservoir seed={args.seed})...")
    train_trajectories = adapter.load_trajectories(
        data_path, sample_size=args.train_size, random_state=args.seed
    )

    log.info(f"Loading evaluation tasks for subsets={subsets}...")
    eval_tasks = adapter.load_eval_tasks(data_path, subsets=subsets)

    if not train_trajectories or not eval_tasks:
        log.error("Failed to load training trajectories or evaluation tasks.")
        sys.exit(1)

    # 1. Task/query disjointness (purge leaking train tasks).
    eval_trajs = _tasks_to_trajectories(eval_tasks)
    clean_train, _ = adapter.sanitize_splits(train_trajectories, eval_trajs)
    disjoint_report = adapter.audit_split_compliance(clean_train, eval_trajs)

    # 2. Tool-level unseenness with a floor.
    unseen_report = adapter.audit_eval_task_unseenness(clean_train, eval_tasks)

    # 3. Candidate-pool integrity.
    missing_pool = [t for t in eval_tasks if not t.get("candidates")]
    no_relevant = [t for t in eval_tasks if not t.get("relevant_tools")]
    integrity = {
        "tasks_no_candidates": len(missing_pool),
        "tasks_no_relevant_in_pool": len(no_relevant),
    }

    strict_unseen = unseen_report["strictly_unseen_tasks"]
    floor_ok = strict_unseen >= args.min_unseen
    disjoint_ok = bool(disjoint_report["compliant"])

    report = {
        "audit_type": "toolbench_split_compliance_v2",
        "seed": args.seed,
        **disjoint_report,
        **unseen_report,
        "integrity": integrity,
        "floor": args.min_unseen,
        "floor_ok": floor_ok,
        "overall_ok": floor_ok and disjoint_ok and integrity["tasks_no_relevant_in_pool"] == 0,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("      TOOLBENCH SPLIT/UNSEEN COMPLIANCE AUDIT")
    print("=" * 70)
    print(f"Train trajectories (clean) : {disjoint_report['train_size']}")
    print(f"Eval tasks                 : {len(eval_tasks)}")
    print(f"Task-ID leakage            : {disjoint_report['task_id_leakage_count']}")
    print(f"Query leakage              : {disjoint_report['intent_leakage_count']}")
    print(f"Train unique tools         : {unseen_report['train_unique_tools']}")
    print(f"Strictly-unseen eval tasks : {strict_unseen}/{unseen_report['valid_tasks']} "
          f"({unseen_report['unseen_task_ratio']:.1%})")
    for src, b in unseen_report["per_subset"].items():
        print(f"    {src:<16} all_unseen={b['all_unseen']:>3}/{b['valid']:<3} "
              f"any_unseen={b['any_unseen']:>3} some_seen={b['some_seen']}")
    print(f"Tasks without candidate pool: {integrity['tasks_no_candidates']}")
    print(f"Tasks with no relevant in pool: {integrity['tasks_no_relevant_in_pool']}")
    print(f"Verdict                   : {'PASS' if report['overall_ok'] else 'FAIL'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
