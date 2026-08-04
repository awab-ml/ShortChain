#!/usr/bin/env python3
"""Run ShortChain's core pipeline on the ToolBench G1/G2 unseen-tools benchmark.

Faithful, leakage-free protocol
-------------------------------
1. Reservoir-sample ``--train-size`` trajectories (default 10,000) from
   ``toolllama_G123_dfs_train.json``.
2. Build the pointwise ``(context, tool, label)`` training DataFrame with a
   ``DatasetBuilder`` whose corpus statistics (``tool_frequency``,
   ``co_occurrence``, ``app_tool_count``) are frozen on the TRAIN set only.
3. Load eval tasks from ``test_instruction`` (default G1_tool + G1_category +
   G2_category). For every task the candidate pool IS its own ``api_list`` (the
   genuine set of available tools) — NOT negative sampling. Positives are the
   ``relevant APIs``; the rest of ``api_list`` are negatives.
4. Score every candidate with the trained classifier, rank per task, and compute
   R-Precision + Recall@k over that honest pool. Evaluation rows reuse the SAME
   feature builders with the SAME frozen train statistics — nothing about the
   evaluation set is visible at scoring time.
5. Report metrics overall, per-subset, and for the strictly-unseen subset
   (tasks whose every relevant tool is absent from the training corpus).
6. Latency is measured per task over the real candidate pools.

Classification metrics (accuracy/F1/AUC) are computed only as diagnostics and
are NOT the benchmark result. There is deliberately no ``2000``-task target:
the faithful eval set is exactly the instruction files requested via ``--subsets``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from shortchain.config import load_config
from shortchain.dataset.builder import DatasetBuilder
from shortchain.evaluation.metrics import compute_metrics
from shortchain.head.trainer import Trainer
from shortchain.ingest.schema import Trajectory
from shortchain.integrations.toolbench import ToolBenchAdapter
from shortchain.utils.logging import get_logger

log = get_logger(__name__)

_DIAGNOSTIC_KEYS = {"accuracy", "precision", "recall", "f1", "auc", "recall_at_10"}


def _build_eval_rows(eval_builder, eval_tasks, train_tools: set[str]) -> pd.DataFrame:
    """Build pointwise eval rows (candidate pool = task api_list) with frozen train stats."""
    rows: list[dict] = []
    for task in eval_tasks:
        traj = Trajectory(
            task_id=task["task_id"],
            intent=task["query"],
            app_name=task["app_name"],
            spans=[],
            success=True,
        )
        relevant = set(task.get("relevant_tools") or set())
        strictly_unseen = bool(
            relevant and not (relevant & train_tools)
        )
        for row in eval_builder.build_candidates(traj, task["candidates"], relevant):
            row["source"] = task["source"]
            row["strictly_unseen"] = int(strictly_unseen)
            row["n_candidates"] = len(task["candidates"])
            rows.append(row)
    return pd.DataFrame(rows)


def _subset_metrics(
    df: pd.DataFrame, y: pd.Series, proba: pd.Series, k_values: list[int]
) -> dict[str, float]:
    X = df.drop(columns=["label"])
    metrics = compute_metrics(y.values, proba.values, X_val=X, k_values=k_values)
    return {k: v for k, v in metrics.items() if k not in _DIAGNOSTIC_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShortChain benchmark on ToolBench unseen tasks.")
    parser.add_argument("--config", type=str, default="configs/toolbench.yaml", help="Path to config file.")
    parser.add_argument("--data-dir", type=str, default="data/data (1).zip", help="Path to ToolBench data file/dir or zip.")
    parser.add_argument("--train-size", type=int, default=10000, help="Number of training trajectories to reservoir-sample.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for train reservoir sampling and negative sampling.")
    parser.add_argument("--subsets", type=str, default="G1_tool,G1_category,G2_category", help="Comma-separated eval subsets.")
    parser.add_argument("--output-dir", type=str, default="models/toolbench_2000eval", help="Output directory for model and metrics.")
    parser.add_argument("--folds", type=int, default=10, help="Cross-validation folds.")
    args = parser.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Configuration
    config = load_config(args.config)
    if args.folds:
        config.splitter.n_folds = args.folds
    if args.seed:
        config.negatives.random_state = args.seed

    # 2. Load training trajectories (uniform reservoir sample) + tool catalog
    adapter = ToolBenchAdapter()
    data_path = Path(args.data_dir)
    if not data_path.exists():
        log.error(f"Data path does not exist: {data_path}")
        sys.exit(1)

    log.info(f"Loading {args.train_size} training trajectories (seed={args.seed})...")
    train_trajs = adapter.load_trajectories(
        data_path, sample_size=args.train_size, random_state=args.seed
    )
    catalog = adapter.load_catalog(data_path)
    log.info(f"Tool catalog (toolenv) size: {len(catalog)}")

    # 3. Load faithful eval tasks (candidate pool = api_list)
    log.info(f"Loading eval tasks for subsets={subsets}...")
    eval_tasks = adapter.load_eval_tasks(data_path, subsets=subsets)
    if not eval_tasks:
        log.error("No evaluation tasks loaded.")
        sys.exit(1)

    # 4. Task/query disjointness sanitization + tool-level unseen audit
    eval_trajs = [
        Trajectory(task_id=t["task_id"], intent=t["query"], app_name=t["app_name"], spans=[])
        for t in eval_tasks
    ]
    clean_train, _ = adapter.sanitize_splits(train_trajs, eval_trajs)
    disjoint = adapter.audit_split_compliance(clean_train, eval_trajs)
    unseen = adapter.audit_eval_task_unseenness(clean_train, eval_tasks)
    log.info(
        f"Disjointness: compliant={disjoint['compliant']} | "
        f"strictly-unseen eval tasks={unseen['strictly_unseen_tasks']}/{unseen['valid_tasks']}"
    )

    # 5. Build training dataset (corpus stats frozen here)
    train_builder = DatasetBuilder(
        config=config.dataset,
        features_config=config.features,
        negatives_config=config.negatives,
        tool_catalog=catalog or None,
    )
    train_df = train_builder.build(clean_train)
    train_tools = set(train_df["tool_name"].unique())
    log.info(f"Train rows: {len(train_df)} | unique train tools: {len(train_tools)}")

    # 6. Build evaluation rows with a builder that reuses the FROZEN train stats
    eval_builder = DatasetBuilder(
        config=config.dataset,
        features_config=config.features,
        negatives_config=config.negatives,
        tool_catalog=catalog or None,
        corpus_stats=train_builder.corpus_stats,
    )
    eval_df = _build_eval_rows(eval_builder, eval_tasks, train_tools)
    # Leak check: positive rows of *strictly-unseen* tasks must carry zero
    # train-derived tool_frequency; any nonzero value would mean the features
    # were recomputed from evaluation data.
    unseen_pos = eval_df[(eval_df["strictly_unseen"] == 1) & (eval_df["label"] == 1)]
    if len(unseen_pos) > 0:
        leaked_freq = float((unseen_pos["tool_frequency"] != 0).mean())
        if leaked_freq > 0.01:
            log.warning(
                f"Leak check FAILED: {leaked_freq:.1%} of strictly-unseen positive "
                "eval rows have nonzero tool_frequency — corpus stats were recomputed."
            )
        else:
            log.info(f"Leak check passed: {leaked_freq:.1%} strictly-unseen positives have nonzero train tool_frequency.")
    log.info(f"Eval rows: {len(eval_df)} over {eval_df['task_id'].nunique()} tasks")

    # 7. Train classifier with group-aware CV + final model
    trainer = Trainer(
        classifier_config=config.classifier,
        splitter_config=config.splitter,
        eval_config=config.evaluation,
    )
    cv_results = trainer.train_with_cv(train_df)
    model_path = output_dir / "shortchain_toolbench.pkl"
    final_model = trainer.train_final(train_df, save_path=model_path)

    # 8. Faithful evaluation over api_list candidate pools
    X_test = eval_df.drop(columns=["label"])
    y_test = eval_df["label"]
    start = time.perf_counter()
    y_proba = pd.Series(final_model.predict_proba(X_test), index=eval_df.index)
    total_ms = (time.perf_counter() - start) * 1000.0
    n_tasks = eval_df["task_id"].nunique()
    latency_per_task_ms = total_ms / max(1, n_tasks)

    overall = _subset_metrics(eval_df, y_test, y_proba, config.evaluation.k_values)
    overall["latency_ms_per_task"] = round(latency_per_task_ms, 4)
    overall["eval_tasks"] = n_tasks
    overall["mean_candidates_per_task"] = round(eval_df["n_candidates"].mean(), 2)

    # Chance-level (random) baseline for context — uniform random scores preserve
    # the candidate-pool sizes but assign ranks at random.
    rng = np.random.default_rng(2024)
    random_scores = pd.Series(rng.random(size=len(eval_df)), index=eval_df.index)
    baseline = _subset_metrics(
        eval_df, y_test, random_scores, config.evaluation.k_values
    )
    overall["random_baseline_r_precision"] = round(baseline.get("r_precision", 0.0), 4)
    overall["random_baseline_recall_at_1"] = round(baseline.get("recall_at_1", 0.0), 4)
    overall["random_baseline_recall_at_3"] = round(baseline.get("recall_at_3", 0.0), 4)

    # Per-subset + strictly-unseen breakdowns
    breakdown = {"overall": overall}
    for source in subsets:
        mask = eval_df["source"] == source
        if mask.sum() == 0:
            continue
        sub = eval_df[mask]
        breakdown[source] = _subset_metrics(
            sub, sub["label"], y_proba[mask], config.evaluation.k_values
        )
        breakdown[source]["eval_tasks"] = sub["task_id"].nunique()

    unseen_mask = eval_df["strictly_unseen"] == 1
    if unseen_mask.sum() > 0:
        sub = eval_df[unseen_mask]
        breakdown["strictly_unseen"] = _subset_metrics(
            sub, sub["label"], y_proba[unseen_mask], config.evaluation.k_values
        )
        breakdown["strictly_unseen"]["eval_tasks"] = sub["task_id"].nunique()

    # 9. Aggregate diagnostics (not the benchmark result)
    diag = compute_metrics(y_test.values, y_proba.values, X_val=X_test)
    diagnostics = {k: diag[k] for k in ("accuracy", "precision", "recall", "f1", "auc") if k in diag}

    results = {
        "benchmark": "ToolBench_g1_g2_unseen",
        "protocol_note": (
            "candidate pool = each query's api_list; corpus stats frozen on train; "
            "no negative sampling at eval time"
        ),
        "train_size": args.train_size,
        "seed": args.seed,
        "subsets": subsets,
        "split_audit": disjoint,
        "unseen_audit": unseen,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(eval_df)),
        "test_tasks": n_tasks,
        "cv_results": cv_results.get("aggregate", {}),
        "headline_metrics": overall,
        "breakdown": {k: v for k, v in breakdown.items()},
        "diagnostics_only": diagnostics,
        "latency_ms_per_task": overall["latency_ms_per_task"],
    }

    metrics_file = output_dir / "evaluation_results.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    log.info(f"Saved evaluation metrics to {metrics_file}")

    # 10. Console report
    print("\n" + "=" * 68)
    print("   SHORTCHAIN — TOOLBENCH UNSEEN-TOOLS BENCHMARK (api_list pool)")
    print("=" * 68)
    print("HEADLINE (over api_list candidate pools, no sampled negatives):")
    for key, value in overall.items():
        if key in ("diagnostics_only",):
            continue
        print(f"   {key:<22}: {value}")
    print("-" * 68)
    print("BY SUBSET:")
    for src, m in breakdown.items():
        line = "   " + f"{src:<14}" + " ".join(
            f"{k}={v:.4f}" for k, v in m.items() if k not in ("eval_tasks", "latency_ms_per_task", "mean_candidates_per_task") and not k.startswith("recall_at")
        )
        print(line)
        rk = {k: round(v, 4) for k, v in m.items() if k.startswith("recall_at")}
        if rk:
            print("       recall@k: " + ", ".join(f"{k.split('_')[2]}={v}" for k, v in sorted(rk.items(), key=lambda x: int(x[0].split("_")[2]))))
    print("-" * 68)
    print(f"Strictly-unseen tasks audited: {unseen['strictly_unseen_tasks']}/{unseen['valid_tasks']}")
    print(f"Latency: {overall['latency_ms_per_task']} ms/task | model: {model_path}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
