#!/usr/bin/env python3
"""Run the full TabAgent benchmark pipeline on ToolBench data.

Usage::

    python scripts/benchmark_toolbench.py --config configs/toolbench.yaml
    python scripts/benchmark_toolbench.py --config configs/toolbench_g2.yaml

This script:
1. Loads the tool catalog from ToolBench's toolenv
2. Loads and filters trajectories to the configured scenario (G1/G2/G3)
3. For G2+: applies step-level expansion and hybrid failure negatives
4. Builds (context, tool, label) dataset pairs
5. Trains the classifier
6. Evaluates with R-Precision, Recall@k, Pass Rate, and step-wise accuracy
7. Saves results to JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabagent.config import load_config
from tabagent.dataset.builder import DatasetBuilder
from tabagent.dataset.splitter import GroupStratifiedSplitter
from tabagent.evaluation.metrics import (
    compute_metrics,
    format_metrics,
    metrics_by_group,
    step_wise_accuracy,
)
from tabagent.evaluation.threshold_tuner import ThresholdTuner
from tabagent.head.trainer import Trainer
from tabagent.ingest.toolbench_catalog import ToolBenchCatalog
from tabagent.ingest.toolbench_loader import ToolBenchLoader
from tabagent.ingest.toolbench_negatives import FailureNegativeExtractor
from tabagent.utils.io import ensure_dir
from tabagent.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TabAgent benchmark on ToolBench data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toolbench.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/toolbench",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Limit training trajectories (for quick testing).",
    )
    args = parser.parse_args()

    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    log.info("[bold]═══ TabAgent × ToolBench Benchmark ═══[/bold]")
    log.info("")

    cfg = load_config(args.config)
    tb = cfg.toolbench
    log.info(f"Scenario:    {tb.scenario}")
    log.info(f"Granularity: {tb.granularity}")
    log.info(f"Step-level:  {tb.step_level}")
    log.info(f"Failure neg: {tb.use_failure_negatives} (ratio={tb.failure_negative_ratio})")
    log.info(f"Classifier:  {cfg.classifier.model_type}")
    log.info("")

    # ------------------------------------------------------------------
    # 2. Load tool catalog
    # ------------------------------------------------------------------
    log.info("[bold]Step 1:[/bold] Loading tool catalog...")

    catalog = ToolBenchCatalog.from_toolenv(tb.toolenv_dir)
    summary = catalog.summary()
    log.info(
        f"  {summary['n_apis']:,} APIs / {summary['n_tools']:,} tools / "
        f"{summary['n_categories']} categories"
    )

    # ------------------------------------------------------------------
    # 3. Load trajectories
    # ------------------------------------------------------------------
    log.info("")
    log.info(f"[bold]Step 2:[/bold] Loading {tb.scenario} trajectories...")

    # For hybrid strategy: load ALL (success + failure), then split
    loader = ToolBenchLoader(catalog=catalog, success_only=False)
    all_trajectories = loader.load_with_filter(
        tb.train_file,
        scenario=tb.scenario,
        step_level=False,  # Expand after splitting success/failure
    )

    # Split into success / failure BEFORE step expansion
    success_trajs = [t for t in all_trajectories if t.success]
    failed_trajs = [t for t in all_trajectories if not t.success]

    log.info(f"  Successful: {len(success_trajs):,}")
    log.info(f"  Failed:     {len(failed_trajs):,}")

    # Apply step-level expansion to successful trajectories only
    if tb.step_level:
        expanded = []
        for traj in success_trajs:
            expanded.extend(ToolBenchLoader.expand_to_step_trajectories(traj))
        log.info(
            f"  Step expansion: {len(success_trajs):,} → {len(expanded):,} step-trajectories"
        )
        trajectories = expanded
    else:
        trajectories = success_trajs

    if args.max_train and len(trajectories) > args.max_train:
        log.info(f"  Limiting to {args.max_train} trajectories (--max-train)")
        trajectories = trajectories[: args.max_train]

    if not trajectories:
        log.error("No trajectories loaded. Check data and config.")
        return

    log.info(f"  Training on {len(trajectories):,} trajectories")

    # ------------------------------------------------------------------
    # 4. Build dataset
    # ------------------------------------------------------------------
    log.info("")
    log.info("[bold]Step 3:[/bold] Building dataset...")

    builder = DatasetBuilder(
        config=cfg.dataset,
        features_config=cfg.features,
        negatives_config=cfg.negatives,
        tool_catalog=catalog.catalog,
        category_map=catalog.category_map,
    )
    df = builder.build(trajectories)
    log.info(f"  Dataset: {len(df):,} rows")

    # Augment with failure negatives (hybrid strategy C)
    if tb.use_failure_negatives and failed_trajs:
        log.info("")
        log.info("[bold]Step 3b:[/bold] Adding failure negatives...")
        extractor = FailureNegativeExtractor(
            catalog=catalog.catalog,
            ratio=tb.failure_negative_ratio,
        )
        df = extractor.augment_negatives(
            df,
            failed_trajectories=failed_trajs,
            corpus_stats=builder.corpus_stats,
        )
        log.info(f"  Augmented dataset: {len(df):,} rows")

    # ------------------------------------------------------------------
    # 5. Split
    # ------------------------------------------------------------------
    log.info("")
    log.info("[bold]Step 4:[/bold] Train/test split...")

    splitter = GroupStratifiedSplitter(cfg.splitter)
    train_df, test_df = splitter.train_test_split(df)
    log.info(f"  Train: {len(train_df):,} rows")
    log.info(f"  Test:  {len(test_df):,} rows")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    log.info("")
    log.info("[bold]Step 5:[/bold] Training classifier...")

    trainer = Trainer(
        classifier_config=cfg.classifier,
        splitter_config=cfg.splitter,
        eval_config=cfg.evaluation,
    )
    model = trainer.train_final(train_df)

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    log.info("")
    log.info("[bold]Step 6:[/bold] Evaluating...")

    y_test = test_df["label"].values
    y_proba = model.predict_proba(test_df)

    metrics = compute_metrics(
        y_test,
        y_proba,
        X_val=test_df,
        k_values=cfg.evaluation.k_values,
    )

    log.info("")
    log.info("[bold]═══ Results ═══[/bold]")
    log.info(format_metrics(metrics))

    # Step-wise accuracy (for G2 step-level)
    step_metrics = {}
    if tb.step_level and "step_index" in test_df.columns:
        import numpy as np
        step_metrics = step_wise_accuracy(
            y_test,
            y_proba,
            test_df["task_id"].values,
            test_df["step_index"].values.astype(int),
            k=3,
        )
        if step_metrics:
            log.info("")
            log.info("[bold]Step-wise Pass Rate@3:[/bold]")
            for step_name, acc in sorted(step_metrics.items()):
                log.info(f"  {step_name}: {acc:.4f}")

    # Per-category breakdown
    if "app_name" in test_df.columns:
        log.info("")
        log.info("[bold]Per-category breakdown:[/bold]")
        cat_metrics = metrics_by_group(
            y_test, y_proba, test_df,
            group_col="app_name",
            k_values=cfg.evaluation.k_values,
        )
        if not cat_metrics.empty:
            if "r_precision" in cat_metrics.columns:
                cat_metrics = cat_metrics.sort_values("r_precision", ascending=False)
                log.info(f"  Top 5 categories by R-Precision:")
                for _, row in cat_metrics.head(5).iterrows():
                    log.info(
                        f"    {row['app_name']:<30s} "
                        f"R-P: {row.get('r_precision', 0):.3f}  "
                        f"n={int(row['n_samples'])}"
                    )

    # Threshold sweep
    log.info("")
    log.info("[bold]Threshold Sweep:[/bold]")
    tuner = ThresholdTuner(
        thresholds=cfg.evaluation.sweep_thresholds,
        target_metric=cfg.evaluation.threshold_target_metric,
    )
    best_t, sweep_results = tuner.find_optimal(y_test, y_proba)

    for t in sorted(sweep_results):
        r = sweep_results[t]
        marker = " ◀ best" if t == best_t else ""
        log.info(
            f"  t={t:.2f}: F1={r['f1']:.4f}  "
            f"P={r['precision']:.4f}  R={r['recall']:.4f}{marker}"
        )

    # Re-compute full metrics at optimal threshold
    if best_t != 0.5:
        log.info("")
        log.info(f"[bold]Results at optimal threshold ({best_t:.2f}):[/bold]")
        metrics_opt = compute_metrics(
            y_test, y_proba, X_val=test_df,
            k_values=cfg.evaluation.k_values, threshold=best_t,
        )
        log.info(format_metrics(metrics_opt))
        # Store both in results
        metrics["optimal_threshold"] = best_t
        metrics["metrics_at_optimal"] = metrics_opt

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    output_dir = ensure_dir(args.output)
    results_file = output_dir / f"{tb.scenario.lower()}_results.json"

    results = {
        "scenario": tb.scenario,
        "step_level": tb.step_level,
        "use_failure_negatives": tb.use_failure_negatives,
        "n_success_trajectories": len(success_trajs),
        "n_failed_trajectories": len(failed_trajs),
        "n_training_trajectories": len(trajectories),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "catalog_size": summary["n_apis"],
        "metrics": metrics,
        "step_metrics": step_metrics,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info("")
    log.info(f"Results saved → {results_file}")
    log.info(f"Total time: {results['elapsed_seconds']}s")
    log.info("[bold green]✓ Benchmark complete[/bold green]")


if __name__ == "__main__":
    main()
