#!/usr/bin/env python3
"""Run the full TabAgent benchmark pipeline on ToolBench data.

Usage::

    python scripts/benchmark_toolbench.py --config configs/toolbench.yaml

This script:
1. Loads the tool catalog from ToolBench's toolenv
2. Loads and filters trajectories to the configured scenario (G1/G2/G3)
3. Builds (context, tool, label) dataset pairs
4. Trains the classifier
5. Evaluates with R-Precision, Recall@k, and Pass Rate
6. Saves results to JSON
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
from tabagent.evaluation.metrics import compute_metrics, format_metrics, metrics_by_group
from tabagent.head.trainer import Trainer
from tabagent.ingest.toolbench_catalog import ToolBenchCatalog
from tabagent.ingest.toolbench_loader import ToolBenchLoader
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

    loader = ToolBenchLoader(catalog=catalog, success_only=cfg.ingest.success_only)
    trajectories = loader.load_with_filter(tb.train_file, scenario=tb.scenario)

    if args.max_train and len(trajectories) > args.max_train:
        log.info(f"  Limiting to {args.max_train} trajectories (--max-train)")
        trajectories = trajectories[: args.max_train]

    if not trajectories:
        log.error("No trajectories loaded. Check data and config.")
        return

    log.info(f"  Loaded {len(trajectories):,} trajectories")

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
    )
    df = builder.build(trajectories)
    log.info(f"  Dataset: {len(df):,} rows")

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
            # Show top and bottom categories by R-precision
            if "r_precision" in cat_metrics.columns:
                cat_metrics = cat_metrics.sort_values("r_precision", ascending=False)
                log.info(f"  Top 5 categories by R-Precision:")
                for _, row in cat_metrics.head(5).iterrows():
                    log.info(
                        f"    {row['app_name']:<30s} "
                        f"R-P: {row.get('r_precision', 0):.3f}  "
                        f"n={int(row['n_samples'])}"
                    )

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    output_dir = ensure_dir(args.output)
    results_file = output_dir / f"{tb.scenario.lower()}_results.json"

    results = {
        "scenario": tb.scenario,
        "n_trajectories": len(trajectories),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "catalog_size": summary["n_apis"],
        "metrics": metrics,
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
