#!/usr/bin/env python3
"""Generic benchmark runner.

Resolves a benchmark adapter by name and runs the full TabAgent pipeline:
ingest → build dataset → train (with CV) → evaluate.

Usage::

    python scripts/run_benchmark.py \\
        --benchmark toolbench \\
        --train-path data/toolbench/train.jsonl \\
        --eval-path data/toolbench/test.jsonl \\
        --config configs/default.yaml

    python scripts/run_benchmark.py \\
        --benchmark toolbench \\
        --train-path data/toolbench/train.jsonl \\
        --eval-path data/toolbench/test.jsonl \\
        --step-level \\
        --max-train 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tabagent.benchmarks import create_adapter
from tabagent.config import load_config
from tabagent.dataset.builder import DatasetBuilder
from tabagent.evaluation.metrics import compute_metrics, format_metrics
from tabagent.head.trainer import Trainer
from tabagent.utils.io import ensure_dir, write_json
from tabagent.utils.logging import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a benchmark using the TabAgent adapter architecture.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="toolbench",
        help="Benchmark adapter name (default: toolbench).",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to training trajectory file or directory.",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        required=True,
        help="Path to evaluation trajectory file or directory.",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default=None,
        help="Path to tool catalog JSON file (optional).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (overrides defaults).",
    )
    parser.add_argument(
        "--step-level",
        action="store_true",
        help="Enable step-level trajectory expansion.",
    )
    parser.add_argument(
        "--failure-negatives",
        action="store_true",
        help="Enable failure-negative augmentation.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Limit training trajectories (for quick experiments).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/tabagent.pkl",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for evaluation results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.time()

    # 1. Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.step_level:
        cfg.benchmark.step_level = True
    if args.failure_negatives:
        cfg.benchmark.use_failure_negatives = True

    log.info(
        f"[bold]Benchmark:[/bold] {args.benchmark}  |  "
        f"step_level={cfg.benchmark.step_level}  |  "
        f"failure_negs={cfg.benchmark.use_failure_negatives}"
    )

    # 2. Resolve adapter
    adapter = create_adapter(
        args.benchmark,
        cfg,
        train_path=args.train_path,
        eval_path=args.eval_path,
        catalog_path=args.catalog_path,
    )

    # 3. Load data via adapter
    log.info("[bold]Step 1:[/bold] Loading data via adapter")
    catalog = adapter.load_catalog()
    train_trajs = adapter.load_trajectories("train")
    test_trajs = adapter.load_trajectories("test")

    # Optional: limit training data for quick experiments
    if args.max_train and len(train_trajs) > args.max_train:
        log.info(
            f"  Limiting training trajectories: "
            f"{len(train_trajs)} → {args.max_train}"
        )
        train_trajs = train_trajs[: args.max_train]

    # Re-derive catalog if it was empty (built from trajectories)
    if not catalog:
        catalog = adapter.load_catalog()

    log.info(
        f"  Catalog: [bold]{len(catalog)}[/bold] tools  |  "
        f"Train: [bold]{len(train_trajs)}[/bold]  |  "
        f"Test: [bold]{len(test_trajs)}[/bold]"
    )

    # 4. Build dataset (core pipeline — fully generic)
    log.info("[bold]Step 2:[/bold] Building training dataset")
    builder = DatasetBuilder(
        config=cfg.dataset,
        features_config=cfg.features,
        negatives_config=cfg.negatives,
        tool_catalog=catalog,
    )
    train_df = builder.build(train_trajs)

    # 5. Adapter-specific augmentation
    train_df = adapter.augment_training(train_df)

    # 6. Train with cross-validation
    log.info("[bold]Step 3:[/bold] Training with cross-validation")
    trainer = Trainer(
        classifier_config=cfg.classifier,
        splitter_config=cfg.splitter,
        eval_config=cfg.evaluation,
    )
    cv_results = trainer.train_with_cv(train_df)

    # 7. Train final model
    log.info("[bold]Step 4:[/bold] Training final model")
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    clf = trainer.train_final(train_df, save_path=output_path)
    log.info(f"  Model saved to {output_path}")

    # 8. Evaluate on test set
    log.info("[bold]Step 5:[/bold] Evaluating on test set")
    test_df = builder.build(test_trajs)
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"].values
    y_proba = clf.predict_proba(X_test)

    test_metrics = compute_metrics(
        y_true=y_test,
        y_proba=y_proba,
        X_val=X_test,
        k_values=cfg.evaluation.k_values,
    )

    log.info("[bold]Test Results:[/bold]")
    print(format_metrics(test_metrics))

    # 9. Save results
    results_dir = ensure_dir(args.results_dir)
    results = {
        "benchmark": args.benchmark,
        "config": {
            "step_level": cfg.benchmark.step_level,
            "use_failure_negatives": cfg.benchmark.use_failure_negatives,
        },
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "total_time_s": round(time.time() - total_start, 2),
    }
    results_path = results_dir / f"{args.benchmark}_results.json"
    write_json(results, results_path)
    log.info(f"  Results saved to {results_path}")

    total_time = time.time() - total_start
    log.info(
        f"[bold green]✓ Benchmark complete[/bold green] "
        f"({total_time:.1f}s total)"
    )


if __name__ == "__main__":
    main()
