#!/usr/bin/env python3
"""Evaluate a trained ShortChain model on a test set.

Usage::

    python scripts/evaluate.py \\
        --model models/shortchain.pkl \\
        --dataset data/datasets/test.csv

Concept
-------
Reports head-matched ranking metrics on the held-out test rows: R-Precision
(P@R, adaptive to the relevant-set size) and Recall@k (fixed budget). This
script evaluates an already-built dataset; for the full leak-free benchmark
(feature construction from traces, baselines, bootstrap CIs) use
``scripts/run_validation.py`` with ``--level task|span``.
"""

from __future__ import annotations

import argparse

import pandas as pd

from shortchain.config import load_config
from shortchain.evaluation.metrics import compute_metrics, format_metrics
from shortchain.head.classifier import ShortChainClassifier
from shortchain.utils.io import write_json
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ShortChain model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.pkl).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the test dataset (CSV) or directory containing test.csv.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Load model
    log.info(f"Loading model from {args.model}")
    clf = ShortChainClassifier.load(args.model)

    # Load test data
    from pathlib import Path
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        test_path = dataset_path / "test.csv"
    else:
        test_path = dataset_path

    if not test_path.exists():
        log.error(f"Test data not found: {test_path}")
        return

    log.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    # Run evaluation
    label_col = "label"
    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col].values
    y_proba = clf.predict_proba(X_test)

    metrics = compute_metrics(
        y_true=y_test,
        y_proba=y_proba,
        X_val=X_test,
        k_values=cfg.evaluation.k_values,
    )

    # Display results
    log.info("[bold]Evaluation Results:[/bold]")
    print(format_metrics(metrics))

    # Optionally save
    if args.output:
        write_json(metrics, args.output)
        log.info(f"Results saved to {args.output}")

    log.info("[bold green]✓ Evaluation complete[/bold green]")


if __name__ == "__main__":
    main()
