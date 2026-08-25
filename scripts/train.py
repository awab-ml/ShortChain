#!/usr/bin/env python3
"""Train a ShortChain classifier.

Usage::

    python scripts/train.py \\
        --dataset /tmp/sc-ds \\
        --model xgboost \\
        --folds 5 \\
        --output /tmp/sc-model.pkl

Concept
-------
Runs group-aware cross-validation (folds keep whole tasks together, so the
same task never appears in both train and validation) and then trains the
final model on all of the training data. Classification-over-candidates is
the point: at inference the model scores each candidate tool and we rank by
probability, so the *ranking* quality is what matters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shortchain.config import load_config
from shortchain.model.trainer import Trainer
from shortchain.utils.io import ensure_dir, write_json
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ShortChain classifier.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory (expects train.csv) or a single CSV file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model type: xgboost, random_forest, logistic (overrides config).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Number of CV folds (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/shortchain.pkl",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.model:
        cfg.classifier.model_type = args.model
    if args.folds:
        cfg.splitter.n_folds = args.folds

    # Load training data
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        train_path = dataset_path / "train.csv"
    else:
        train_path = dataset_path

    if not train_path.exists():
        log.error(f"Training data not found: {train_path}")
        return

    log.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    log.info(f"Training data: {len(train_df)} rows, {train_df.columns.tolist()}")

    # Train with cross-validation
    trainer = Trainer(
        classifier_config=cfg.classifier,
        splitter_config=cfg.splitter,
        eval_config=cfg.evaluation,
    )

    log.info("[bold]Running cross-validation...[/bold]")
    cv_results = trainer.train_with_cv(train_df)

    # Train final model
    log.info("[bold]Training final model on all data...[/bold]")
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    trainer.train_final(train_df, save_path=output_path)

    # Save CV results
    results_path = output_path.parent / "cv_results.json"
    write_json(cv_results, results_path)
    log.info(f"CV results saved to {results_path}")

    log.info("[bold green]✓ Training complete[/bold green]")


if __name__ == "__main__":
    main()
