"""Training pipeline with cross-validation.

Orchestrates the full training loop: split → train → evaluate per fold,
then retrain on all training data and save the final model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabagent.config import ClassifierConfig, SplitterConfig, EvaluationConfig
from tabagent.dataset.splitter import GroupStratifiedSplitter
from tabagent.evaluation.metrics import compute_metrics
from tabagent.head.classifier import TabAgentClassifier
from tabagent.utils.logging import get_logger

log = get_logger(__name__)


class Trainer:
    """Train a TabAgent classifier with cross-validation.

    Parameters
    ----------
    classifier_config
        Classifier hyper-parameters.
    splitter_config
        Cross-validation split settings.
    eval_config
        Evaluation metric settings.
    """

    def __init__(
        self,
        classifier_config: ClassifierConfig | None = None,
        splitter_config: SplitterConfig | None = None,
        eval_config: EvaluationConfig | None = None,
    ) -> None:
        self.classifier_config = classifier_config or ClassifierConfig()
        self.splitter_config = splitter_config or SplitterConfig()
        self.eval_config = eval_config or EvaluationConfig()

    def train_with_cv(
        self,
        train_df: pd.DataFrame,
        label_col: str = "label",
    ) -> dict[str, Any]:
        """Run k-fold cross-validation and return aggregate metrics.

        Parameters
        ----------
        train_df
            Training DataFrame with features and labels.
        label_col
            Name of the label column.

        Returns
        -------
        dict
            ``{"fold_metrics": [...], "aggregate": {...}, "training_time_s": float}``
        """
        splitter = GroupStratifiedSplitter(self.splitter_config)
        fold_metrics: list[dict[str, float]] = []
        total_start = time.time()

        for fold_idx, (fold_train, fold_val) in enumerate(
            splitter.kfold_split(train_df)
        ):
            clf = TabAgentClassifier(self.classifier_config)
            X_train = fold_train.drop(columns=[label_col])
            y_train = fold_train[label_col]
            X_val = fold_val.drop(columns=[label_col])
            y_val = fold_val[label_col]

            fold_start = time.time()
            clf.fit(X_train, y_train)
            fold_time = time.time() - fold_start

            # Evaluate
            y_proba = clf.predict_proba(X_val)
            metrics = compute_metrics(
                y_true=y_val.values,
                y_proba=y_proba,
                X_val=X_val,
                k_values=self.eval_config.k_values,
            )
            metrics["fold"] = fold_idx + 1
            metrics["train_time_s"] = round(fold_time, 2)
            fold_metrics.append(metrics)

            log.info(
                f"  Fold {fold_idx + 1}: "
                f"P@R={metrics.get('r_precision', 0):.3f}  "
                f"R@5={metrics.get('recall_at_5', 0):.3f}  "
                f"F1={metrics.get('f1', 0):.3f}  "
                f"({fold_time:.1f}s)"
            )

        total_time = time.time() - total_start

        # Aggregate across folds
        metric_keys = [k for k in fold_metrics[0] if k not in ("fold", "train_time_s")]
        aggregate = {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in metric_keys
        }

        log.info(
            f"[bold green]CV Results ({self.splitter_config.n_folds} folds):[/bold green] "
            f"P@R={aggregate.get('r_precision', 0):.3f}  "
            f"R@5={aggregate.get('recall_at_5', 0):.3f}  "
            f"F1={aggregate.get('f1', 0):.3f}"
        )

        return {
            "fold_metrics": fold_metrics,
            "aggregate": aggregate,
            "training_time_s": round(total_time, 2),
        }

    def train_final(
        self,
        train_df: pd.DataFrame,
        label_col: str = "label",
        save_path: str | Path | None = None,
    ) -> TabAgentClassifier:
        """Train the final model on all training data.

        Parameters
        ----------
        train_df
            Full training DataFrame.
        label_col
            Name of the label column.
        save_path
            If provided, persist the trained model to this path.

        Returns
        -------
        TabAgentClassifier
            The trained classifier.
        """
        clf = TabAgentClassifier(self.classifier_config)
        X = train_df.drop(columns=[label_col])
        y = train_df[label_col]

        start = time.time()
        clf.fit(X, y)
        elapsed = time.time() - start

        log.info(
            f"Final model trained on {len(X)} samples in {elapsed:.1f}s"
        )

        if save_path:
            clf.save(save_path)

        return clf
