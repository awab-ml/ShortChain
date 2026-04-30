"""Unified classifier interface wrapping multiple backends.

Phase 1 backends: XGBoost (default), RandomForest, LogisticRegression.
Text features are encoded via TF-IDF (or optionally sentence-transformer
embeddings).  The classifier operates on the pointwise-reduced dataset
where each row is a ``(context, candidate_tool)`` pair with a binary
label.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse, hstack as sp_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from tabagent.config import ClassifierConfig
from tabagent.utils.logging import get_logger

log = get_logger(__name__)

# Text columns that get TF-IDF encoded
_TEXT_COLS = ["intent", "previous_tools", "last_thought", "tool_name", "tool_description"]
# Numeric columns passed through directly
_NUM_COLS = ["n_steps"]
# Categorical columns that get label-encoded
_CAT_COLS = ["app_name"]


class TabAgentClassifier:
    """Unified classifier for tool shortlisting.

    Parameters
    ----------
    config
        Classifier configuration (model type + hyper-parameters).
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.model: Any = None
        self.tfidf_vectorizers: dict[str, TfidfVectorizer] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}
        self._skipped_text_cols: set[str] = set()
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabAgentClassifier":
        """Fit the classifier on training data.

        Parameters
        ----------
        X
            Feature DataFrame (raw — encoding is handled internally).
        y
            Binary labels (1 = positive, 0 = negative).

        Returns
        -------
        self
        """
        self.model = self._create_model()
        X_enc = self._fit_transform(X)

        log.info(
            f"Training [bold]{self.config.model_type}[/bold] on "
            f"{X_enc.shape[0]} samples × {X_enc.shape[1]} features"
        )
        self.model.fit(X_enc, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities for the positive class.

        Parameters
        ----------
        X
            Feature DataFrame (raw).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` — probability of ``label=1``.
        """
        self._check_fitted()
        X_enc = self._transform(X)
        proba = self.model.predict_proba(X_enc)
        # Some models return (n, 2), take positive-class column
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions."""
        self._check_fitted()
        X_enc = self._transform(X)
        return self.model.predict(X_enc)

    def shortlist(
        self,
        X: pd.DataFrame,
        top_k: int = 7,
    ) -> list[list[tuple[str, float]]]:
        """Score candidates and return top-K per task.

        Parameters
        ----------
        X
            DataFrame with one row per ``(context, candidate_tool)`` pair.
            Must contain a ``task_id`` column to group by task.
        top_k
            Number of tools to return per task.

        Returns
        -------
        list[list[tuple[str, float]]]
            For each unique task, a list of ``(tool_name, score)``
            tuples sorted by descending score.
        """
        self._check_fitted()
        scores = self.predict_proba(X)
        results: list[list[tuple[str, float]]] = []

        for task_id in X["task_id"].unique():
            mask = X["task_id"] == task_id
            task_tools = X.loc[mask, "tool_name"].values
            task_scores = scores[mask]
            ranked = sorted(
                zip(task_tools, task_scores), key=lambda x: x[1], reverse=True
            )
            results.append([(str(t), float(s)) for t, s in ranked[:top_k]])

        return results

    def save(self, path: str | Path) -> Path:
        """Persist the trained model and feature encoders."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "config": self.config.model_dump(),
            "tfidf_vectorizers": self.tfidf_vectorizers,
            "label_encoders": self.label_encoders,
            "skipped_text_cols": self._skipped_text_cols,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TabAgentClassifier":
        """Load a trained classifier from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        config = ClassifierConfig.model_validate(state["config"])
        obj = cls(config=config)
        obj.model = state["model"]
        obj.tfidf_vectorizers = state["tfidf_vectorizers"]
        obj.label_encoders = state["label_encoders"]
        obj._skipped_text_cols = state.get("skipped_text_cols", set())
        obj._is_fitted = True
        log.info(f"Model loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    def _fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit encoders on training data and return encoded features."""
        parts = []

        # TF-IDF for text columns
        for col in _TEXT_COLS:
            if col in X.columns:
                vec = TfidfVectorizer(
                    max_features=5000,
                    sublinear_tf=True,
                    dtype=np.float32,
                )
                try:
                    encoded = vec.fit_transform(X[col].fillna("").astype(str))
                    self.tfidf_vectorizers[col] = vec
                    parts.append(encoded)
                except ValueError:
                    # Column contains only empty strings or stop words — skip it
                    log.debug(f"Skipping TF-IDF for column '{col}' (empty vocabulary)")
                    self._skipped_text_cols.add(col)

        # Label encoding for categoricals
        for col in _CAT_COLS:
            if col in X.columns:
                le = LabelEncoder()
                vals = X[col].fillna("__unknown__").astype(str)
                encoded = le.fit_transform(vals).reshape(-1, 1).astype(np.float32)
                self.label_encoders[col] = le
                parts.append(encoded)

        # Numeric columns
        for col in _NUM_COLS:
            if col in X.columns:
                vals = X[col].fillna(0).values.reshape(-1, 1).astype(np.float32)
                parts.append(vals)

        return self._concat_features(parts)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted encoders."""
        parts = []

        for col in _TEXT_COLS:
            if col in self.tfidf_vectorizers:
                vec = self.tfidf_vectorizers[col]
                encoded = vec.transform(X[col].fillna("").astype(str))
                parts.append(encoded)

        for col in _CAT_COLS:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                vals = X[col].fillna("__unknown__").astype(str)
                # Handle unseen labels gracefully
                encoded = np.array(
                    [le.transform([v])[0] if v in le.classes_ else -1 for v in vals],
                    dtype=np.float32,
                ).reshape(-1, 1)
                parts.append(encoded)

        for col in _NUM_COLS:
            if col in X.columns:
                vals = X[col].fillna(0).values.reshape(-1, 1).astype(np.float32)
                parts.append(vals)

        return self._concat_features(parts)

    @staticmethod
    def _concat_features(parts: list) -> np.ndarray:
        """Concatenate sparse and dense feature matrices."""
        if not parts:
            raise ValueError("No features to concatenate — check column names")

        # Separate sparse and dense
        sparse_parts = [p for p in parts if issparse(p)]
        dense_parts = [p for p in parts if not issparse(p)]

        if sparse_parts:
            sparse_combined = sp_hstack(sparse_parts)
            if dense_parts:
                dense_combined = np.hstack(dense_parts)
                # Convert sparse to dense for final concatenation
                return np.hstack([sparse_combined.toarray(), dense_combined])
            return sparse_combined.toarray()
        return np.hstack(dense_parts)

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _create_model(self) -> Any:
        """Instantiate the underlying sklearn-compatible model."""
        model_type = self.config.model_type

        if model_type == "xgboost":
            from xgboost import XGBClassifier

            params = self.config.xgboost.model_dump()
            early_stopping = params.pop("early_stopping_rounds", None)
            return XGBClassifier(
                **params,
                use_label_encoder=False,
                verbosity=0,
            )

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                **self.config.random_forest.model_dump(),
                n_jobs=-1,
                random_state=42,
            )

        elif model_type == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                **self.config.logistic.model_dump(),
                random_state=42,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type!r}")

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been fitted yet. Call .fit() first.")
