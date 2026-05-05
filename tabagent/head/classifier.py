"""Unified classifier interface wrapping multiple backends.

Phase 1 backends: XGBoost (default), RandomForest, LogisticRegression.

Phase 2: Feature encoding is delegated to ``FeaturePipeline``.
Legacy models (pickled without a pipeline) are supported via a
compatibility adapter that falls back to inline encoding.
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

from tabagent.config import ClassifierConfig, FeaturesConfig
from tabagent.features.pipeline import FeaturePipeline
from tabagent.utils.logging import get_logger

log = get_logger(__name__)


class TabAgentClassifier:
    """Unified classifier for tool shortlisting.

    Parameters
    ----------
    config
        Classifier configuration (model type + hyper-parameters).
    features_config
        Feature pipeline configuration.  If provided, a
        ``FeaturePipeline`` is used for encoding.
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        features_config: FeaturesConfig | None = None,
    ) -> None:
        self.config = config or ClassifierConfig()
        self.features_config = features_config
        self.model: Any = None
        self.pipeline: FeaturePipeline | None = None
        self._is_fitted = False

        # Legacy encoding state (used only for backward-compat loading)
        self._legacy_tfidf: dict[str, TfidfVectorizer] = {}
        self._legacy_le: dict[str, LabelEncoder] = {}
        self._legacy_skipped: set[str] = set()
        self._use_legacy = False

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

        # Use FeaturePipeline for encoding
        self.pipeline = FeaturePipeline(
            config=self.features_config or FeaturesConfig()
        )
        X_enc = self.pipeline.fit_transform(X)

        log.info(
            f"Training [bold]{self.config.model_type}[/bold] on "
            f"{X_enc.shape[0]} samples × {X_enc.shape[1]} features"
        )
        self.model.fit(X_enc, y)
        self._is_fitted = True
        self._use_legacy = False
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
        X_enc = self._encode(X)
        proba = self.model.predict_proba(X_enc)
        # Some models return (n, 2), take positive-class column
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions."""
        self._check_fitted()
        X_enc = self._encode(X)
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
        """Persist the trained model and feature pipeline."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "config": self.config.model_dump(),
            "features_config": (
                self.features_config.model_dump() if self.features_config else None
            ),
            "pipeline": self.pipeline,
            "version": 2,  # Phase 2 format marker
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TabAgentClassifier":
        """Load a trained classifier from disk.

        Supports both Phase 1 (legacy) and Phase 2 model formats.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        config = ClassifierConfig.model_validate(state["config"])
        version = state.get("version", 1)

        if version >= 2:
            # Phase 2 format
            features_config = (
                FeaturesConfig.model_validate(state["features_config"])
                if state.get("features_config")
                else None
            )
            obj = cls(config=config, features_config=features_config)
            obj.model = state["model"]
            obj.pipeline = state["pipeline"]
            obj._is_fitted = True
            obj._use_legacy = False
            log.info(f"Model loaded from {path} (v2 format)")
        else:
            # Phase 1 legacy format — use compatibility adapter
            obj = cls(config=config)
            obj.model = state["model"]
            obj._legacy_tfidf = state.get("tfidf_vectorizers", {})
            obj._legacy_le = state.get("label_encoders", {})
            obj._legacy_skipped = state.get("skipped_text_cols", set())
            obj._is_fitted = True
            obj._use_legacy = True
            log.info(f"Model loaded from {path} (v1 legacy format)")

        return obj

    # ------------------------------------------------------------------
    # Encoding (dispatch)
    # ------------------------------------------------------------------

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        """Encode features using either the pipeline or legacy path."""
        if self._use_legacy:
            return self._legacy_transform(X)
        if self.pipeline is None:
            raise RuntimeError("No FeaturePipeline available")
        return self.pipeline.transform(X)

    # ------------------------------------------------------------------
    # Legacy compatibility adapter
    # ------------------------------------------------------------------

    # Text columns used in Phase 1
    _LEGACY_TEXT = ["intent", "previous_tools", "last_thought", "tool_name", "tool_description"]
    _LEGACY_NUM = ["n_steps"]
    _LEGACY_CAT = ["app_name"]

    def _legacy_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using Phase 1 inline encoders (backward compat)."""
        parts = []

        for col in self._LEGACY_TEXT:
            if col in self._legacy_tfidf:
                vec = self._legacy_tfidf[col]
                encoded = vec.transform(X[col].fillna("").astype(str))
                parts.append(encoded.toarray())

        for col in self._LEGACY_CAT:
            if col in self._legacy_le:
                le = self._legacy_le[col]
                vals = X[col].fillna("__unknown__").astype(str)
                encoded = np.array(
                    [le.transform([v])[0] if v in le.classes_ else -1 for v in vals],
                    dtype=np.float32,
                ).reshape(-1, 1)
                parts.append(encoded)

        for col in self._LEGACY_NUM:
            if col in X.columns:
                vals = X[col].fillna(0).values.reshape(-1, 1).astype(np.float32)
                parts.append(vals)

        if not parts:
            raise ValueError("No features to encode")
        return np.hstack(parts)

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
