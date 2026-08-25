"""Unified classifier interface wrapping multiple ML backends for ShortChain.

This module provides a high-level wrapper (`ShortChainClassifier`) designed to
unify training, inference, candidate tool shortlisting, and serialization across
multiple model backends (e.g., XGBoost, Random Forest, Logistic Regression).

Architecture Overview:
---------------------
- **v1 (Legacy)**: Inline feature vectorization using individual TF-IDF vectorizers
  and LabelEncoders persisted inside the classifier instance.
- **v2 (Current)**: Standardized feature encoding delegated entirely to the
  ``FeaturePipeline`` module.
- **Backward Compatibility**: Automated fallback detection during model deserialization
  (`ShortChainClassifier.load`) allowing seamless inference on v1 legacy models.

Learning & inference contract
-----------------------------
`fit(X, y)` trains the backend AND the `FeaturePipeline` (fitted encoders) on
the supplied rows — in the harness these are TRAIN / fold-train rows only.
`predict_proba` then encodes via the fitted pipeline and returns the positive
class probability, which the inference engine sorts to rank candidate tools.
There is deliberately no test-time fitting anywhere: everything that encodes
or learns is built from the training set, so evaluation rows are projected
onto a fixed, train-derived space.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from shortchain.config import ClassifierConfig, FeaturesConfig
from shortchain.features.pipeline import FeaturePipeline
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class ShortChainClassifier:
    """Unified classifier interface for candidate tool shortlisting and prediction.

    Wraps lower-level scikit-learn and XGBoost estimators behind a consistent API.
    Handles feature transformation (via ``FeaturePipeline`` or legacy inline transformers),
    train-validation splitting for early stopping, model persistence, and K-best ranking.

    Attributes
    ----------
    config : ClassifierConfig
        Active configuration defining the estimator type and hyper-parameters.
    features_config : FeaturesConfig | None
        Configuration governing the feature preprocessing pipeline.
    model : Any
        Instantiated underlying machine learning model (e.g., ``XGBClassifier``).
    pipeline : FeaturePipeline | None
        Active feature pipeline instance for vectorizing raw input DataFrames.
    _is_fitted : bool
        Flag indicating whether the classifier has been fitted and is ready for inference.
    _use_legacy : bool
        Internal flag indicating whether inference should use legacy inline transformers.

    Parameters
    ----------
    config : ClassifierConfig | None, optional
        Classifier configuration. Defaults to a fresh ``ClassifierConfig()`` if None.
    features_config : FeaturesConfig | None, optional
        Feature pipeline configuration. When provided, a ``FeaturePipeline`` is instantiated
        during fitting.
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        features_config: FeaturesConfig | None = None,
    ) -> None:
        # Initialize primary configuration and state variables
        self.config = config or ClassifierConfig()
        self.features_config = features_config
        self.model: Any = None
        self.pipeline: FeaturePipeline | None = None
        self._is_fitted = False

        # Legacy encoding state (retained strictly for backward-compatibility loading)
        self._legacy_tfidf: dict[str, TfidfVectorizer] = {}
        self._legacy_le: dict[str, LabelEncoder] = {}
        self._legacy_skipped: set[str] = set()
        self._use_legacy = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ShortChainClassifier":
        """Fit the classifier on raw training data and binary target labels.

        Instantiates the model backend and feature pipeline, transforms the input
        DataFrame into encoded feature arrays, handles early-stopping evaluation splits
        (if XGBoost is configured), and trains the underlying estimator.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature DataFrame containing text, categorical, and numeric columns.
        y : pd.Series
            Binary ground-truth labels (1 for positive relevance, 0 for negative).

        Returns
        -------
        ShortChainClassifier
            The fitted classifier instance (self).
        """
        # Step 1: Instantiate model estimator based on active configuration
        self.model = self._create_model()

        # Step 2: Initialize and fit the v2 FeaturePipeline
        self.pipeline = FeaturePipeline(
            config=self.features_config or FeaturesConfig()
        )
        X_enc = self.pipeline.fit_transform(X)

        # Step 3: Handle early stopping split specifically for XGBoost backends
        fit_kwargs: dict = {}
        if self.config.model_type == "xgboost" and self.config.xgboost.early_stopping_rounds:
            from sklearn.model_selection import train_test_split

            # Hold out 10% of training data for evaluation during boosted tree training
            X_enc, X_val, y, y_val = train_test_split(
                X_enc, y, test_size=0.1, random_state=42,
            )
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        log.info(
            f"Training [bold]{self.config.model_type}[/bold] on "
            f"{X_enc.shape[0]} samples × {X_enc.shape[1]} features"
        )

        # Step 4: Fit model on processed feature matrix
        self.model.fit(X_enc, y, **fit_kwargs)
        self._is_fitted = True
        self._use_legacy = False
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute predicted probabilities for the positive class (label=1).

        Parameters
        ----------
        X : pd.DataFrame
            Raw input feature DataFrame.

        Returns
        -------
        np.ndarray
            1D array of floats with shape ``(n_samples,)`` representing the
            probability of belonging to class 1.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted or loaded.
        """
        self._check_fitted()
        X_enc = self._encode(X)
        proba = self.model.predict_proba(X_enc)

        # Extract positive class column (index 1) for 2D probability outputs
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate binary predictions (0 or 1) for input samples.

        Parameters
        ----------
        X : pd.DataFrame
            Raw input feature DataFrame.

        Returns
        -------
        np.ndarray
            1D array of predicted binary class labels (0 or 1).

        Raises
        ------
        RuntimeError
            If called before the model has been fitted or loaded.
        """
        self._check_fitted()
        X_enc = self._encode(X)
        return self.model.predict(X_enc)

    def shortlist(
        self,
        X: pd.DataFrame,
        top_k: int = 7,
    ) -> list[list[tuple[str, float]]]:
        """Score candidate tools across distinct task contexts and return the top-K.

        Groups candidates by `task_id`, calculates relevance scores using predicted
        probabilities, and sorts candidates per task in descending order.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing candidate pairs. Must include:
            - ``task_id``: Identifier used to group candidate tools per task context.
            - ``tool_name``: Name/identifier of candidate tools.
        top_k : int, default=7
            Maximum number of top-scoring candidates to return per task.

        Returns
        -------
        list[list[tuple[str, float]]]
            Nested list where each element corresponds to a unique task ID, containing
            a list of ``(tool_name, probability_score)`` tuples sorted by score descending.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted or loaded.
        """
        self._check_fitted()
        scores = self.predict_proba(X)
        results: list[list[tuple[str, float]]] = []

        # Iterate over unique task groupings to rank tools per context
        for task_id in X["task_id"].unique():
            mask = X["task_id"] == task_id
            task_tools = X.loc[mask, "tool_name"].values
            task_scores = scores[mask]

            # Pair tool names with predicted scores and rank descending
            ranked = sorted(
                zip(task_tools, task_scores), key=lambda x: x[1], reverse=True
            )
            results.append([(str(t), float(s)) for t, s in ranked[:top_k]])

        return results

    def save(self, path: str | Path) -> Path:
        """Persist the trained model, configurations, and pipeline state to disk.

        Saves state as a pickled dictionary marked with format version 2.

        Parameters
        ----------
        path : str | Path
            Target destination filepath for the serialized pickle object.

        Returns
        -------
        Path
            Path object pointing to the created model artifact.

        Raises
        ------
        RuntimeError
            If attempting to save an unfitted model instance.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build state payload including v2 metadata format version 2
        state = {
            "model": self.model,
            "config": self.config.model_dump(),
            "features_config": (
                self.features_config.model_dump() if self.features_config else None
            ),
            "pipeline": self.pipeline,
            "version": 2,  # v2 format marker
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ShortChainClassifier":
        """Load a persisted classifier artifact from disk.

        Automatically inspects version metadata to support both v2 (pipeline-based)
        and v1 (legacy inline vectorizer) format structures.

        Parameters
        ----------
        path : str | Path
            Filepath of the serialized pickle artifact.

        Returns
        -------
        ShortChainClassifier
            Reconstructed classifier instance ready for inference.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        config = ClassifierConfig.model_validate(state["config"])
        version = state.get("version", 1)

        if version >= 2:
            # Reconstruct v2 instance with FeaturePipeline state
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
            # Reconstruct v1 legacy instance with backward-compatibility adapter
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
        """Route input DataFrame to active encoding pipeline or legacy adapter.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature DataFrame.

        Returns
        -------
        np.ndarray
            Encoded feature matrix suitable for model inference.

        Raises
        ------
        RuntimeError
            If non-legacy mode is active but pipeline is uninitialized.
        """
        if self._use_legacy:
            return self._legacy_transform(X)
        if self.pipeline is None:
            raise RuntimeError("No FeaturePipeline available")
        return self.pipeline.transform(X)

    # ------------------------------------------------------------------
    # Legacy compatibility adapter
    # ------------------------------------------------------------------

    # Schema definition for v1 legacy feature transformation
    _LEGACY_TEXT = ["intent", "previous_tools", "last_thought", "tool_name", "tool_description"]
    _LEGACY_NUM = ["n_spans"]
    _LEGACY_CAT = ["app_name"]

    def _legacy_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw input using saved v1 inline transformers.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature DataFrame.

        Returns
        -------
        np.ndarray
            Concatenated dense feature matrix derived from legacy TF-IDF,
            LabelEncoder, and numerical features.

        Raises
        ------
        ValueError
            If no features could be generated from the input columns.
        """
        parts = []

        # Vectorize text columns via legacy TF-IDF instances
        for col in self._LEGACY_TEXT:
            if col in self._legacy_tfidf:
                vec = self._legacy_tfidf[col]
                encoded = vec.transform(X[col].fillna("").astype(str))
                parts.append(encoded.toarray())

        # Encode categorical columns using saved LabelEncoder instances
        for col in self._LEGACY_CAT:
            if col in self._legacy_le:
                le = self._legacy_le[col]
                vals = X[col].fillna("__unknown__").astype(str)
                encoded = np.array(
                    [le.transform([v])[0] if v in le.classes_ else -1 for v in vals],
                    dtype=np.float32,
                ).reshape(-1, 1)
                parts.append(encoded)

        # Extract numerical features directly
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
        """Instantiate underlying estimator according to configuration specifications.

        Returns
        -------
        Any
            Configured scikit-learn or XGBoost estimator instance.

        Raises
        ------
        ValueError
            If the configured model type is not supported.
        """
        model_type = self.config.model_type

        if model_type == "xgboost":
            from xgboost import XGBClassifier

            params = self.config.xgboost.model_dump()
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
        """Verify that the classifier has been fitted or loaded prior to inference.

        Raises
        ------
        RuntimeError
            If ``_is_fitted`` is False.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been fitted yet. Call .fit() first.")