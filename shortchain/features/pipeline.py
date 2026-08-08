"""Feature pipeline orchestrator for ShortChain.

This module provides the ``FeaturePipeline`` class, which acts as the primary orchestrator
for feature extraction, transformation, and encoding. It converts heterogeneous raw input
data (text fields, numerical features, boolean flags, and categorical variables) into a
dense numeric NumPy matrix ready for consumption by machine learning estimators.

Key Features:
-------------
- **Input Flexibility**: Accepts both ``pd.DataFrame`` instances and list-of-dictionary records.
- **Multi-Modal Encoding**:
  - Text: Vectorized dynamically using TF-IDF or transformer-based embeddings via ``create_encoder``.
  - Categoricals: Encoded via scikit-learn's ``LabelEncoder`` with unknown handling.
  - Numerics: Parsed, cleaned, imputed with 0, and passed through as float32 matrices.
  - Booleans: Formatted directly into 0.0/1.0 float32 arrays.
- **Serialization**: Full pickle-based save and load support for production persistence.

Leak-free encoding
------------------
Encoding is fitted ONLY on the training sample (``fit_transform`` in
``ShortChainClassifier.fit``); evaluation rows are passed through
``transform`` with the *fitted* encoders. Unknown categorical levels map to
``-1`` and text is projected onto the training vocabulary, so unseen test
values cannot teach the model anything new at inference time.
Layout note: ``_TEXT_COLS``/``_NUM_COLS``/``_BOOL_COLS``/``_CAT_COLS`` list
YOUR candidate schema. Columns present in a DataFrame are encoded; columns
absent are silently skipped — which keeps the pipeline backward compatible as
feature groups are added or removed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from shortchain.config import FeaturesConfig
from shortchain.features.encoders import create_encoder
from shortchain.utils.logging import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------
# Schema Column Definitions
# ------------------------------------------------------------------

# Text columns processed via text encoding strategy (e.g., TF-IDF or embeddings)
_TEXT_COLS = [
    "intent",
    "previous_tools",
    "last_thought",
    "tool_name",
    "tool_description",
    "history_summary",
    "last_observation",
]

# Numeric columns passed through directly after cleaning and missing-value imputation
_NUM_COLS = [
    "n_spans",
    "span_index",
    "unique_tools_so_far",
    "tool_diversity",
    "app_tool_count",
    "tool_name_length",
    "tool_frequency",
    "tool_co_occurrence",
    # Static per-tool schema features (P2); absent columns are simply skipped.
    "n_params",
    "n_string_params",
    "n_integer_params",
    "n_number_params",
    "n_boolean_params",
    "n_array_params",
    "n_enum_params",
]

# Boolean flags converted to binary float representations (0.0 / 1.0)
_BOOL_COLS = [
    "has_description",
    "tool_app_match",
    "has_parameters",
]

# Categorical variables encoded using integer label encoding
_CAT_COLS = ["app_name"]


class FeaturePipeline:
    """Orchestrates multi-modal feature extraction, transformation, and vectorization.

    Handles text vectorization, categorical label encoding, numeric parsing, and boolean
    casting across raw DataFrames or list-of-dict representations.

    Attributes
    ----------
    config : FeaturesConfig
        Pipeline configuration governing encoder selection and hyperparameters.
    _encoders : dict[str, Any]
        Fitted text encoder instances keyed by column name.
    _label_encoders : dict[str, LabelEncoder]
        Fitted LabelEncoder instances keyed by categorical column name.
    _skipped_text_cols : set[str]
        Set of text columns skipped during fitting due to empty vocabularies or errors.
    _is_fitted : bool
        Flag indicating whether the pipeline has been fitted and is ready for transform.

    Parameters
    ----------
    config : FeaturesConfig | None, optional
        Feature pipeline configuration object. Defaults to a fresh ``FeaturesConfig()`` if None.
    """

    def __init__(self, config: FeaturesConfig | None = None) -> None:
        self.config = config or FeaturesConfig()
        self._encoders: dict[str, Any] = {}
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._skipped_text_cols: set[str] = set()
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, data: pd.DataFrame | list[dict]) -> np.ndarray:
        """Fit feature encoders on raw input data and return the encoded feature matrix.

        Iterates through schema-defined text, categorical, numerical, and boolean columns,
        fits corresponding encoders, and horizontally stacks transformed outputs into a
        single dense 2D matrix.

        Parameters
        ----------
        data : pd.DataFrame | list[dict]
            Raw input dataset containing feature columns.

        Returns
        -------
        np.ndarray
            A 2D dense float32 array of shape ``(n_samples, n_features)``.

        Raises
        ------
        ValueError
            If no valid columns from the feature schema are present in the input.
        """
        df = self._to_dataframe(data)
        parts: list[np.ndarray] = []

        # 1. Process Text Columns: Instantiate, fit, and transform text encoders
        for col in _TEXT_COLS:
            if col not in df.columns:
                continue
            enc = create_encoder(
                name=self.config.text_encoder,
                max_features=self.config.tfidf_max_features,
                model_name=self.config.e5_model_name,
            )
            texts = df[col].fillna("").astype(str).tolist()
            try:
                encoded = enc.fit_transform(texts)
                self._encoders[col] = enc
                parts.append(encoded)
            except ValueError:
                log.debug(f"Skipping encoder for column '{col}' (empty vocabulary)")
                self._skipped_text_cols.add(col)

        # 2. Process Categorical Columns: Fit LabelEncoder and reshape to column vector
        for col in _CAT_COLS:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            vals = df[col].fillna("__unknown__").astype(str)
            encoded = le.fit_transform(vals).reshape(-1, 1).astype(np.float32)
            self._label_encoders[col] = le
            parts.append(encoded)

        # 3. Process Numeric Columns: Cast to numeric, handle NaNs, reshape to column vector
        for col in _NUM_COLS:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            parts.append(vals.values.reshape(-1, 1).astype(np.float32))

        # 4. Process Boolean Columns: Convert boolean flags to binary float32 values
        for col in _BOOL_COLS:
            if col not in df.columns:
                continue
            vals = df[col].astype(int).values.reshape(-1, 1).astype(np.float32)
            parts.append(vals)

        # Verify that at least one feature column was successfully encoded
        if not parts:
            raise ValueError("No features to encode — check column names")

        self._is_fitted = True
        return np.hstack(parts)

    def transform(self, data: pd.DataFrame | list[dict]) -> np.ndarray:
        """Transform raw input data using previously fitted encoders.

        Parameters
        ----------
        data : pd.DataFrame | list[dict]
            Raw input dataset containing feature columns.

        Returns
        -------
        np.ndarray
            A 2D dense float32 array of shape ``(n_samples, n_features)``.

        Raises
        ------
        RuntimeError
            If called before the pipeline has been fitted.
        ValueError
            If no valid features could be constructed from input columns.
        """
        self._check_fitted()
        df = self._to_dataframe(data)
        parts: list[np.ndarray] = []

        # 1. Transform Text Columns using stored encoder state
        for col in _TEXT_COLS:
            if col not in self._encoders:
                continue
            enc = self._encoders[col]
            texts = df[col].fillna("").astype(str).tolist()
            parts.append(enc.transform(texts))

        # 2. Transform Categorical Columns handling unknown levels cleanly (-1)
        for col in _CAT_COLS:
            if col not in self._label_encoders:
                continue
            le = self._label_encoders[col]
            vals = df[col].fillna("__unknown__").astype(str)
            encoded = np.array(
                [le.transform([v])[0] if v in le.classes_ else -1 for v in vals],
                dtype=np.float32,
            ).reshape(-1, 1)
            parts.append(encoded)

        # 3. Transform Numeric Columns
        for col in _NUM_COLS:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            parts.append(vals.values.reshape(-1, 1).astype(np.float32))

        # 4. Transform Boolean Columns
        for col in _BOOL_COLS:
            if col not in df.columns:
                continue
            vals = df[col].astype(int).values.reshape(-1, 1).astype(np.float32)
            parts.append(vals)

        if not parts:
            raise ValueError("No features to encode — check column names")

        return np.hstack(parts)

    def save(self, path: str | Path) -> None:
        """Serialize and save the fitted pipeline state to disk via pickle.

        Parameters
        ----------
        path : str | Path
            Destination filepath for the saved artifact.

        Raises
        ------
        RuntimeError
            If attempting to save an unfitted pipeline instance.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config.model_dump(),
            "encoders": self._encoders,
            "label_encoders": self._label_encoders,
            "skipped_text_cols": self._skipped_text_cols,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info(f"FeaturePipeline saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Load a persisted FeaturePipeline instance from a pickle file.

        Parameters
        ----------
        path : str | Path
            Filepath of the saved pipeline pickle object.

        Returns
        -------
        FeaturePipeline
            Reconstructed pipeline instance with fitted encoders restored.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        config = FeaturesConfig.model_validate(state["config"])
        obj = cls(config=config)
        obj._encoders = state["encoders"]
        obj._label_encoders = state["label_encoders"]
        obj._skipped_text_cols = state.get("skipped_text_cols", set())
        obj._is_fitted = True
        log.info(f"FeaturePipeline loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframe(data: pd.DataFrame | list[dict]) -> pd.DataFrame:
        """Standardize raw input types to a pandas DataFrame.

        Parameters
        ----------
        data : pd.DataFrame | list[dict]
            Input data represented as a DataFrame or a list of dictionaries.

        Returns
        -------
        pd.DataFrame
            Normalized pandas DataFrame representation.

        Raises
        ------
        TypeError
            If data is neither a pandas DataFrame nor a list of dictionaries.
        """
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise TypeError(f"Expected DataFrame or list[dict], got {type(data)}")

    def _check_fitted(self) -> None:
        """Verify that the feature pipeline has been fitted prior to transform or save.

        Raises
        ------
        RuntimeError
            If ``_is_fitted`` is False.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeaturePipeline has not been fitted. Call .fit_transform() first."
            )