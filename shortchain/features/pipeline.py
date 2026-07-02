"""Feature pipeline orchestrator for ShortChain.

``FeaturePipeline`` is the single entry-point for transforming raw
feature dictionaries or DataFrames into the numeric matrix consumed
by the classifier.  It replaces the inline encoding logic previously
baked into ``classifier.py``.

Accepts **both** ``pd.DataFrame`` and ``list[dict]`` as input for
production flexibility.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from shortchain.config import FeaturesConfig
from shortchain.features.encoders import TfidfEncoder, create_encoder
from shortchain.utils.logging import get_logger

log = get_logger(__name__)

# Text columns that get encoded via the text encoder
_TEXT_COLS = [
    "intent",
    "previous_tools",
    "last_thought",
    "tool_name",
    "tool_description",
    "history_summary",
    "last_observation",
]

# Numeric columns passed through directly
_NUM_COLS = [
    "n_spans",
    "span_index",
    "unique_tools_so_far",
    "tool_diversity",
    "app_tool_count",
    "tool_name_length",
    "tool_frequency",
    "tool_co_occurrence",
]

# Boolean columns converted to 0/1
_BOOL_COLS = [
    "has_description",
    "tool_app_match",
]

# Categorical columns that get label-encoded
_CAT_COLS = ["app_name"]


class FeaturePipeline:
    """Orchestrates feature extraction and encoding.

    Parameters
    ----------
    config
        Feature pipeline configuration.
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
        """Fit the pipeline on *data* and return encoded features.

        Parameters
        ----------
        data
            Raw feature data as a DataFrame or list of dicts.

        Returns
        -------
        np.ndarray
            Encoded feature matrix of shape ``(n_samples, n_features)``.
        """
        df = self._to_dataframe(data)
        parts: list[np.ndarray] = []

        # Text columns → encoder
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

        # Categorical → label encoding
        for col in _CAT_COLS:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            vals = df[col].fillna("__unknown__").astype(str)
            encoded = le.fit_transform(vals).reshape(-1, 1).astype(np.float32)
            self._label_encoders[col] = le
            parts.append(encoded)

        # Numeric columns
        for col in _NUM_COLS:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            parts.append(vals.values.reshape(-1, 1).astype(np.float32))

        # Boolean columns
        for col in _BOOL_COLS:
            if col not in df.columns:
                continue
            vals = df[col].astype(int).values.reshape(-1, 1).astype(np.float32)
            parts.append(vals)

        if not parts:
            raise ValueError("No features to encode — check column names")

        self._is_fitted = True
        return np.hstack(parts)

    def transform(self, data: pd.DataFrame | list[dict]) -> np.ndarray:
        """Transform new data using fitted encoders.

        Parameters
        ----------
        data
            Raw feature data.

        Returns
        -------
        np.ndarray
            Encoded feature matrix.
        """
        self._check_fitted()
        df = self._to_dataframe(data)
        parts: list[np.ndarray] = []

        # Text columns
        for col in _TEXT_COLS:
            if col not in self._encoders:
                continue
            enc = self._encoders[col]
            texts = df[col].fillna("").astype(str).tolist()
            parts.append(enc.transform(texts))

        # Categorical
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

        # Numeric columns
        for col in _NUM_COLS:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            parts.append(vals.values.reshape(-1, 1).astype(np.float32))

        # Boolean columns
        for col in _BOOL_COLS:
            if col not in df.columns:
                continue
            vals = df[col].astype(int).values.reshape(-1, 1).astype(np.float32)
            parts.append(vals)

        if not parts:
            raise ValueError("No features to encode — check column names")

        return np.hstack(parts)

    def save(self, path: str | Path) -> None:
        """Persist the fitted pipeline (pickle)."""
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
        """Load a fitted pipeline from disk."""
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
        """Normalise input to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise TypeError(f"Expected DataFrame or list[dict], got {type(data)}")

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "FeaturePipeline has not been fitted. Call .fit_transform() first."
            )
