"""Text encoders for the TabAgent feature pipeline.

Provides a protocol-based abstraction over text encoding strategies:

- ``TfidfEncoder``: wraps scikit-learn's ``TfidfVectorizer`` (default).
- ``DenseEncoder``: uses ``sentence-transformers`` E5-small embeddings.
  Falls back to ``TfidfEncoder`` with a warning if the dependency is
  unavailable.

Use ``create_encoder()`` as the factory entry-point.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from tabagent.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TextEncoder(Protocol):
    """Protocol for text encoding strategies."""

    def fit(self, texts: list[str]) -> "TextEncoder":
        """Fit on a list of raw text strings."""
        ...

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform raw text strings into a feature matrix."""
        ...

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        ...

    @property
    def output_dim(self) -> int:
        """Dimensionality of the output feature vectors."""
        ...


# ---------------------------------------------------------------------------
# TF-IDF encoder
# ---------------------------------------------------------------------------

class TfidfEncoder:
    """TF-IDF text encoder wrapping ``sklearn.TfidfVectorizer``.

    Parameters
    ----------
    max_features
        Maximum vocabulary size.
    """

    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> "TfidfEncoder":
        """Fit the TF-IDF vocabulary on *texts*."""
        clean = [str(t) if t else "" for t in texts]
        try:
            self._vectorizer.fit(clean)
            self._is_fitted = True
        except ValueError:
            # All-empty corpus — mark fitted with empty vocab
            log.warning("TF-IDF fit on empty/stop-word-only corpus; encoder will produce zeros")
            self._is_fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform *texts* to a dense TF-IDF matrix."""
        clean = [str(t) if t else "" for t in texts]
        if not self._is_fitted or not hasattr(self._vectorizer, "vocabulary_"):
            return np.zeros((len(clean), 1), dtype=np.float32)
        mat = self._vectorizer.transform(clean)
        return mat.toarray()

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        clean = [str(t) if t else "" for t in texts]
        try:
            mat = self._vectorizer.fit_transform(clean)
            self._is_fitted = True
            return mat.toarray()
        except ValueError:
            self._is_fitted = True
            return np.zeros((len(clean), 1), dtype=np.float32)

    @property
    def output_dim(self) -> int:
        if self._is_fitted and hasattr(self._vectorizer, "vocabulary_"):
            return len(self._vectorizer.vocabulary_)
        return 0


# ---------------------------------------------------------------------------
# Dense (E5-small) encoder
# ---------------------------------------------------------------------------

class DenseEncoder:
    """Dense text encoder using ``sentence-transformers``.

    Falls back to ``TfidfEncoder`` with a warning if
    ``sentence-transformers`` is not installed or model loading fails.

    Parameters
    ----------
    model_name
        HuggingFace model identifier.
    max_features
        TF-IDF fallback max features (only used on fallback).
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        max_features: int = 5000,
    ) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._fallback: TfidfEncoder | None = None
        self._dim: int = 0

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            log.info(f"DenseEncoder loaded: {model_name} (dim={self._dim})")
        except Exception as exc:
            log.warning(
                f"Could not load DenseEncoder ({model_name}): {exc}. "
                "Falling back to TfidfEncoder."
            )
            self._fallback = TfidfEncoder(max_features=max_features)

    @property
    def _is_fallback(self) -> bool:
        return self._fallback is not None

    def fit(self, texts: list[str]) -> "DenseEncoder":
        """Fit (no-op for pretrained models, fits TF-IDF on fallback)."""
        if self._is_fallback:
            self._fallback.fit(texts)
        # Pretrained models don't need fitting
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Encode texts to dense vectors."""
        if self._is_fallback:
            return self._fallback.transform(texts)
        clean = [f"query: {str(t)}" if t else "query: " for t in texts]
        embeddings = self._model.encode(clean, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    @property
    def output_dim(self) -> int:
        if self._is_fallback:
            return self._fallback.output_dim
        return self._dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_encoder(
    name: str = "tfidf",
    max_features: int = 5000,
    model_name: str = "intfloat/e5-small-v2",
) -> TfidfEncoder | DenseEncoder:
    """Create a text encoder by name.

    Parameters
    ----------
    name
        One of ``'tfidf'``, ``'e5-small'``, or ``'auto'``.
        ``'auto'`` tries ``DenseEncoder`` first, falls back to TF-IDF.
    max_features
        TF-IDF vocabulary size.
    model_name
        HuggingFace model for dense encoding.

    Returns
    -------
    TfidfEncoder | DenseEncoder
    """
    name = name.lower().strip()

    if name == "tfidf":
        return TfidfEncoder(max_features=max_features)
    elif name in ("e5-small", "e5_small", "dense"):
        return DenseEncoder(model_name=model_name, max_features=max_features)
    elif name == "auto":
        enc = DenseEncoder(model_name=model_name, max_features=max_features)
        if enc._is_fallback:
            log.info("Auto encoder: using TF-IDF (sentence-transformers unavailable)")
        return enc
    else:
        raise ValueError(f"Unknown encoder: {name!r}. Choose from: tfidf, e5-small, auto")
