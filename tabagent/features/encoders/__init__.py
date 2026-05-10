"""Text encoders for the TabAgent feature pipeline.

Provides a protocol-based abstraction over text encoding strategies:

- ``TfidfEncoder``: wraps scikit-learn's ``TfidfVectorizer`` (default).
- ``DenseEncoder``: uses ``sentence-transformers`` E5-small embeddings.
  Falls back to ``TfidfEncoder`` with a warning if the dependency is
  unavailable.

Use ``create_encoder()`` as the factory entry-point.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from tabagent.features.encoders.tfidf import TfidfEncoder
from tabagent.features.encoders.dense import DenseEncoder
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


__all__ = [
    "TextEncoder",
    "TfidfEncoder",
    "DenseEncoder",
    "create_encoder",
]
