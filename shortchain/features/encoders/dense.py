"""Dense text encoder using ``sentence-transformers`` (optional dependency).

Falls back to ``TfidfEncoder`` with a warning if the ``sentence-transformers``
package is not installed or model loading fails.

Install the optional dependency via::

    pip install shortchain[embeddings]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shortchain.features.encoders.tfidf import TfidfEncoder
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


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
            self._dim = self._model.get_embedding_dimension()
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
