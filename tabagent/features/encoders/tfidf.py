"""TF-IDF text encoder for the TabAgent feature pipeline.

Wraps scikit-learn's ``TfidfVectorizer`` with a consistent interface
matching the ``TextEncoder`` protocol.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from tabagent.utils.logging import get_logger

log = get_logger(__name__)


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
