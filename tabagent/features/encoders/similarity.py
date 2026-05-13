"""Semantic similarity encoder using e5-small-v2 (or compatible models).

Computes pairwise cosine similarity between two text columns (e.g.,
``intent`` ↔ ``tool_description``) using pre-trained sentence embeddings.
This is used as a **numeric feature** alongside TF-IDF, not as a
replacement.

Key design:

- **Lazy-loads** the model (only when called).
- **Deduplicates** texts before embedding for efficiency.
- **Correct prefixes**: ``"query: "`` for queries, ``"passage: "`` for documents.
- **Graceful fallback** to token-overlap similarity if ``sentence-transformers``
  is not installed.

Install the optional dependency via::

    pip install tabagent[embeddings]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tabagent.utils.logging import get_logger

log = get_logger(__name__)


class SemanticSimilarityEncoder:
    """Compute cosine similarity between text pairs using dense embeddings.

    Parameters
    ----------
    model_name
        HuggingFace model identifier.
    """

    def __init__(self, model_name: str = "intfloat/e5-small-v2") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._available = False
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._available = True
            log.info(
                f"SemanticSimilarityEncoder loaded: {self.model_name} "
                f"(dim={self._model.get_embedding_dimension()})"
            )
        except Exception as exc:
            log.warning(
                f"Could not load similarity model ({self.model_name}): {exc}. "
                "Falling back to token-overlap similarity."
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        """Whether the dense model loaded successfully."""
        return self._available

    def compute_similarity(
        self,
        texts_a: list[str],
        texts_b: list[str],
    ) -> np.ndarray:
        """Compute cosine similarity for each ``(a, b)`` pair.

        Parameters
        ----------
        texts_a
            Query texts (e.g., intents). Prefixed with ``"query: "``.
        texts_b
            Passage texts (e.g., tool descriptions). Prefixed with ``"passage: "``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` of float32 similarities in ``[-1, 1]``.
        """
        if len(texts_a) != len(texts_b):
            raise ValueError(
                f"Length mismatch: {len(texts_a)} vs {len(texts_b)}"
            )
        n = len(texts_a)
        if n == 0:
            return np.array([], dtype=np.float32)

        if not self._available:
            return self._fallback_similarity(texts_a, texts_b)

        # Deduplicate for efficiency
        unique_a = list(set(texts_a))
        unique_b = list(set(texts_b))

        # Embed with correct e5 prefixes
        emb_a_map = self._embed_unique(unique_a, prefix="query: ")
        emb_b_map = self._embed_unique(unique_b, prefix="passage: ")

        # Compute per-row cosine similarity
        sims = np.zeros(n, dtype=np.float32)
        for i in range(n):
            va = emb_a_map[texts_a[i]]
            vb = emb_b_map[texts_b[i]]
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a > 0 and norm_b > 0:
                sims[i] = float(np.dot(va, vb) / (norm_a * norm_b))
            else:
                sims[i] = 0.0

        return sims

    def _embed_unique(
        self, texts: list[str], prefix: str
    ) -> dict[str, np.ndarray]:
        """Embed unique texts and return a text→embedding mapping."""
        prefixed = [f"{prefix}{t}" if t else f"{prefix}" for t in texts]
        embeddings = self._model.encode(prefixed, show_progress_bar=False)
        return {
            text: np.asarray(emb, dtype=np.float32)
            for text, emb in zip(texts, embeddings)
        }

    @staticmethod
    def _fallback_similarity(
        texts_a: list[str], texts_b: list[str]
    ) -> np.ndarray:
        """Token-overlap (Jaccard) similarity as fallback.

        Used when ``sentence-transformers`` is not installed.
        """
        sims = np.zeros(len(texts_a), dtype=np.float32)
        for i, (a, b) in enumerate(zip(texts_a, texts_b)):
            tokens_a = set(str(a).lower().split())
            tokens_b = set(str(b).lower().split())
            if tokens_a or tokens_b:
                intersection = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                sims[i] = intersection / max(union, 1)
        return sims
