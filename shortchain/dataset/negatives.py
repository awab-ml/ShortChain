"""Negative sampling strategies for dataset construction.

Provides pluggable strategies for selecting negative (label=0) tools:

- ``RandomSampler``: uniform random from catalog \\ positives (Phase 1 default).
- ``HardNegativeSampler``: same-app, co-usage, and description-similar tools.
- ``MixedSampler``: configurable mix of random + hard negatives.

Use ``create_sampler()`` as the factory entry-point.
"""
from __future__ import annotations
from pandas import plotting



import random
from typing import Any

import numpy as np

from shortchain.config import NegativeSamplingConfig
from shortchain.features.stats import CorpusStats
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class NegativeSampler:
    """Base class for negative sampling strategies.

    Parameters
    ----------
    catalog
        Full tool catalog ``{tool_name: description}``.
    corpus_stats
        Optional precomputed corpus statistics.
    random_state
        Seed for reproducible sampling.
    """

    def __init__(
        self,
        catalog: dict[str, str],
        corpus_stats: CorpusStats | None = None,
        random_state: int | None = None,
    ) -> None:
        self.catalog = catalog
        self.corpus_stats = corpus_stats
        self._rng = random.Random(random_state)

    def sample(
        self,
        positive_tools: set[str],
        app_name: str,
        n: int,
    ) -> list[str]:
        """Sample *n* negative tools.

        Parameters
        ----------
        positive_tools
            Tools that are positive (should be excluded).
        app_name
            App context of the current trajectory.
        n
            Number of negative tools to return.

        Returns
        -------
        list[str]
            Sampled negative tool names.
        """
        raise NotImplementedError


class RandomSampler(NegativeSampler):
    """Uniform random negative sampling (Phase 1 behaviour)."""

    def sample(
        self,
        positive_tools: set[str],
        app_name: str,
        n: int,):
        pool = [t for t in self.catalog if t not in positive_tools]
        k = min(len(pool), n)
        if k == 0:
            return []
        return self._rng.sample(pool, k)


class HardNegativeSampler(NegativeSampler):
    """Hard negative sampling based on same-app, co-usage, and similarity.

    Precomputes candidate pools at construction time so that ``sample()``
    is fast.

    Parameters
    ----------
    same_app_weight
        Proportion of hard negatives from same-app tools.
    co_usage_weight
        Proportion from co-usage ranked tools.
    similarity_weight
        Proportion from description-similar tools (uses simple
        token overlap for now; dense similarity in Phase 3).
    """

    def __init__(
        self,
        catalog: dict[str, str],
        corpus_stats: CorpusStats | None = None,
        random_state: int | None = None,
        same_app_weight: float = 0.4,
        co_usage_weight: float = 0.3,
        similarity_weight: float = 0.3,
    ) -> None:
        super().__init__(catalog, corpus_stats, random_state)
        self.same_app_weight = same_app_weight
        self.co_usage_weight = co_usage_weight
        self.similarity_weight = similarity_weight

        # Precompute: same-app candidate pools
        self._app_pools: dict[str, list[str]] = {}
        if corpus_stats:
            for app, tools in corpus_stats.app_tools.items():
                self._app_pools[app] = sorted(tools)

        # Precompute: co-usage rankings per tool
        self._co_usage_ranked: dict[str, list[str]] = {}
        if corpus_stats:
            for tool, co_occ in corpus_stats.co_occurrence.items():
                ranked = sorted(co_occ, key=co_occ.get, reverse=True)
                self._co_usage_ranked[tool] = ranked

        # Precompute: description token sets for similarity
        self._desc_tokens: dict[str, set[str]] = {}
        for name, desc in catalog.items():
            tokens = set((desc or name).lower().split())
            self._desc_tokens[name] = tokens

    def sample(
        self,
        positive_tools: set[str],
        app_name: str,
        n: int,
    ) -> list[str]:
        pool = set(self.catalog) - positive_tools
        if not pool:
            return []

        n_same_app = max(1, int(n * self.same_app_weight))
        n_co_usage = max(1, int(n * self.co_usage_weight))
        n_similar = n - n_same_app - n_co_usage

        selected: list[str] = []

        # 1. Same-app negatives
        same_app = [t for t in self._app_pools.get(app_name, []) if t in pool]
        self._rng.shuffle(same_app)
        selected.extend(same_app[:n_same_app])

        # 2. Co-usage negatives (tools that co-occur with positives)
        co_usage_candidates: list[str] = []
        for pos_tool in positive_tools:
            for t in self._co_usage_ranked.get(pos_tool, []):
                if t in pool and t not in selected and t not in positive_tools:
                    co_usage_candidates.append(t)
        # Deduplicate preserving order
        seen = set(selected)
        co_usage_unique = []
        for t in co_usage_candidates:
            if t not in seen:
                co_usage_unique.append(t)
                seen.add(t)
        selected.extend(co_usage_unique[:n_co_usage])

        # 3. Description-similar negatives (token overlap)
        remaining_pool = [t for t in pool if t not in set(selected)]
        if remaining_pool and positive_tools:
            pos_tokens = set()
            for pt in positive_tools:
                pos_tokens |= self._desc_tokens.get(pt, set())
            if pos_tokens:
                scored = []
                for t in remaining_pool:
                    overlap = len(self._desc_tokens.get(t, set()) & pos_tokens)
                    scored.append((t, overlap))
                scored.sort(key=lambda x: x[1], reverse=True)
                selected.extend(t for t, _ in scored[:n_similar])

        # Fill any remaining slots with random
        if len(selected) < n:
            leftover = [t for t in pool if t not in set(selected)]
            self._rng.shuffle(leftover)
            selected.extend(leftover[: n - len(selected)])

        return selected[:n]


class MixedSampler(NegativeSampler):
    """Mix of random and hard negatives.

    Parameters
    ----------
    hard_ratio
        Fraction of negatives that are hard (rest are random).
    """

    def __init__(
        self,
        catalog: dict[str, str],
        corpus_stats: CorpusStats | None = None,
        random_state: int | None = None,
        hard_ratio: float = 0.5,
        same_app_weight: float = 0.4,
        co_usage_weight: float = 0.3,
        similarity_weight: float = 0.3,
    ) -> None:
        super().__init__(catalog, corpus_stats, random_state)
        self.hard_ratio = hard_ratio
        self._hard = HardNegativeSampler(
            catalog, corpus_stats, random_state,
            same_app_weight, co_usage_weight, similarity_weight,
        )
        self._random = RandomSampler(catalog, corpus_stats, random_state)

    def sample(
        self,
        positive_tools: set[str],
        app_name: str,
        n: int,
    ) -> list[str]:
        n_hard = max(1, int(n * self.hard_ratio))
        n_random = n - n_hard

        hard_samples = self._hard.sample(positive_tools, app_name, n_hard)
        exclude = positive_tools | set(hard_samples)

        # Random from the remaining pool
        remaining_pool = [t for t in self.catalog if t not in exclude]
        k = min(len(remaining_pool), n_random)
        random_samples = self._rng.sample(remaining_pool, k) if k > 0 else []

        result = hard_samples + random_samples
        return result[:n]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_sampler(
    config: NegativeSamplingConfig | None = None,
    catalog: dict[str, str] | None = None,
    corpus_stats: CorpusStats | None = None,
) -> NegativeSampler:
    """Create a negative sampler from config.

    Parameters
    ----------
    config
        Sampling strategy configuration.
    catalog
        Tool catalog ``{name: description}``.
    corpus_stats
        Precomputed corpus statistics.

    Returns
    -------
    NegativeSampler
    """
    config = config or NegativeSamplingConfig()
    catalog = catalog or {}

    strategy = config.strategy.lower().strip()

    if strategy == "random":
        return RandomSampler(
            catalog, corpus_stats, config.random_state,
        )
    elif strategy == "hard":
        return HardNegativeSampler(
            catalog, corpus_stats, config.random_state,
            config.same_app_weight, config.co_usage_weight,
            config.similarity_weight,
        )
    elif strategy == "mixed":
        return MixedSampler(
            catalog, corpus_stats, config.random_state,
            config.hard_negative_ratio,
            config.same_app_weight, config.co_usage_weight,
            config.similarity_weight,
        )
    else:
        raise ValueError(
            f"Unknown negative strategy: {strategy!r}. "
            "Choose from: random, hard, mixed"
        )
