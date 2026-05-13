"""Tests for SemanticSimilarityEncoder and pipeline integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabagent.config import FeaturesConfig
from tabagent.features.encoders.similarity import SemanticSimilarityEncoder
from tabagent.features.pipeline import FeaturePipeline


# ------------------------------------------------------------------
# SemanticSimilarityEncoder
# ------------------------------------------------------------------


class TestSemanticSimilarityEncoder:
    """Tests for the similarity encoder (uses fallback if no model)."""

    def test_identical_texts_high_similarity(self) -> None:
        """Identical texts should have similarity close to 1.0."""
        enc = SemanticSimilarityEncoder()
        texts = ["get the weather forecast"] * 3
        sims = enc.compute_similarity(texts, texts)

        assert sims.shape == (3,)
        for s in sims:
            assert s > 0.8  # Should be very high

    def test_unrelated_texts_lower_similarity(self) -> None:
        """Unrelated texts should have lower similarity."""
        enc = SemanticSimilarityEncoder()
        texts_a = ["get weather forecast", "book a flight"]
        texts_b = ["play heavy metal music", "quantum physics equation"]
        sims = enc.compute_similarity(texts_a, texts_b)

        assert sims.shape == (2,)
        # Should be lower than identical texts (exact value depends on model vs fallback)
        identical_sims = enc.compute_similarity(texts_a, texts_a)
        assert sims.mean() < identical_sims.mean()

    def test_empty_texts(self) -> None:
        """Empty input should return empty array."""
        enc = SemanticSimilarityEncoder()
        sims = enc.compute_similarity([], [])
        assert sims.shape == (0,)

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths should raise ValueError."""
        enc = SemanticSimilarityEncoder()
        with pytest.raises(ValueError, match="Length mismatch"):
            enc.compute_similarity(["a", "b"], ["c"])

    def test_output_shape(self) -> None:
        """Output should be (n_samples,) float32."""
        enc = SemanticSimilarityEncoder()
        texts_a = ["hello world", "foo bar", "test query"]
        texts_b = ["hi earth", "baz qux", "test passage"]
        sims = enc.compute_similarity(texts_a, texts_b)

        assert sims.shape == (3,)
        assert sims.dtype == np.float32

    def test_similarity_range(self) -> None:
        """Similarities should be in [-1, 1]."""
        enc = SemanticSimilarityEncoder()
        texts_a = ["weather", "flights", "music"]
        texts_b = ["rain forecast", "airplane tickets", "guitar solo"]
        sims = enc.compute_similarity(texts_a, texts_b)

        assert all(-1.0 <= s <= 1.0 for s in sims)

    def test_fallback_similarity(self) -> None:
        """Fallback (Jaccard) should work without model."""
        sims = SemanticSimilarityEncoder._fallback_similarity(
            ["the weather is nice", "buy some food"],
            ["the weather forecast", "play some music"],
        )
        assert sims.shape == (2,)
        # "the weather" overlaps in first pair → higher
        assert sims[0] > sims[1]


# ------------------------------------------------------------------
# Pipeline integration
# ------------------------------------------------------------------


class TestPipelineWithSimilarity:
    """Tests for FeaturePipeline with semantic similarity."""

    def _make_df(self) -> pd.DataFrame:
        """Create a minimal DataFrame with required columns."""
        return pd.DataFrame({
            "intent": ["get weather", "book flight", "get weather"],
            "tool_name": ["weather_api", "flight_api", "rain_api"],
            "tool_description": [
                "Get weather forecast",
                "Book airplane tickets",
                "Get rain predictions",
            ],
            "tool_category": ["Weather", "Travel", "Weather"],
            "last_thought": ["need weather", "need flight", "check rain"],
            "app_name": ["Weather", "Travel", "Weather"],
            "n_steps": [1, 2, 1],
            "has_description": [True, True, True],
            "label": [1, 1, 0],
        })

    def test_similarity_disabled_by_default(self) -> None:
        """With default config, no similarity features should be added."""
        config = FeaturesConfig()  # default: include_semantic_similarity=False
        pipeline = FeaturePipeline(config=config)
        df = self._make_df()

        X = pipeline.fit_transform(df)
        assert pipeline._similarity_encoder is None

    def test_similarity_enabled_adds_features(self) -> None:
        """With similarity enabled, extra features should be added."""
        config_off = FeaturesConfig(include_semantic_similarity=False)
        config_on = FeaturesConfig(include_semantic_similarity=True)

        df = self._make_df()

        pipeline_off = FeaturePipeline(config=config_off)
        X_off = pipeline_off.fit_transform(df)

        pipeline_on = FeaturePipeline(config=config_on)
        X_on = pipeline_on.fit_transform(df)

        # Similarity adds 3 extra columns
        assert X_on.shape[1] == X_off.shape[1] + 3
        assert X_on.shape[0] == X_off.shape[0]

    def test_similarity_features_are_numeric(self) -> None:
        """Similarity features should be finite floats."""
        config = FeaturesConfig(include_semantic_similarity=True)
        pipeline = FeaturePipeline(config=config)
        df = self._make_df()

        X = pipeline.fit_transform(df)
        # Last 3 columns are similarity features
        sim_features = X[:, -3:]
        assert np.all(np.isfinite(sim_features))

    def test_transform_matches_fit_transform_shape(self) -> None:
        """transform() should produce same shape as fit_transform()."""
        config = FeaturesConfig(include_semantic_similarity=True)
        pipeline = FeaturePipeline(config=config)
        df = self._make_df()

        X_fit = pipeline.fit_transform(df)
        X_transform = pipeline.transform(df)

        assert X_fit.shape == X_transform.shape
