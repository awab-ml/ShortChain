"""Feature pipeline for ShortChain.

Provides modular feature extraction, text encoding, and a composable
pipeline that separates feature engineering from the classifier.
"""

from shortchain.features.context import ContextFeatureBuilder
from shortchain.features.encoders import (
    TfidfEncoder,
    DenseEncoder,
    create_encoder,
)
from shortchain.features.pipeline import FeaturePipeline
from shortchain.features.stats import CorpusStats
from shortchain.features.tool import ToolFeatureBuilder

__all__ = [
    "ContextFeatureBuilder",
    "CorpusStats",
    "DenseEncoder",
    "FeaturePipeline",
    "TfidfEncoder",
    "ToolFeatureBuilder",
    "create_encoder",
]
