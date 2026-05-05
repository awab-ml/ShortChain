"""Feature pipeline for TabAgent.

Provides modular feature extraction, text encoding, and a composable
pipeline that separates feature engineering from the classifier.
"""

from tabagent.features.context import ContextFeatureBuilder
from tabagent.features.encoders import (
    TfidfEncoder,
    DenseEncoder,
    create_encoder,
)
from tabagent.features.pipeline import FeaturePipeline
from tabagent.features.stats import CorpusStats
from tabagent.features.tool import ToolFeatureBuilder

__all__ = [
    "ContextFeatureBuilder",
    "CorpusStats",
    "DenseEncoder",
    "FeaturePipeline",
    "TfidfEncoder",
    "ToolFeatureBuilder",
    "create_encoder",
]
