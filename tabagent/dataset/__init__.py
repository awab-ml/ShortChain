"""Dataset construction — transform trajectories into training data."""

from tabagent.dataset.builder import DatasetBuilder, build_dataset
from tabagent.dataset.splitter import GroupStratifiedSplitter

__all__ = [
    "DatasetBuilder",
    "build_dataset",
    "GroupStratifiedSplitter",
]
