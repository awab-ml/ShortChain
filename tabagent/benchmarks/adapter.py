"""Benchmark adapter protocol.

Every benchmark dataset implements this protocol to plug into the
generic ``run_benchmark.py`` runner.  The adapter is responsible for:

1. Loading the tool catalog.
2. Loading trajectories (mapped to the universal ``Trajectory`` schema).
3. Optionally providing category metadata.
4. Optionally augmenting the training DataFrame (e.g., failure negatives).

The core pipeline (``DatasetBuilder``, ``FeaturePipeline``, ``Trainer``,
``metrics``) stays untouched — the adapter simply feeds it clean data.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from tabagent.ingest.schema import Trajectory


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Protocol for dataset-specific benchmark adapters.

    Implementations must provide at minimum ``name``,
    ``load_catalog()``, ``load_trajectories()``, and
    ``category_map()``.  The ``augment_training()`` hook has a
    default no-op implementation for adapters that don't need it.
    """

    name: str  # e.g., "toolbench", "apibank", "gorilla"

    def load_catalog(self) -> dict[str, str]:
        """Return ``{tool_name: description}`` catalog.

        The catalog defines the full universe of tools that the
        classifier will be evaluated against.
        """
        ...

    def load_trajectories(self, split: str) -> list[Trajectory]:
        """Load trajectories for *split* (``'train'`` or ``'test'``).

        Must return ``Trajectory`` objects with:
        - ``tools_used`` populated
        - ``metadata["available_tools"]`` if candidate-constraint is desired
        - ``metadata["step_index"]`` if step-level expansion was applied

        Parameters
        ----------
        split
            One of ``'train'`` or ``'test'``.
        """
        ...

    def category_map(self) -> dict[str, str]:
        """Return ``{tool_name: category}`` or empty dict.

        Used for stratified evaluation and category-aware metrics.
        """
        ...

    def augment_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional post-processing on the training DataFrame.

        Use for dataset-specific augmentation such as failure-negative
        injection.  The default implementation returns *df* unchanged.

        Parameters
        ----------
        df
            Training DataFrame produced by ``DatasetBuilder.build()``.

        Returns
        -------
        pd.DataFrame
            Augmented training DataFrame.
        """
        return df
