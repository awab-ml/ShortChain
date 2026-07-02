"""ToolBench benchmark adapter.

Wraps the existing ``JSONLTrajectoryLoader`` and generic ingestion
machinery behind the ``BenchmarkAdapter`` protocol.  No existing modules
are modified — this adapter simply composes them.

The adapter handles ToolBench-specific concerns:

- Loading the tool catalog from a directory of tool descriptions or
  deriving it from trajectories.
- Span-level expansion via the core ``expand_to_span_trajectories``
  transform (controlled by ``BenchmarkConfig.span_level``).
- Failure-negative augmentation on the training DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from shortchain.config import BenchmarkConfig, IngestConfig
from shortchain.data.transforms import expand_to_span_trajectories
from shortchain.ingest.loader import JSONLTrajectoryLoader
from shortchain.ingest.schema import Trajectory
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class ToolBenchAdapter:
    """Adapter for OpenBMB/ToolBench benchmark.

    Wraps the existing JSONL trajectory loader and catalog handling
    behind the ``BenchmarkAdapter`` protocol so the generic runner
    can treat ToolBench like any other benchmark.

    Parameters
    ----------
    benchmark_config
        Generic benchmark settings (span_level, failure negatives, etc.).
    ingest_config
        Ingestion settings (field mappings, success_only filter, etc.).
    train_path
        Path to training trajectory file/directory.
    eval_path
        Path to evaluation trajectory file/directory.
    catalog_path
        Optional path to a JSON file with ``{tool_name: description}``.
        If ``None``, the catalog is derived from loaded trajectories.
    """

    name = "toolbench"

    def __init__(
        self,
        benchmark_config: BenchmarkConfig | None = None,
        ingest_config: IngestConfig | None = None,
        train_path: str | Path | None = None,
        eval_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        self.benchmark_config = benchmark_config or BenchmarkConfig()
        self.ingest_config = ingest_config or IngestConfig()
        self.train_path = Path(train_path) if train_path else None
        self.eval_path = Path(eval_path) if eval_path else None
        self.catalog_path = Path(catalog_path) if catalog_path else None

        # Loader: success_only=False so we can access failures for augmentation
        train_ingest = IngestConfig(
            format=self.ingest_config.format,
            success_only=False,
            field_map=self.ingest_config.field_map,
        )
        self._train_loader = JSONLTrajectoryLoader(config=train_ingest)
        self._eval_loader = JSONLTrajectoryLoader(config=self.ingest_config)

        # Cached state
        self._catalog: dict[str, str] | None = None
        self._failed_trajs: list[Trajectory] = []
        self._category_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # BenchmarkAdapter protocol
    # ------------------------------------------------------------------

    def load_catalog(self) -> dict[str, str]:
        """Return ``{tool_name: description}`` catalog.

        If a ``catalog_path`` JSON file was provided, load from it.
        Otherwise, derive from all trajectories seen in ``load_trajectories()``.
        """
        if self._catalog is not None:
            return self._catalog

        if self.catalog_path and self.catalog_path.exists():
            from shortchain.utils.io import read_json

            raw = read_json(self.catalog_path)
            if isinstance(raw, dict):
                self._catalog = {str(k): str(v) for k, v in raw.items()}
            elif isinstance(raw, list):
                # List of {"name": ..., "description": ...} dicts
                self._catalog = {
                    item.get("name", ""): item.get("description", "")
                    for item in raw
                    if isinstance(item, dict)
                }
            else:
                self._catalog = {}
            log.info(
                f"Loaded catalog from {self.catalog_path}: "
                f"[bold]{len(self._catalog)}[/bold] tools"
            )
            return self._catalog

        # Will be lazily populated from trajectories
        self._catalog = {}
        return self._catalog

    def load_trajectories(self, split: str) -> list[Trajectory]:
        """Load trajectories for 'train' or 'test' split.

        For the training split:
        - Loads all trajectories (including failures).
        - Separates successful from failed trajectories.
        - Optionally applies span-level expansion to successful ones.
        - Caches failed trajectories for ``augment_training()``.

        For the test split:
        - Returns only successful trajectories.
        """
        if split == "train":
            return self._load_train()
        elif split == "test":
            return self._load_test()
        else:
            raise ValueError(f"Unknown split: {split!r}. Use 'train' or 'test'.")

    def category_map(self) -> dict[str, str]:
        """Return ``{tool_name: category}`` or empty dict."""
        return self._category_map

    def augment_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject failure-negative rows if configured.

        Uses failed trajectories cached during ``load_trajectories('train')``
        to create additional negative training examples.
        """
        if not self.benchmark_config.use_failure_negatives:
            return df
        if not self._failed_trajs:
            log.info("No failed trajectories available for failure-negative augmentation")
            return df

        catalog = self.load_catalog()
        ratio = self.benchmark_config.failure_negative_ratio

        # Build failure-negative rows
        fail_rows: list[dict[str, Any]] = []
        for traj in self._failed_trajs:
            for tool in traj.tools_used:
                if tool in catalog:
                    fail_rows.append({
                        "task_id": traj.task_id,
                        "tool_name": tool,
                        "label": 0,
                    })

        if not fail_rows:
            return df

        # Sample a fraction
        import random

        n_target = int(len(df) * ratio)
        n_sample = min(len(fail_rows), n_target)
        if n_sample > 0:
            sampled = random.sample(fail_rows, n_sample)
            fail_df = pd.DataFrame(sampled)
            # Only add columns that exist in the training df
            common_cols = [c for c in fail_df.columns if c in df.columns]
            if common_cols:
                augmented = pd.concat([df, fail_df[common_cols]], ignore_index=True)
                log.info(
                    f"Augmented training data with [bold]{n_sample}[/bold] "
                    f"failure negatives (total: {len(augmented)} rows)"
                )
                return augmented

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_train(self) -> list[Trajectory]:
        """Load training trajectories with optional span expansion."""
        if self.train_path is None:
            raise ValueError("train_path must be set to load training trajectories")

        all_trajs = self._train_loader.load(self.train_path)
        log.info(f"Loaded [bold]{len(all_trajs)}[/bold] total training trajectories")

        success = [t for t in all_trajs if t.success]
        self._failed_trajs = [t for t in all_trajs if not t.success]
        log.info(
            f"  ├─ [green]{len(success)}[/green] successful, "
            f"[yellow]{len(self._failed_trajs)}[/yellow] failed"
        )

        # Derive catalog from training data if not explicitly provided
        if not self._catalog:
            self._catalog = self._derive_catalog(all_trajs)

        # Span-level expansion (uses core transform from shortchain.data)
        if self.benchmark_config.span_level:
            expanded: list[Trajectory] = []
            for t in success:
                expanded.extend(expand_to_span_trajectories(t))
            log.info(
                f"  └─ Span expansion: {len(success)} trajectories → "
                f"[bold]{len(expanded)}[/bold] span-level samples"
            )
            return expanded

        return success

    def _load_test(self) -> list[Trajectory]:
        """Load test trajectories (successful only)."""
        if self.eval_path is None:
            raise ValueError("eval_path must be set to load test trajectories")

        trajs = self._eval_loader.load(self.eval_path)
        log.info(f"Loaded [bold]{len(trajs)}[/bold] test trajectories")
        return trajs

    def _derive_catalog(self, trajectories: list[Trajectory]) -> dict[str, str]:
        """Derive tool catalog from trajectory data."""
        catalog: dict[str, str] = {}
        for traj in trajectories:
            for tool in traj.tools_used:
                if tool not in catalog:
                    catalog[tool] = ""
        log.info(
            f"Derived catalog from trajectories: "
            f"[bold]{len(catalog)}[/bold] unique tools"
        )
        return catalog
