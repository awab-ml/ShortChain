"""Abstract base class for benchmark integration adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from shortchain.ingest.schema import Trajectory


class BaseBenchmarkAdapter(ABC):
    """Abstract interface standardizing benchmark data loading and audit."""

    @abstractmethod
    def load_trajectories(self, path: str | Path) -> list[Trajectory]:
        """Load benchmark execution traces into ShortChain Trajectory objects."""
        ...

    @abstractmethod
    def load_catalog(self, path: str | Path) -> dict[str, Any]:
        """Load benchmark tool catalog mapping tool names to descriptions or metadata."""
        ...

    @abstractmethod
    def audit_split_compliance(
        self,
        train_trajectories: list[Trajectory],
        test_trajectories: list[Trajectory],
    ) -> dict[str, Any]:
        """Audit train and test trajectories for compliance with ShortChain principles."""
        ...
