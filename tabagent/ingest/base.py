"""Abstract trajectory loader protocol.

Any data source that can produce ``Trajectory`` objects should implement
this protocol.  The generic ``JSONLTrajectoryLoader`` covers the common
case; agent-specific loaders (CUGA, LangChain, etc.) can be added later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from tabagent.ingest.schema import Trajectory


@runtime_checkable
class TrajectoryLoader(Protocol):
    """Protocol for trajectory data loaders."""

    def load(self, path: str | Path) -> list[Trajectory]:
        """Load trajectories from *path* (file or directory).

        Returns
        -------
        list[Trajectory]
            Validated trajectory objects ready for downstream processing.
        """
        ...
