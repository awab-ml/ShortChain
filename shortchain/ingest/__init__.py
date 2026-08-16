"""Trajectory ingestion — load and validate agent execution logs."""

from shortchain.ingest.schema import Span, Trajectory
from shortchain.ingest.base import TrajectoryLoader
from shortchain.ingest.loader import JSONLTrajectoryLoader, load_trajectories
from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    OtelTraceProjector,
    ProjectionResult,
)

__all__ = [
    "Span",
    "Trajectory",
    "TrajectoryLoader",
    "JSONLTrajectoryLoader",
    "load_trajectories",
    "OtelSpan",
    "OtelTrace",
    "OtelTraceProjector",
    "ProjectionResult",
]