"""Generic JSON / JSONL trajectory loader.

Reads agent execution logs and maps them to the internal ``Trajectory``
schema via configurable field mappings.  Works out-of-the-box with flat
JSON structures; for deeply-nested or agent-specific formats, implement
a dedicated ``TrajectoryLoader``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shortchain.config import IngestConfig, FieldMapConfig
from shortchain.ingest.schema import Step, Trajectory
from shortchain.utils.io import read_json, read_jsonl, find_files
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class JSONLTrajectoryLoader:
    """Load trajectories from JSON or JSONL files using field mappings.

    Parameters
    ----------
    config
        Ingestion configuration.  If ``None``, defaults are used.
    """

    def __init__(self, config: IngestConfig | None = None) -> None:
        self.config = config or IngestConfig()
        self.fm: FieldMapConfig = self.config.field_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> list[Trajectory]:
        """Load trajectories from a file or directory.

        Parameters
        ----------
        path
            Path to a ``.json`` / ``.jsonl`` file, or a directory
            containing such files.

        Returns
        -------
        list[Trajectory]
        """
        path = Path(path)
        if path.is_dir():
            return self._load_directory(path)
        return self._load_file(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_directory(self, directory: Path) -> list[Trajectory]:
        """Load all matching files from a directory."""
        pattern = f"*.{self.config.format}"
        files = find_files(directory, pattern)
        if not files:
            # Fallback: try both extensions
            files = find_files(directory, "*.json") + find_files(directory, "*.jsonl")

        if not files:
            log.warning(f"No trajectory files found in {directory}")
            return []

        log.info(f"Found [bold]{len(files)}[/bold] trajectory files in {directory}")

        trajectories: list[Trajectory] = []
        for fp in files:
            trajectories.extend(self._load_file(fp))

        log.info(
            f"Loaded [bold green]{len(trajectories)}[/bold green] trajectories"
            f" ({sum(t.success for t in trajectories)} successful)"
        )
        return trajectories

    def _load_file(self, file_path: Path) -> list[Trajectory]:
        """Load trajectories from a single file."""
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            raw_records = read_jsonl(file_path)
        elif suffix == ".json":
            data = read_json(file_path)
            raw_records = data if isinstance(data, list) else [data]
        else:
            log.warning(f"Skipping unsupported file type: {file_path}")
            return []

        trajectories: list[Trajectory] = []
        for i, record in enumerate(raw_records):
            try:
                traj = self._parse_record(record)
                if self.config.success_only and not traj.success:
                    continue
                trajectories.append(traj)
            except Exception as exc:
                log.warning(f"Skipping record {i} in {file_path}: {exc}")

        return trajectories

    def _parse_record(self, record: dict[str, Any]) -> Trajectory:
        """Parse a single raw record into a ``Trajectory``."""
        fm = self.fm

        # Extract steps
        raw_steps = record.get(fm.steps, [])
        steps: list[Step] = []
        for raw_step in raw_steps:
            if isinstance(raw_step, dict):
                steps.append(
                    Step(
                        agent_name=raw_step.get(fm.agent_name, ""),
                        action=raw_step.get(fm.action),
                        observation=raw_step.get(fm.observation),
                        thoughts=raw_step.get(fm.thoughts),
                        metadata={
                            k: v
                            for k, v in raw_step.items()
                            if k not in {fm.agent_name, fm.action, fm.observation, fm.thoughts}
                        },
                    )
                )
            elif isinstance(raw_step, str):
                # Simple format: each step is a tool name string
                steps.append(Step(action=raw_step))

        # Extract tools_used if explicitly provided
        tools_used: set[str] = set()
        if "tools_used" in record:
            tools_used = set(record["tools_used"])

        # Determine success
        success_val = record.get(fm.success, record.get("score", True))
        if isinstance(success_val, (int, float)):
            success = success_val >= 1.0
        else:
            success = bool(success_val)

        return Trajectory(
            task_id=str(record.get(fm.task_id, "")),
            intent=str(record.get(fm.intent, "")),
            steps=steps,
            success=success,
            app_name=str(record.get("app_name", record.get("app", ""))),
            tools_used=tools_used,
            metadata={
                k: v
                for k, v in record.items()
                if k not in {fm.task_id, fm.intent, fm.steps, fm.success, "app_name", "app",
                             "tools_used", "score"}
            },
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def load_trajectories(
    path: str | Path,
    config: IngestConfig | None = None,
) -> list[Trajectory]:
    """Load trajectories from *path* using the default JSONL loader.

    This is the recommended entry point for most use cases.
    """
    loader = JSONLTrajectoryLoader(config)
    return loader.load(path)
