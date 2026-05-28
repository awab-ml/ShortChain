"""File I/O helpers for ShortChain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_json(path: str | Path) -> Any:
    """Read a JSON file and return the parsed object."""
    with open(path) as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-Lines file and return a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lazily iterate over a JSON-Lines file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write *data* as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    return path


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write *records* as JSON-Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    return path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files(directory: str | Path, pattern: str = "*.jsonl") -> list[Path]:
    """Recursively find files matching *pattern* under *directory*."""
    return sorted(Path(directory).rglob(pattern))
