"""Unified ``shortchain`` command-line interface.

Subcommands
-----------
- ``receive``   start the OTLP/HTTP telemetry receiver
- ``dataset``   build a pointwise training dataset from trajectories
- ``train``     train a classifier from a dataset
- ``evaluate``  evaluate a trained model on a test set

Each subcommand is a thin wrapper over the same logic exposed by the
``scripts/`` maintainer utilities, so the documented public interface is a
single ``shortchain`` entry point.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script(name: str):
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _print_help() -> None:
    print(
        "usage: shortchain <command> [options]\n\n"
        "commands:\n"
        "  receive    start the OTLP/HTTP telemetry receiver\n"
        "  dataset    build a training dataset from trajectories\n"
        "  train      train a classifier from a dataset\n"
        "  evaluate   evaluate a trained model on a test set"
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        _print_help()
        return 0

    sub = argv[0]
    rest = argv[1:]

    if sub == "receive":
        from shortchain.telemetry.cli import main as _receive

        _receive(rest)
    elif sub in ("dataset", "train", "evaluate"):
        # The script mains parse ``sys.argv[1:]``; drop the subcommand name.
        sys.argv = ["shortchain", *rest]
        _load_script("build_dataset" if sub == "dataset" else sub).main()
    else:
        _print_help()
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())