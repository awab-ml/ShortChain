"""Unified ``shortchain`` command-line interface.

Subcommands
-----------
- ``receive``   start the OTLP/HTTP telemetry receiver
- ``dataset``   build a pointwise training dataset from trajectories
- ``train``     train a classifier from a dataset
- ``evaluate``  evaluate a trained model on a test set

Every subcommand lives inside the installed package (``shortchain.commands``
and ``shortchain.telemetry``), so the entry point works from a wheel with no
dependency on the checkout's ``scripts/`` directory.
"""

from __future__ import annotations

import sys


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        _print_help()
        return 0

    sub = argv[0]
    rest = argv[1:]

    if sub == "receive":
        from shortchain.telemetry.cli import main as _receive

        _receive(rest)
    elif sub == "dataset":
        from shortchain.commands.build_dataset import main as _dataset

        _dataset(rest)
    elif sub == "train":
        from shortchain.commands.train import main as _train

        _train(rest)
    elif sub == "evaluate":
        from shortchain.commands.evaluate import main as _evaluate

        _evaluate(rest)
    elif sub in ("-h", "--help", "help"):
        _print_help()
    else:
        _print_help()
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())