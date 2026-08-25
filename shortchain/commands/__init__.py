"""CLI subcommands packaged for ``pip install shortchain``.

These modules implement the ``shortchain dataset|train|evaluate`` commands
inside the installed package so the console script works from a wheel (no
dependency on a ``scripts/`` directory). The repo's ``scripts/*.py`` files are
thin wrappers over these functions.
"""

from shortchain.commands.build_dataset import main as build_dataset_main
from shortchain.commands.evaluate import main as evaluate_main
from shortchain.commands.train import main as train_main

__all__ = ["build_dataset_main", "train_main", "evaluate_main"]