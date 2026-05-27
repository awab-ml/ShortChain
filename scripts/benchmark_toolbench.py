#!/usr/bin/env python3
"""DEPRECATED: ToolBench-specific benchmark script.

.. deprecated:: 0.2.0
    This script is soft-deprecated. Use ``scripts/run_benchmark.py`` with
    ``--benchmark toolbench`` instead.  This shim will be removed in the
    next release.

This script forwards all arguments to the generic ``run_benchmark.py``
runner.  It exists solely to avoid breaking existing workflows and
CI/CD pipelines during the transition period.

Usage (old — still works, but warns)::

    python scripts/benchmark_toolbench.py \\
        --train-path data/toolbench/train.jsonl \\
        --eval-path data/toolbench/test.jsonl

Equivalent (new — preferred)::

    python scripts/run_benchmark.py \\
        --benchmark toolbench \\
        --train-path data/toolbench/train.jsonl \\
        --eval-path data/toolbench/test.jsonl
"""

from __future__ import annotations

import sys
import warnings


_DEPRECATION_MSG = (
    "\n"
    "╔══════════════════════════════════════════════════════════════════╗\n"
    "║  ⚠  DEPRECATION WARNING                                       ║\n"
    "║                                                                ║\n"
    "║  benchmark_toolbench.py is DEPRECATED and will be removed in   ║\n"
    "║  the next release (v0.3.0).                                    ║\n"
    "║                                                                ║\n"
    "║  Use the generic runner instead:                               ║\n"
    "║                                                                ║\n"
    "║    python scripts/run_benchmark.py \\                           ║\n"
    "║        --benchmark toolbench \\                                 ║\n"
    "║        --train-path <TRAIN> \\                                  ║\n"
    "║        --eval-path <EVAL>                                      ║\n"
    "║                                                                ║\n"
    "╚══════════════════════════════════════════════════════════════════╝\n"
)


def main() -> None:
    # Print the deprecation warning prominently
    print(_DEPRECATION_MSG, file=sys.stderr)
    warnings.warn(
        "benchmark_toolbench.py is deprecated. "
        "Use 'scripts/run_benchmark.py --benchmark toolbench' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Inject --benchmark toolbench if not already present
    argv = sys.argv[1:]
    if "--benchmark" not in argv:
        argv = ["--benchmark", "toolbench"] + argv

    # Delegate to the generic runner
    sys.argv = [sys.argv[0]] + argv

    from scripts.run_benchmark import main as run_main

    run_main()


if __name__ == "__main__":
    main()
