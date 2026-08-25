#!/usr/bin/env python3
"""Build a training dataset from agent execution trajectories.

Thin wrapper over ``shortchain.commands.build_dataset`` — the same logic the
``shortchain dataset`` console command runs, kept here for checkout users and
the reproducibility test suite.
"""

from __future__ import annotations

from shortchain.commands.build_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())