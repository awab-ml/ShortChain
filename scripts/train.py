#!/usr/bin/env python3
"""Train a ShortChain classifier.

Thin wrapper over ``shortchain.commands.train`` — the same logic the
``shortchain train`` console command runs, kept here for checkout users.
"""

from __future__ import annotations

from shortchain.commands.train import main

if __name__ == "__main__":
    raise SystemExit(main())