#!/usr/bin/env python3
"""Evaluate a trained ShortChain model on a test set.

Thin wrapper over ``shortchain.commands.evaluate`` — the same logic the
``shortchain evaluate`` console command runs, kept here for checkout users.
"""

from __future__ import annotations

from shortchain.commands.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())