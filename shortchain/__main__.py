"""Allow ``python -m shortchain <subcommand>``."""

from __future__ import annotations

import sys

from shortchain.cli import main

if __name__ == "__main__":
    sys.exit(main())