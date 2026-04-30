"""Structured logging for TabAgent.

Provides a pre-configured Rich-based logger that all modules import.
"""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with Rich console output.

    Parameters
    ----------
    name
        Logger name, typically ``__name__`` from the calling module.
    level
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        handler.setLevel(level)
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)

    # Prevent duplicate logs when library is used inside another app
    logger.propagate = False
    return logger


def setup_file_logging(
    log_path: str,
    level: int = logging.DEBUG,
) -> None:
    """Add a file handler to the root ``tabagent`` logger.

    Useful for persisting full debug output during long pipeline runs.
    """
    root = logging.getLogger("tabagent")
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)


# Module-level convenience so callers can do:
#   from tabagent.utils.logging import log
log = get_logger("tabagent")
