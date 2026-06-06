"""Structured logging setup for indx.

A single ``get_logger`` helper backed by Rich so CLI logs and library logs share one
format. Verbosity is a process-level concern set once from the CLI (``--quiet``/``--verbose``).
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

_CONFIGURED = False


def configure(level: int = logging.INFO) -> None:
    """Install a single RichHandler on the ``indx`` logger (idempotent)."""
    global _CONFIGURED
    logger = logging.getLogger("indx")
    logger.setLevel(level)
    if not _CONFIGURED:
        handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=False,
            show_time=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the ``indx`` logger (configuring it on first use)."""
    if not _CONFIGURED:
        configure()
    return logging.getLogger("indx" if name is None else f"indx.{name}")
