"""Logging utilities for ATLAS experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "atlas",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name.
        level: Logging level (default INFO).
        log_file: Optional path to a log file. If provided, logs will also
            be written to this file.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Optional file handler
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(path), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
