"""
Logging utilities for Emily SLM.

Provides a centralized logger factory with Rich-based colorized console output
and optional file logging. All modules should use ``get_logger(__name__)``
instead of calling ``logging.getLogger`` directly so formatting is consistent.

Example:
    >>> from slm.utils.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Module-level console (shared across all loggers)
_console = Console(stderr=True)

# Registry so we don't configure the same logger twice
_configured_loggers: set[str] = set()


def setup_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: Optional[str | Path] = None,
    rich_console: bool = True,
) -> logging.Logger:
    """
    Configure and return a named logger with Rich console output.

    If the logger has already been configured, it is returned as-is without
    adding duplicate handlers.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level as integer or string (e.g., ``"DEBUG"``).
        log_file: Optional path to a log file. When provided, a
            ``FileHandler`` writing plain text is added alongside the
            console handler.
        rich_console: Use ``RichHandler`` for colourised console output.
            Set to ``False`` in non-interactive / CI environments.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Resolve string level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    if name in _configured_loggers:
        return logger

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # ----- Console handler -----
    if rich_console:
        console_handler = RichHandler(
            console=_console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        console_handler.setLevel(level)
        # Rich handles its own formatting; use minimal format string
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(console_handler)

    # ----- File handler -----
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always write DEBUG to file
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given name.

    If the logger has not been explicitly configured via :func:`setup_logger`,
    a default logger with INFO level and Rich console output is returned.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        :class:`logging.Logger` instance ready to use.
    """
    if name not in _configured_loggers:
        return setup_logger(name)
    return logging.getLogger(name)


def set_global_level(level: int | str) -> None:
    """
    Update the log level for all previously configured Emily loggers.

    Args:
        level: New log level (e.g., ``logging.DEBUG`` or ``"DEBUG"``).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    for name in _configured_loggers:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        for handler in lg.handlers:
            handler.setLevel(level)


# Module-level default logger
logger = get_logger(__name__)
