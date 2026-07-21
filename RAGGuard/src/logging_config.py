"""Structured logging configuration for RAGGuard.

Usage::

    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Processing case %s", case_id)
    logger.debug("Tool result: %s", tool_output)

Call ``setup_logging()`` once at application startup to configure handlers.
"""

import logging
import os
import sys
from datetime import datetime


# Module-level logger cache
_loggers: dict = {}

# Track whether setup has been called
_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: str = "outputs/ragguard.log",
    console: bool = True,
) -> None:
    """Configure the root RAGGuard logger with console and file handlers.

    Args:
        level: Minimum log level for file handler (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. Created if it doesn't exist.
        console: Whether to also emit INFO+ to stderr.

    Called once at application startup. Idempotent — subsequent calls are no-ops.
    """
    global _initialized
    if _initialized:
        return

    root = logging.getLogger("ragguard")
    root.setLevel(logging.DEBUG)  # Root allows all; handlers filter

    # Avoid duplicate handlers if called multiple times
    if root.handlers:
        _initialized = True
        return

    # ── Formatter ──────────────────────────────────────────
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── File handler (DEBUG+) ──────────────────────────────
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(_level_str_to_int(level))
    fh.setFormatter(file_fmt)
    root.addHandler(fh)

    # ── Console handler (INFO+) ────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        ch.setFormatter(console_fmt)
        root.addHandler(ch)

    _initialized = True

    # Banner
    root.info("RAGGuard logging initialized (level=%s, file=%s)", level, log_file)


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger under the 'ragguard' hierarchy.

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        A logger that inherits the handlers configured by ``setup_logging``.
    """
    if name in _loggers:
        return _loggers[name]

    # Nest under 'ragguard.' so it inherits handlers from setup_logging
    full_name = f"ragguard.{name.split('.')[-1]}" if not name.startswith("ragguard") else name
    logger = logging.getLogger(full_name)
    _loggers[name] = logger
    return logger


def _level_str_to_int(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)
