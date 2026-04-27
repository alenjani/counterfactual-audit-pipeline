"""Logging via loguru — colored, structured, GCS-friendly."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

_configured = False


def get_logger(log_file: str | Path | None = None):
    global _configured
    if not _configured:
        _logger.remove()
        _logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
        )
        if log_file is not None:
            _logger.add(log_file, rotation="100 MB", level="DEBUG", enqueue=True)
        _configured = True
    return _logger
