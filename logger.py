"""Logging configuration for Copilot AI service."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logging(log_path: str | None = None) -> logging.Logger:
    """
    Configure logging with rotation.

    Logs are written to /var/log/copilot-ai/copilot-ai.log with:
      - maxBytes: 1 MB
      - backupCount: 3

    LOG_LEVEL=DISABLE disables logging entirely.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    logger = logging.getLogger("copilot_ai")

    # Clear existing handlers to avoid duplication
    logger.handlers.clear()

    # Disable all logging if requested
    if level_name == "DISABLE":
        logging.disable(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    # Enable logging
    logging.disable(logging.NOTSET)
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    # Setup handler with fallback to console
    if not log_path:
        log_path = "/var/log/copilot-ai/copilot-ai.log"

    handler, fallback_err = _create_log_handler(log_path)

    # Setup formatter
    formatter = _create_log_formatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    if fallback_err is not None:
        logger.warning(
            "Failed to open log file %r (%s). Falling back to stdout/stderr logging.",
            log_path,
            fallback_err,
        )
    logger.propagate = False
    # NOTE: traffic logger handlers are attached in copilot_ai_service.py after config is loaded.
    return logger


def _create_log_handler(log_path: str) -> tuple[logging.Handler, Exception | None]:
    """Create log handler with fallback to StreamHandler on error."""
    try:
        return RotatingFileHandler(
            log_path,
            maxBytes=1_048_576,  # 1 MB
            backupCount=3,
            encoding="utf-8",
        ), None
    except Exception as e:
        return logging.StreamHandler(), e


def _create_log_formatter() -> logging.Formatter:
    """Create log formatter, colored if available."""
    if HAS_COLORLOG and os.getenv("LOG_COLOR", "true").lower() in ("true", "1", "yes"):
        return colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(name)s - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
    return logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")


def mask_secret(s: str, keep_start: int = 6, keep_end: int = 4) -> str:
    """Mask a secret string, keeping only start and end characters."""
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) <= keep_start + keep_end:
        return "*" * len(s)
    return f"{s[:keep_start]}...{s[-keep_end:]}"
