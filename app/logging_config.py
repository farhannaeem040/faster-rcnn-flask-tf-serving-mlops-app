# ──────────────────────────────────────────────────────────────
# logging_config.py  –  Sets up how and where the app writes logs
# ──────────────────────────────────────────────────────────────
# Logs help us understand what the application is doing at runtime
# and make it easy to track down bugs.
# This file configures two log outputs:
#   1. A file on disk  (so logs survive after the server restarts)
#   2. The console / terminal  (so you can watch live in Docker)

from __future__ import annotations

import logging
# RotatingFileHandler limits log file size — when it gets too big,
# it automatically starts a fresh file instead of filling the disk.
from logging.handlers import RotatingFileHandler

# Import the folder, filename, and verbosity level from our central config.
from config import LOG_FILE, LOG_LEVEL, LOGS_DIR


def configure_logging(app) -> None:
    """Attach file and console log handlers to the Flask app."""

    # Make sure the logs folder actually exists before trying to write to it.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # This is the "template" for every log line.
    # Example output:  2025-04-21 12:00:00 | INFO | app.routes | Prediction request received.
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )

    # The root logger is the parent of all other loggers in Python.
    # Setting it here means every module automatically inherits these settings.
    root_logger = logging.getLogger()
    # Apply the log level from config (e.g. INFO means we skip DEBUG messages).
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # ── File handler ──────────────────────────────────────────
    # Only add the file handler if one is not already attached.
    # (Calling configure_logging twice would otherwise duplicate every log line.)
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # ── Console handler ───────────────────────────────────────
    # Only add the console handler if one is not already attached.
    # The extra isinstance check makes sure we don't accidentally
    # count the RotatingFileHandler (which is also a StreamHandler).
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # ── Wire up Flask's own logger ────────────────────────────
    # Flask has its own internal logger (app.logger).
    # We remove its default handlers and replace them with ours
    # so all log output looks the same no matter where it comes from.
    app.logger.handlers.clear()
    for handler in root_logger.handlers:
        app.logger.addHandler(handler)

    # Match Flask's log level to the root logger.
    app.logger.setLevel(root_logger.level)

    # Stop Flask from passing its log messages up to the root logger a
    # second time (that would print every message twice).
    app.logger.propagate = False
