"""Logging helpers.

The library logs to the ``highhxpack`` logger and installs only a
``NullHandler``; applications decide where log records go.  Secrets are never
logged: pass any value that might contain one through :func:`redact`.
"""

from __future__ import annotations

import logging
import re

LOGGER_NAME = "highhxpack"

_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_ASSIGNMENT_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization)(\s*[=:]\s*)(\S+)"
)
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+")

logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or one of its children."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def redact(text: str) -> str:
    """Mask substrings that look like API keys, bearer tokens or passwords."""
    text = _OPENAI_KEY_RE.sub("sk-***", text)
    # Bearer tokens first: otherwise "Authorization: Bearer <token>" would only
    # have the word "Bearer" masked by the assignment rule.
    text = _BEARER_RE.sub("Bearer ***", text)
    return _ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


def configure_cli_logging(verbosity: int) -> None:
    """Attach a stderr handler for CLI use. ``verbosity`` 0=warnings, 1=info, 2+=debug."""
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_highhxpack_cli", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._highhxpack_cli = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(level)
