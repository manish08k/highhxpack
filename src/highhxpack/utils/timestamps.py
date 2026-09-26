"""Timestamp helpers.

All timestamps are timezone-aware UTC ``datetime`` objects in memory and are
stored as fixed-width ISO-8601 strings (``YYYY-MM-DDTHH:MM:SS.ffffffZ``) so that
lexicographic order in SQLite equals chronological order.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from highhxpack.exceptions import ValidationError

_STORAGE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_DURATION_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def utcnow() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC datetime; naive values are treated as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_storage(value: datetime) -> str:
    """Serialize a datetime to the canonical storage string."""
    return ensure_utc(value).strftime(_STORAGE_FORMAT)


def from_storage(value: str) -> datetime:
    """Parse a canonical storage string back into an aware UTC datetime."""
    return datetime.strptime(value, _STORAGE_FORMAT).replace(tzinfo=UTC)


def parse_datetime(value: str | datetime) -> datetime:
    """Parse user input (ISO-8601 string or datetime) into an aware UTC datetime."""
    if isinstance(value, datetime):
        return ensure_utc(value)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return ensure_utc(datetime.fromisoformat(text))
    except ValueError as exc:
        raise ValidationError(
            f"Invalid date/time: {value!r}",
            hint="Use ISO-8601, for example '2024-05-01' or '2024-05-01T13:45:00Z'.",
        ) from exc


def parse_duration(value: str | timedelta) -> timedelta:
    """Parse a duration such as ``"30d"``, ``"12h"`` or ``"90m"``."""
    if isinstance(value, timedelta):
        duration = value
    else:
        match = _DURATION_RE.match(value)
        if not match:
            raise ValidationError(
                f"Invalid duration: {value!r}",
                hint="Use a number followed by s, m, h, d or w, for example '30d' or '12h'.",
            )
        amount, unit = float(match.group(1)), match.group(2).lower()
        duration = timedelta(**{_DURATION_UNITS[unit]: amount})
    if duration <= timedelta(0):
        raise ValidationError(f"Duration must be positive, got {value!r}.")
    return duration


def age_in_days(then: datetime, now: datetime) -> float:
    """Return the non-negative age of ``then`` relative to ``now`` in days."""
    return max((now - then).total_seconds(), 0.0) / 86_400.0
