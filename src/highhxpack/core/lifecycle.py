"""Memory lifetime: expiry (TTL) handling.

A memory may carry ``expires_at``.  Expired memories are hidden from recall,
search and listing immediately; :meth:`Memory.consolidate` archives them
(status ``archived``) so they stop occupying the active set while remaining in
history.  Nothing is deleted without an explicit ``delete``/``clear`` call.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from highhxpack.exceptions import ValidationError
from highhxpack.utils.timestamps import parse_datetime, parse_duration, utcnow

Clock = Callable[[], datetime]

#: The default clock; tests inject a fixed clock instead.
system_clock: Clock = utcnow


def compute_expiry(
    now: datetime,
    *,
    ttl: str | timedelta | None = None,
    expires_at: str | datetime | None = None,
) -> datetime | None:
    """Resolve ``ttl`` or ``expires_at`` into an absolute expiry time."""
    if ttl is not None and expires_at is not None:
        raise ValidationError("Pass either ttl or expires_at, not both.")
    if ttl is not None:
        return now + parse_duration(ttl)
    if expires_at is not None:
        value = parse_datetime(expires_at)
        if value <= now:
            raise ValidationError(
                f"expires_at is in the past: {value.isoformat()}",
                hint="Use a future time, or delete the memory instead.",
            )
        return value
    return None
