"""Test helpers shared by unit and integration tests."""

from __future__ import annotations

import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path


class FakeClock:
    """A controllable UTC clock that advances one millisecond per call.

    Auto-advancing keeps timestamps strictly increasing (like a real clock)
    while staying fully deterministic.
    """

    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2025, 1, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(milliseconds=1)
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


def file_mode(path: Path) -> int:
    """Permission bits of ``path`` (POSIX)."""
    return stat.S_IMODE(path.stat().st_mode)
