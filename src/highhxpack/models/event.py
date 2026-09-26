"""Memory history events.

Every change to a memory appends an immutable :class:`MemoryEvent` recording
the memory's content and version at that point, so history is never lost when
information changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from highhxpack.utils.timestamps import to_storage


class EventKind(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    REINFORCED = "reinforced"
    SUPERSEDED = "superseded"
    MERGED = "merged"
    ARCHIVED = "archived"
    RESTORED = "restored"
    CONFLICT_DETECTED = "conflict_detected"
    IMPORTED = "imported"


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """One entry in a memory's history."""

    memory_id: str
    user_id: str
    kind: str
    version: int
    content: str
    created_at: datetime
    related_id: str | None = None
    reason: str | None = None
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "kind": self.kind,
            "version": self.version,
            "content": self.content,
            "related_id": self.related_id,
            "reason": self.reason,
            "created_at": to_storage(self.created_at),
        }
