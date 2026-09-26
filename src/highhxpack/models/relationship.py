"""Knowledge-graph relationships and memory conflicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from highhxpack.utils.timestamps import to_storage


@dataclass(frozen=True, slots=True)
class Relationship:
    """A directed, labelled edge such as ``user --prefers--> Python``.

    ``memory_id`` records the memory the relationship was derived from, if any.
    """

    id: str
    user_id: str
    source: str
    relation: str
    target: str
    confidence: float
    created_at: datetime
    memory_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "confidence": self.confidence,
            "memory_id": self.memory_id,
            "metadata": dict(self.metadata),
            "created_at": to_storage(self.created_at),
        }


class ConflictStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class Conflict:
    """Two memories that make incompatible claims.

    ``memory_id`` is the newer memory and ``other_id`` the older one.  A conflict
    is ``resolved`` when a policy or the user chose which one is current;
    ``resolution`` explains how.
    """

    id: str
    user_id: str
    memory_id: str
    other_id: str
    kind: str
    status: str
    created_at: datetime
    resolution: str | None = None
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_id": self.memory_id,
            "other_id": self.other_id,
            "kind": self.kind,
            "status": self.status,
            "resolution": self.resolution,
            "created_at": to_storage(self.created_at),
            "resolved_at": to_storage(self.resolved_at) if self.resolved_at else None,
        }
