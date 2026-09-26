"""The core memory record."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from highhxpack.utils.timestamps import parse_datetime, to_storage


class MemoryType(StrEnum):
    """Built-in memory types.

    Memory types are open-ended strings; these constants cover common cases.
    Any lower-case identifier (for example ``"project_note"``) is also valid.
    """

    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    GOAL = "goal"
    RELATIONSHIP = "relationship"
    KNOWLEDGE = "knowledge"
    CONVERSATION = "conversation"
    DECISION = "decision"


class MemoryStatus(StrEnum):
    """Lifecycle status of a memory.

    * ``active`` - current belief, returned by recall.
    * ``superseded`` - replaced by newer, conflicting information; kept as history.
    * ``merged`` - folded into a duplicate memory during consolidation; kept as history.
    * ``archived`` - hidden from recall by an explicit user or policy action.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    MERGED = "merged"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A single stored memory.

    Instances are immutable; the API returns a new object after every change.
    """

    id: str
    user_id: str
    content: str
    memory_type: str
    source: str
    importance: float
    confidence: float
    created_at: datetime
    updated_at: datetime
    status: str = MemoryStatus.ACTIVE.value
    version: int = 1
    mention_count: int = 1
    access_count: int = 0
    last_accessed_at: datetime | None = None
    expires_at: datetime | None = None
    superseded_by: str | None = None
    embedding_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == MemoryStatus.ACTIVE

    @property
    def short_id(self) -> str:
        """The first eight characters of the id, used for display."""
        return self.id[:8]

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now

    def with_changes(self, **changes: Any) -> MemoryRecord:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source": self.source,
            "status": self.status,
            "importance": self.importance,
            "confidence": self.confidence,
            "version": self.version,
            "mention_count": self.mention_count,
            "access_count": self.access_count,
            "created_at": to_storage(self.created_at),
            "updated_at": to_storage(self.updated_at),
            "last_accessed_at": _opt_ts(self.last_accessed_at),
            "expires_at": _opt_ts(self.expires_at),
            "superseded_by": self.superseded_by,
            "embedding_model": self.embedding_model,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        """Build a memory from :meth:`to_dict` output.

        Only structural conversion happens here; semantic validation of
        imported data is performed by the import pipeline.
        """
        return cls(
            id=str(data["id"]),
            user_id=str(data["user_id"]),
            content=str(data["content"]),
            memory_type=str(data.get("memory_type", MemoryType.FACT.value)),
            source=str(data.get("source", "import")),
            status=str(data.get("status", MemoryStatus.ACTIVE.value)),
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 1.0)),
            version=int(data.get("version", 1)),
            mention_count=int(data.get("mention_count", 1)),
            access_count=int(data.get("access_count", 0)),
            created_at=parse_datetime(data["created_at"]),
            updated_at=parse_datetime(data.get("updated_at", data["created_at"])),
            last_accessed_at=_opt_parse(data.get("last_accessed_at")),
            expires_at=_opt_parse(data.get("expires_at")),
            superseded_by=data.get("superseded_by"),
            embedding_model=data.get("embedding_model"),
            metadata=dict(data.get("metadata") or {}),
        )


def _opt_ts(value: datetime | None) -> str | None:
    return to_storage(value) if value is not None else None


def _opt_parse(value: Any) -> datetime | None:
    return parse_datetime(value) if value else None
