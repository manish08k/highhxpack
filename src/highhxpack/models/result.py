"""Typed return values of the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from highhxpack.models.memory import MemoryRecord
from highhxpack.models.relationship import Conflict


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Every component that contributed to a retrieval score.

    All components are in ``[0, 1]``.  ``relevance`` combines ``keyword`` and
    ``semantic``; ``final`` is the weighted sum defined by
    :class:`highhxpack.retrieval.ranking.ScoringConfig`.
    """

    keyword: float
    semantic: float
    relevance: float
    importance: float
    recency: float
    confidence: float
    frequency: float
    final: float

    def to_dict(self) -> dict[str, float]:
        return {
            "keyword": round(self.keyword, 6),
            "semantic": round(self.semantic, 6),
            "relevance": round(self.relevance, 6),
            "importance": round(self.importance, 6),
            "recency": round(self.recency, 6),
            "confidence": round(self.confidence, 6),
            "frequency": round(self.frequency, 6),
            "final": round(self.final, 6),
        }


@dataclass(frozen=True, slots=True)
class RecallResult:
    """A memory returned by :meth:`Memory.recall` or :meth:`Memory.search`."""

    memory: MemoryRecord
    score: float
    breakdown: ScoreBreakdown
    has_conflict: bool = False

    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def content(self) -> str:
        return self.memory.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": round(self.score, 6),
            "breakdown": self.breakdown.to_dict(),
            "has_conflict": self.has_conflict,
        }


class RememberAction(StrEnum):
    """What :meth:`Memory.remember` did with the new information."""

    CREATED = "created"
    #: The content duplicated an existing memory, which was reinforced instead.
    REINFORCED = "reinforced"


@dataclass(frozen=True, slots=True)
class RememberResult:
    """Outcome of storing one memory.

    ``memory`` is the stored record: the new memory when ``action`` is
    ``created``, or the existing memory that was reinforced when the content was
    a duplicate.  ``superseded_ids`` lists older memories this one replaced as
    the current belief (they are kept as history), and ``conflicts`` lists every
    contradiction that was detected, resolved or not.
    """

    memory: MemoryRecord
    action: RememberAction
    superseded_ids: tuple[str, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def created(self) -> bool:
        return self.action == RememberAction.CREATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "memory": self.memory.to_dict(),
            "superseded_ids": list(self.superseded_ids),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """Storage-wide or per-user statistics."""

    total: int
    active: int
    users: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    relationships: int
    open_conflicts: int
    vectors: int
    missing_vectors: int
    storage_path: str
    storage_bytes: int
    schema_version: int
    embedding_model: str | None
    user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total": self.total,
            "active": self.active,
            "users": self.users,
            "by_type": dict(self.by_type),
            "by_status": dict(self.by_status),
            "relationships": self.relationships,
            "open_conflicts": self.open_conflicts,
            "vectors": self.vectors,
            "missing_vectors": self.missing_vectors,
            "storage_path": self.storage_path,
            "storage_bytes": self.storage_bytes,
            "schema_version": self.schema_version,
            "embedding_model": self.embedding_model,
        }


@dataclass(frozen=True, slots=True)
class ExportReport:
    """Summary of an export."""

    path: Path
    memories: int
    events: int
    relationships: int
    conflicts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "memories": self.memories,
            "events": self.events,
            "relationships": self.relationships,
            "conflicts": self.conflicts,
        }


@dataclass(frozen=True, slots=True)
class ImportReport:
    """Summary of an import.  ``errors`` lists records that were rejected."""

    imported: int
    replaced: int
    skipped: int
    relationships: int
    events: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "imported": self.imported,
            "replaced": self.replaced,
            "skipped": self.skipped,
            "relationships": self.relationships,
            "events": self.events,
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class ConsolidationReport:
    """What a consolidation pass changed (or would change, when ``dry_run``)."""

    user_id: str
    dry_run: bool
    merged: tuple[tuple[str, str], ...] = ()
    superseded: tuple[tuple[str, str], ...] = ()
    conflicts_detected: int = 0
    expired_archived: tuple[str, ...] = ()
    summaries_created: tuple[str, ...] = ()
    actions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return bool(
            self.merged
            or self.superseded
            or self.conflicts_detected
            or self.expired_archived
            or self.summaries_created
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "dry_run": self.dry_run,
            "merged": [list(p) for p in self.merged],
            "superseded": [list(p) for p in self.superseded],
            "conflicts_detected": self.conflicts_detected,
            "expired_archived": list(self.expired_archived),
            "summaries_created": list(self.summaries_created),
            "actions": list(self.actions),
        }
