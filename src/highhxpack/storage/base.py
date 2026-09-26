"""The storage backend interface.

:class:`StorageBackend` is the contract between the memory engine and a
persistence layer.  SQLite (:class:`~highhxpack.storage.sqlite.SQLiteStorage`)
is the default and reference implementation; another backend only needs to
implement these methods to plug into :class:`highhxpack.Memory`.

Backends store *derived* index data alongside each memory: keyword postings
(``terms``), a claim slot used for conflict detection, and an optional vector.
The engine computes that data; backends only persist and query it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from highhxpack.models.event import MemoryEvent
from highhxpack.models.memory import MemoryRecord
from highhxpack.models.relationship import Conflict, Relationship

if TYPE_CHECKING:
    from highhxpack.retrieval.filters import MemoryFilter

ListOrder = Literal["newest", "oldest", "importance", "updated"]
Direction = Literal["out", "in", "both"]


@dataclass(frozen=True, slots=True)
class IndexData:
    """Derived data stored with a memory for retrieval and conflict detection."""

    content_hash: str
    terms: dict[str, int]
    claim_slot: str | None = None
    vector: tuple[str, list[float]] | None = None  # (model name, unit vector)


@dataclass(frozen=True, slots=True)
class KeywordData:
    """Corpus statistics and postings needed to compute BM25 for a query."""

    total_documents: int
    average_length: float
    document_frequency: dict[str, int]
    # (memory_id, term, term frequency, document length)
    postings: list[tuple[str, str, int, int]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class StorageStats:
    total: int
    active: int
    users: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    relationships: int
    open_conflicts: int
    vectors: int
    storage_bytes: int
    first_memory_at: datetime | None
    last_memory_at: datetime | None


class StorageBackend(ABC):
    """Abstract persistence layer.  Implementations must be thread-safe."""

    @property
    @abstractmethod
    def location(self) -> str:
        """Human-readable location of the data (e.g. a file path)."""

    @property
    @abstractmethod
    def schema_version(self) -> int: ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources.  Calling ``close`` twice is allowed."""

    @abstractmethod
    def transaction(self) -> AbstractContextManager[None]:
        """Group several writes atomically.  Nested use joins the outer transaction."""

    # -- memories ---------------------------------------------------------
    @abstractmethod
    def insert_memory(self, memory: MemoryRecord, index: IndexData) -> None: ...

    @abstractmethod
    def update_memory(self, memory: MemoryRecord, index: IndexData | None = None) -> None:
        """Persist all fields of ``memory``; replace index data when given."""

    @abstractmethod
    def get_memory(self, memory_id: str) -> MemoryRecord | None: ...

    @abstractmethod
    def get_memories(self, memory_ids: Sequence[str]) -> dict[str, MemoryRecord]: ...

    @abstractmethod
    def resolve_prefix(self, prefix: str, *, limit: int = 2) -> list[str]:
        """Return ids starting with ``prefix`` (at most ``limit``)."""

    @abstractmethod
    def list_memories(
        self, flt: MemoryFilter, *, limit: int, offset: int = 0, order: ListOrder = "newest"
    ) -> list[MemoryRecord]: ...

    @abstractmethod
    def count_memories(self, flt: MemoryFilter) -> int: ...

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Permanently delete a memory and everything derived from it."""

    @abstractmethod
    def delete_user(self, user_id: str) -> int:
        """Permanently delete all data of one user; return the number of memories."""

    @abstractmethod
    def delete_all(self) -> int: ...

    @abstractmethod
    def find_by_hash(self, user_id: str, content_hash: str) -> list[MemoryRecord]:
        """Active memories of ``user_id`` with an identical content hash."""

    @abstractmethod
    def find_by_slot(self, user_id: str, claim_slot: str) -> list[MemoryRecord]:
        """Active memories of ``user_id`` that make a claim about ``claim_slot``."""

    @abstractmethod
    def keyword_data(self, terms: Sequence[str], flt: MemoryFilter) -> KeywordData: ...

    @abstractmethod
    def iter_vectors(self, flt: MemoryFilter, model: str) -> Iterator[tuple[str, list[float]]]:
        """Yield ``(memory_id, vector)`` for memories matching ``flt`` embedded by ``model``."""

    @abstractmethod
    def set_vector(self, memory_id: str, model: str, vector: Sequence[float]) -> None: ...

    @abstractmethod
    def ids_missing_vector(self, model: str, user_id: str | None = None) -> list[str]:
        """Ids of memories without a vector from ``model``."""

    @abstractmethod
    def record_access(self, memory_ids: Sequence[str], when: datetime) -> None: ...

    @abstractmethod
    def users(self) -> list[tuple[str, int]]:
        """``(user_id, memory count)`` pairs ordered by user id."""

    # -- history ----------------------------------------------------------
    @abstractmethod
    def add_event(self, event: MemoryEvent) -> None: ...

    @abstractmethod
    def get_events(self, memory_id: str) -> list[MemoryEvent]: ...

    @abstractmethod
    def list_events(self, user_id: str | None = None) -> list[MemoryEvent]: ...

    # -- graph ------------------------------------------------------------
    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> Relationship:
        """Insert an edge, or return the existing identical edge."""

    @abstractmethod
    def list_relationships(
        self,
        user_id: str | None,
        *,
        node: str | None = None,
        relation: str | None = None,
        direction: Direction = "both",
        memory_id: str | None = None,
    ) -> list[Relationship]: ...

    @abstractmethod
    def delete_relationship(self, relationship_id: str) -> bool: ...

    # -- conflicts --------------------------------------------------------
    @abstractmethod
    def add_conflict(self, conflict: Conflict) -> None: ...

    @abstractmethod
    def update_conflict(self, conflict: Conflict) -> None: ...

    @abstractmethod
    def get_conflict(self, conflict_id: str) -> Conflict | None: ...

    @abstractmethod
    def list_conflicts(
        self,
        user_id: str | None = None,
        *,
        status: str | None = None,
        memory_ids: Sequence[str] | None = None,
    ) -> list[Conflict]: ...

    @abstractmethod
    def stats(self, user_id: str | None = None) -> StorageStats: ...

    def __enter__(self) -> StorageBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
