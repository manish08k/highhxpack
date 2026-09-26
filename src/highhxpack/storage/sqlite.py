"""SQLite storage backend (the default).

Design notes:

* One connection per :class:`SQLiteStorage`, guarded by a re-entrant lock, so a
  single instance can be shared between threads.  Several processes may open
  the same file: WAL mode lets readers proceed while a writer commits, and
  ``busy_timeout`` makes writers wait instead of failing immediately.
* Every write runs inside an explicit ``BEGIN IMMEDIATE`` transaction.
* On POSIX the database directory is created with mode ``0700`` and the file
  with mode ``0600`` because memories may contain personal data.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from highhxpack.__version__ import __version__
from highhxpack.exceptions import StorageError
from highhxpack.models.event import MemoryEvent
from highhxpack.models.memory import MemoryRecord, MemoryStatus
from highhxpack.models.relationship import Conflict, ConflictStatus, Relationship
from highhxpack.storage import vector as vec
from highhxpack.storage.base import (
    Direction,
    IndexData,
    KeywordData,
    ListOrder,
    StorageBackend,
    StorageStats,
)
from highhxpack.storage.migrations import apply_migrations, current_version
from highhxpack.utils.logging import get_logger
from highhxpack.utils.timestamps import from_storage, to_storage

if TYPE_CHECKING:
    from highhxpack.retrieval.filters import MemoryFilter

logger = get_logger("storage.sqlite")

T = TypeVar("T")

MEMORY_PATH = ":memory:"
#: Oldest SQLite library version the schema and queries are tested against.
MIN_SQLITE_VERSION = (3, 32, 0)
_DEFAULT_TIMEOUT_SECONDS = 30.0

_MEMORY_COLUMNS = (
    "m.id, m.user_id, m.content, m.memory_type, m.source, m.status, m.importance, "
    "m.confidence, m.version, m.mention_count, m.access_count, m.superseded_by, "
    "m.created_at, m.updated_at, m.last_accessed_at, m.expires_at, m.metadata, v.model"
)
_MEMORY_SELECT = (
    f"SELECT {_MEMORY_COLUMNS} FROM memories m LEFT JOIN memory_vectors v ON v.memory_id = m.id"
)
_ORDER_BY: dict[str, str] = {
    "newest": "m.created_at DESC, m.id",
    "oldest": "m.created_at ASC, m.id",
    "importance": "m.importance DESC, m.created_at DESC, m.id",
    "updated": "m.updated_at DESC, m.id",
}


def _opt_ts(value: str | None) -> datetime | None:
    return from_storage(value) if value else None


def _row_to_memory(row: sqlite3.Row | tuple[Any, ...]) -> MemoryRecord:
    return MemoryRecord(
        id=row[0],
        user_id=row[1],
        content=row[2],
        memory_type=row[3],
        source=row[4],
        status=row[5],
        importance=row[6],
        confidence=row[7],
        version=row[8],
        mention_count=row[9],
        access_count=row[10],
        superseded_by=row[11],
        created_at=from_storage(row[12]),
        updated_at=from_storage(row[13]),
        last_accessed_at=_opt_ts(row[14]),
        expires_at=_opt_ts(row[15]),
        metadata=json.loads(row[16]),
        embedding_model=row[17],
    )


def _row_to_event(row: tuple[Any, ...]) -> MemoryEvent:
    return MemoryEvent(
        id=row[0],
        memory_id=row[1],
        user_id=row[2],
        kind=row[3],
        version=row[4],
        content=row[5],
        related_id=row[6],
        reason=row[7],
        created_at=from_storage(row[8]),
    )


def _row_to_relationship(row: tuple[Any, ...]) -> Relationship:
    return Relationship(
        id=row[0],
        user_id=row[1],
        source=row[2],
        relation=row[3],
        target=row[4],
        confidence=row[5],
        memory_id=row[6],
        metadata=json.loads(row[7]),
        created_at=from_storage(row[8]),
    )


def _row_to_conflict(row: tuple[Any, ...]) -> Conflict:
    return Conflict(
        id=row[0],
        user_id=row[1],
        memory_id=row[2],
        other_id=row[3],
        kind=row[4],
        status=row[5],
        resolution=row[6],
        created_at=from_storage(row[7]),
        resolved_at=_opt_ts(row[8]),
    )


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


def compile_filter(flt: MemoryFilter) -> tuple[str, list[Any]]:
    """Translate a :class:`MemoryFilter` into a SQL ``WHERE`` body over alias ``m``.

    Only placeholders are interpolated into the SQL text; every value is bound.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if flt.user_id is not None:
        clauses.append("m.user_id = ?")
        params.append(flt.user_id)
    for column, values in (
        ("m.memory_type", flt.memory_types),
        ("m.source", flt.sources),
        ("m.status", flt.statuses),
    ):
        if values:
            clauses.append(f"{column} IN ({_placeholders(len(values))})")
            params.extend(values)
    if not flt.include_expired and flt.now is not None:
        clauses.append("(m.expires_at IS NULL OR m.expires_at > ?)")
        params.append(to_storage(flt.now))
    if flt.created_after is not None:
        clauses.append("m.created_at >= ?")
        params.append(to_storage(flt.created_after))
    if flt.created_before is not None:
        clauses.append("m.created_at < ?")
        params.append(to_storage(flt.created_before))
    if flt.min_importance is not None:
        clauses.append("m.importance >= ?")
        params.append(flt.min_importance)
    if flt.min_confidence is not None:
        clauses.append("m.confidence >= ?")
        params.append(flt.min_confidence)
    for key, value in flt.metadata.items():
        # Keys are validated against [A-Za-z0-9_-] by build_filter; the JSON path
        # is still passed as a bound parameter.
        path = f'$."{key}"'
        if value is None:
            clauses.append("json_type(m.metadata, ?) = 'null'")
            params.append(path)
        elif isinstance(value, bool):
            clauses.append("json_type(m.metadata, ?) = ?")
            params.extend([path, "true" if value else "false"])
        else:
            clauses.append("json_type(m.metadata, ?) NOT IN ('true', 'false')")
            clauses.append("json_extract(m.metadata, ?) = ?")
            params.extend([path, path, value])
    return (" AND ".join(clauses) or "1 = 1"), params


class SQLiteStorage(StorageBackend):
    """SQLite-backed storage.  Pass ``":memory:"`` for a throw-away database."""

    def __init__(self, path: str | Path, *, timeout: float = _DEFAULT_TIMEOUT_SECONDS):
        self._in_memory = str(path) == MEMORY_PATH
        self._path = Path(path) if not self._in_memory else None
        self._lock = threading.RLock()
        self._depth = 0
        self._closed = False
        if self._path is not None:
            self._prepare_path(self._path)
        try:
            self._conn = sqlite3.connect(
                MEMORY_PATH if self._path is None else str(self._path),
                timeout=timeout,
                isolation_level=None,
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not open the memory database at {self.location}.",
                reason=str(exc),
                hint="Check that the directory exists and is writable.",
            ) from exc
        # Safety net for callers that never call close(): release the connection
        # when this object is garbage-collected (close() does it deterministically).
        self._finalizer = weakref.finalize(self, self._conn.close)
        try:
            self._configure(timeout)
            _retry_while_locked(lambda: apply_migrations(self._conn), timeout)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('created_with', ?)",
                (__version__,),
            )
        except sqlite3.OperationalError as exc:
            self._finalizer()
            self._closed = True
            if _is_locked(exc):
                raise StorageError(
                    f"Timed out waiting for the database at {self.location}.",
                    reason=str(exc),
                    hint="Another process is holding a write lock; retry, or increase the timeout.",
                ) from exc
            raise StorageError(
                f"Could not open the memory database at {self.location}.",
                reason=str(exc),
                hint="Check the file and directory permissions and free disk space.",
            ) from exc
        except sqlite3.DatabaseError as exc:
            self._finalizer()
            self._closed = True
            raise StorageError(
                f"{self.location} is not a usable HighHXPack database.",
                reason=str(exc),
                hint="The file may be corrupt or not a SQLite database. Restore it from a "
                "backup or point `storage_path` at a new file.",
            ) from exc
        except BaseException:
            self._finalizer()
            self._closed = True
            raise

    # -- setup ------------------------------------------------------------
    @staticmethod
    def _prepare_path(path: Path) -> None:
        if path.exists() and path.is_dir():
            raise StorageError(
                f"Storage path {path} is a directory.",
                hint="Point storage_path at a file such as ~/memories/memory.db.",
            )
        parent = path.parent
        try:
            if not parent.exists():
                parent.mkdir(parents=True, mode=0o700)
            if not path.exists():
                # Create the file ourselves so it is never readable by other users,
                # not even briefly.  SQLite gives its -wal/-shm files the same mode.
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
        except FileExistsError:  # pragma: no cover - created concurrently by another process
            pass
        except OSError as exc:
            raise StorageError(
                f"Could not create the memory database at {path}.",
                reason=str(exc),
                hint="Create the directory manually or choose a writable storage_path.",
            ) from exc

    def _configure(self, timeout: float) -> None:
        conn = self._conn
        _check_sqlite_features(conn)
        # Fail fast with a clear error if the file is not a SQLite database.
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        if not self._in_memory:
            # Switching to WAL needs an exclusive lock and SQLite may report "locked"
            # without waiting, e.g. while other processes create the same database.
            mode = _retry_while_locked(
                lambda: conn.execute("PRAGMA journal_mode = WAL").fetchone()[0], timeout
            )
            if str(mode).lower() != "wal":  # pragma: no cover - e.g. network filesystems
                logger.warning("WAL mode unavailable (journal_mode=%s); using default", mode)
            conn.execute("PRAGMA synchronous = NORMAL")

    # -- infrastructure ---------------------------------------------------
    @property
    def location(self) -> str:
        return MEMORY_PATH if self._path is None else str(self._path)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def schema_version(self) -> int:
        with self._guard():
            return current_version(self._conn)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if not self._in_memory:
                    # Fold the WAL back into the main file for a clean shutdown.
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc:  # pragma: no cover - best effort
                logger.warning("WAL checkpoint on close failed: %s", exc)
            finally:
                self._finalizer()

    @property
    def closed(self) -> bool:
        return self._closed

    def check_integrity(self) -> list[str]:
        """Run ``PRAGMA integrity_check``; an empty list means the database is healthy."""
        with self._guard():
            rows = self._conn.execute("PRAGMA integrity_check").fetchall()
        problems = [str(r[0]) for r in rows]
        return [] if problems == ["ok"] else problems

    @contextmanager
    def _guard(self) -> Iterator[sqlite3.Connection]:
        """Serialize access and translate sqlite errors into :class:`StorageError`."""
        with self._lock:
            if self._closed:
                raise StorageError(
                    "The memory store is closed.",
                    hint="Create a new Memory() instance.",
                )
            try:
                yield self._conn
            except sqlite3.OperationalError as exc:
                message = str(exc)
                hint = None
                if "locked" in message or "busy" in message:
                    hint = "Another process is writing to the database; retry shortly."
                elif "readonly" in message:
                    hint = "Check the file permissions of the database and its directory."
                elif "disk" in message or "full" in message:
                    hint = "Free disk space and retry."
                raise StorageError("Database operation failed.", reason=message, hint=hint) from exc
            except sqlite3.DatabaseError as exc:
                raise StorageError(
                    "Database error.",
                    reason=str(exc),
                    hint="The database may be corrupt; run `highhxpack stats` and restore "
                    "from an export if needed.",
                ) from exc

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._guard() as conn:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self._depth = 0
                conn.execute("ROLLBACK")
                raise
            self._depth = 0
            conn.execute("COMMIT")

    # -- memories ---------------------------------------------------------
    def insert_memory(self, memory: MemoryRecord, index: IndexData) -> None:
        with self.transaction(), self._guard() as conn:
            conn.execute(
                "INSERT INTO memories (id, user_id, content, content_hash, memory_type, source, "
                "status, importance, confidence, version, mention_count, access_count, "
                "term_count, claim_slot, superseded_by, created_at, updated_at, "
                "last_accessed_at, expires_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.user_id,
                    memory.content,
                    index.content_hash,
                    memory.memory_type,
                    memory.source,
                    memory.status,
                    memory.importance,
                    memory.confidence,
                    memory.version,
                    memory.mention_count,
                    memory.access_count,
                    sum(index.terms.values()),
                    index.claim_slot,
                    memory.superseded_by,
                    to_storage(memory.created_at),
                    to_storage(memory.updated_at),
                    to_storage(memory.last_accessed_at) if memory.last_accessed_at else None,
                    to_storage(memory.expires_at) if memory.expires_at else None,
                    json.dumps(memory.metadata, sort_keys=True),
                ),
            )
            self._write_index(conn, memory.id, index)

    def update_memory(self, memory: MemoryRecord, index: IndexData | None = None) -> None:
        with self.transaction(), self._guard() as conn:
            cursor = conn.execute(
                "UPDATE memories SET content = ?, memory_type = ?, source = ?, status = ?, "
                "importance = ?, confidence = ?, version = ?, mention_count = ?, "
                "access_count = ?, superseded_by = ?, updated_at = ?, last_accessed_at = ?, "
                "expires_at = ?, metadata = ? WHERE id = ?",
                (
                    memory.content,
                    memory.memory_type,
                    memory.source,
                    memory.status,
                    memory.importance,
                    memory.confidence,
                    memory.version,
                    memory.mention_count,
                    memory.access_count,
                    memory.superseded_by,
                    to_storage(memory.updated_at),
                    to_storage(memory.last_accessed_at) if memory.last_accessed_at else None,
                    to_storage(memory.expires_at) if memory.expires_at else None,
                    json.dumps(memory.metadata, sort_keys=True),
                    memory.id,
                ),
            )
            if cursor.rowcount == 0:
                raise StorageError(f"Cannot update memory {memory.id!r}: it does not exist.")
            if index is not None:
                conn.execute(
                    "UPDATE memories SET content_hash = ?, term_count = ?, claim_slot = ? "
                    "WHERE id = ?",
                    (index.content_hash, sum(index.terms.values()), index.claim_slot, memory.id),
                )
                conn.execute("DELETE FROM memory_terms WHERE memory_id = ?", (memory.id,))
                conn.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory.id,))
                self._write_index(conn, memory.id, index)

    @staticmethod
    def _write_index(conn: sqlite3.Connection, memory_id: str, index: IndexData) -> None:
        if index.terms:
            conn.executemany(
                "INSERT INTO memory_terms (term, memory_id, tf) VALUES (?, ?, ?)",
                [(term, memory_id, tf) for term, tf in sorted(index.terms.items())],
            )
        if index.vector is not None:
            model, values = index.vector
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (memory_id, model, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                (memory_id, model, len(values), vec.encode(values)),
            )

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self._guard() as conn:
            row = conn.execute(f"{_MEMORY_SELECT} WHERE m.id = ?", (memory_id,)).fetchone()
        return _row_to_memory(row) if row else None

    def get_memories(self, memory_ids: Sequence[str]) -> dict[str, MemoryRecord]:
        result: dict[str, MemoryRecord] = {}
        ids = list(dict.fromkeys(memory_ids))
        chunk = 500
        with self._guard() as conn:
            for start in range(0, len(ids), chunk):
                part = ids[start : start + chunk]
                rows = conn.execute(
                    f"{_MEMORY_SELECT} WHERE m.id IN ({_placeholders(len(part))})", part
                ).fetchall()
                for row in rows:
                    memory = _row_to_memory(row)
                    result[memory.id] = memory
        return result

    def resolve_prefix(self, prefix: str, *, limit: int = 2) -> list[str]:
        # Prefixes are validated as hex by the caller; escape LIKE wildcards anyway.
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._guard() as conn:
            rows = conn.execute(
                "SELECT id FROM memories WHERE id LIKE ? ESCAPE '\\' ORDER BY id LIMIT ?",
                (escaped + "%", limit),
            ).fetchall()
        return [r[0] for r in rows]

    def list_memories(
        self, flt: MemoryFilter, *, limit: int, offset: int = 0, order: ListOrder = "newest"
    ) -> list[MemoryRecord]:
        where, params = compile_filter(flt)
        order_sql = _ORDER_BY.get(order)
        if order_sql is None:
            raise StorageError(f"Unknown list order {order!r}.")
        with self._guard() as conn:
            rows = conn.execute(
                f"{_MEMORY_SELECT} WHERE {where} ORDER BY {order_sql} LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def count_memories(self, flt: MemoryFilter) -> int:
        where, params = compile_filter(flt)
        with self._guard() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM memories m WHERE {where}", params).fetchone()
        return int(row[0])

    def delete_memory(self, memory_id: str) -> bool:
        with self.transaction(), self._guard() as conn:
            # Conflicts reference two memories; cascades handle both columns.
            conn.execute(
                "UPDATE memories SET superseded_by = NULL WHERE superseded_by = ?", (memory_id,)
            )
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def delete_user(self, user_id: str) -> int:
        with self.transaction(), self._guard() as conn:
            conn.execute("DELETE FROM relationships WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM conflicts WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM memory_events WHERE user_id = ?", (user_id,))
            cursor = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            return cursor.rowcount

    def delete_all(self) -> int:
        with self.transaction(), self._guard() as conn:
            for table in ("relationships", "conflicts", "memory_events"):
                conn.execute(f"DELETE FROM {table}")
            cursor = conn.execute("DELETE FROM memories")
            return cursor.rowcount

    def find_by_hash(self, user_id: str, content_hash: str) -> list[MemoryRecord]:
        with self._guard() as conn:
            rows = conn.execute(
                f"{_MEMORY_SELECT} WHERE m.user_id = ? AND m.content_hash = ? AND m.status = ? "
                "ORDER BY m.created_at, m.id",
                (user_id, content_hash, MemoryStatus.ACTIVE.value),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def find_by_slot(self, user_id: str, claim_slot: str) -> list[MemoryRecord]:
        with self._guard() as conn:
            rows = conn.execute(
                f"{_MEMORY_SELECT} WHERE m.user_id = ? AND m.claim_slot = ? AND m.status = ? "
                "ORDER BY m.created_at, m.id",
                (user_id, claim_slot, MemoryStatus.ACTIVE.value),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def keyword_data(self, terms: Sequence[str], flt: MemoryFilter) -> KeywordData:
        unique = sorted(set(terms))
        where, params = compile_filter(flt)
        with self._guard() as conn:
            total, avg_len = conn.execute(
                f"SELECT COUNT(*), AVG(m.term_count) FROM memories m WHERE {where}", params
            ).fetchone()
            if not unique or not total:
                return KeywordData(int(total or 0), float(avg_len or 0.0), {}, [])
            in_terms = _placeholders(len(unique))
            rows = conn.execute(
                "SELECT t.memory_id, t.term, t.tf, m.term_count FROM memory_terms t "
                "JOIN memories m ON m.id = t.memory_id "
                f"WHERE t.term IN ({in_terms}) AND {where}",
                [*unique, *params],
            ).fetchall()
        document_frequency: dict[str, int] = {}
        for _, term, _, _ in rows:
            document_frequency[term] = document_frequency.get(term, 0) + 1
        return KeywordData(
            total_documents=int(total),
            average_length=float(avg_len or 0.0),
            document_frequency=document_frequency,
            postings=[(r[0], r[1], int(r[2]), int(r[3])) for r in rows],
        )

    def iter_vectors(self, flt: MemoryFilter, model: str) -> Iterator[tuple[str, list[float]]]:
        where, params = compile_filter(flt)
        with self._guard() as conn:
            rows = conn.execute(
                "SELECT v.memory_id, v.dim, v.vector FROM memory_vectors v "
                f"JOIN memories m ON m.id = v.memory_id WHERE v.model = ? AND {where}",
                [model, *params],
            ).fetchall()
        for memory_id, dim, blob in rows:
            yield memory_id, vec.decode(blob, dim)

    def set_vector(self, memory_id: str, model: str, vector: Sequence[float]) -> None:
        with self.transaction(), self._guard() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (memory_id, model, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                (memory_id, model, len(vector), vec.encode(vector)),
            )

    def ids_missing_vector(self, model: str, user_id: str | None = None) -> list[str]:
        sql = (
            "SELECT m.id FROM memories m LEFT JOIN memory_vectors v ON v.memory_id = m.id "
            "WHERE (v.memory_id IS NULL OR v.model != ?)"
        )
        params: list[Any] = [model]
        if user_id is not None:
            sql += " AND m.user_id = ?"
            params.append(user_id)
        with self._guard() as conn:
            rows = conn.execute(sql + " ORDER BY m.created_at, m.id", params).fetchall()
        return [r[0] for r in rows]

    def record_access(self, memory_ids: Sequence[str], when: datetime) -> None:
        if not memory_ids:
            return
        with self.transaction(), self._guard() as conn:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
                f"WHERE id IN ({_placeholders(len(memory_ids))})",
                [to_storage(when), *memory_ids],
            )

    def users(self) -> list[tuple[str, int]]:
        with self._guard() as conn:
            rows = conn.execute(
                "SELECT user_id, COUNT(*) FROM memories GROUP BY user_id ORDER BY user_id"
            ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    # -- history ----------------------------------------------------------
    def add_event(self, event: MemoryEvent) -> None:
        with self.transaction(), self._guard() as conn:
            conn.execute(
                "INSERT INTO memory_events (memory_id, user_id, kind, version, content, "
                "related_id, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.memory_id,
                    event.user_id,
                    event.kind,
                    event.version,
                    event.content,
                    event.related_id,
                    event.reason,
                    to_storage(event.created_at),
                ),
            )

    _EVENT_SELECT = (
        "SELECT id, memory_id, user_id, kind, version, content, related_id, reason, created_at "
        "FROM memory_events"
    )

    def get_events(self, memory_id: str) -> list[MemoryEvent]:
        with self._guard() as conn:
            rows = conn.execute(
                f"{self._EVENT_SELECT} WHERE memory_id = ? ORDER BY id", (memory_id,)
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def list_events(self, user_id: str | None = None) -> list[MemoryEvent]:
        with self._guard() as conn:
            if user_id is None:
                rows = conn.execute(f"{self._EVENT_SELECT} ORDER BY id").fetchall()
            else:
                rows = conn.execute(
                    f"{self._EVENT_SELECT} WHERE user_id = ? ORDER BY id", (user_id,)
                ).fetchall()
        return [_row_to_event(r) for r in rows]

    # -- graph ------------------------------------------------------------
    _REL_SELECT = (
        "SELECT id, user_id, source, relation, target, confidence, memory_id, metadata, "
        "created_at FROM relationships"
    )

    def add_relationship(self, relationship: Relationship) -> Relationship:
        with self.transaction(), self._guard() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO relationships (id, user_id, source, relation, target, "
                "confidence, memory_id, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    relationship.id,
                    relationship.user_id,
                    relationship.source,
                    relationship.relation,
                    relationship.target,
                    relationship.confidence,
                    relationship.memory_id,
                    json.dumps(relationship.metadata, sort_keys=True),
                    to_storage(relationship.created_at),
                ),
            )
            row = conn.execute(
                f"{self._REL_SELECT} WHERE user_id = ? AND source = ? AND relation = ? "
                "AND target = ?",
                (
                    relationship.user_id,
                    relationship.source,
                    relationship.relation,
                    relationship.target,
                ),
            ).fetchone()
        if row is None:
            raise StorageError(
                f"Relationship id {relationship.id!r} is already used by a different edge.",
                hint="Use a new id for the relationship.",
            )
        return _row_to_relationship(row)

    def list_relationships(
        self,
        user_id: str | None,
        *,
        node: str | None = None,
        relation: str | None = None,
        direction: Direction = "both",
        memory_id: str | None = None,
    ) -> list[Relationship]:
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if node is not None:
            if direction == "out":
                clauses.append("source = ?")
                params.append(node)
            elif direction == "in":
                clauses.append("target = ?")
                params.append(node)
            else:
                clauses.append("(source = ? OR target = ?)")
                params.extend([node, node])
        if relation is not None:
            clauses.append("relation = ?")
            params.append(relation)
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        where = " AND ".join(clauses) or "1 = 1"
        with self._guard() as conn:
            rows = conn.execute(
                f"{self._REL_SELECT} WHERE {where} ORDER BY source, relation, target, id", params
            ).fetchall()
        return [_row_to_relationship(r) for r in rows]

    def delete_relationship(self, relationship_id: str) -> bool:
        with self.transaction(), self._guard() as conn:
            cursor = conn.execute("DELETE FROM relationships WHERE id = ?", (relationship_id,))
            return cursor.rowcount > 0

    # -- conflicts --------------------------------------------------------
    _CONFLICT_SELECT = (
        "SELECT id, user_id, memory_id, other_id, kind, status, resolution, created_at, "
        "resolved_at FROM conflicts"
    )

    def add_conflict(self, conflict: Conflict) -> None:
        with self.transaction(), self._guard() as conn:
            conn.execute(
                "INSERT INTO conflicts (id, user_id, memory_id, other_id, kind, status, "
                "resolution, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conflict.id,
                    conflict.user_id,
                    conflict.memory_id,
                    conflict.other_id,
                    conflict.kind,
                    conflict.status,
                    conflict.resolution,
                    to_storage(conflict.created_at),
                    to_storage(conflict.resolved_at) if conflict.resolved_at else None,
                ),
            )

    def update_conflict(self, conflict: Conflict) -> None:
        with self.transaction(), self._guard() as conn:
            conn.execute(
                "UPDATE conflicts SET status = ?, resolution = ?, resolved_at = ? WHERE id = ?",
                (
                    conflict.status,
                    conflict.resolution,
                    to_storage(conflict.resolved_at) if conflict.resolved_at else None,
                    conflict.id,
                ),
            )

    def get_conflict(self, conflict_id: str) -> Conflict | None:
        with self._guard() as conn:
            row = conn.execute(f"{self._CONFLICT_SELECT} WHERE id = ?", (conflict_id,)).fetchone()
        return _row_to_conflict(row) if row else None

    def list_conflicts(
        self,
        user_id: str | None = None,
        *,
        status: str | None = None,
        memory_ids: Sequence[str] | None = None,
    ) -> list[Conflict]:
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if memory_ids is not None:
            if not memory_ids:
                return []
            marks = _placeholders(len(memory_ids))
            clauses.append(f"(memory_id IN ({marks}) OR other_id IN ({marks}))")
            params.extend([*memory_ids, *memory_ids])
        where = " AND ".join(clauses) or "1 = 1"
        with self._guard() as conn:
            rows = conn.execute(
                f"{self._CONFLICT_SELECT} WHERE {where} ORDER BY created_at, id", params
            ).fetchall()
        return [_row_to_conflict(r) for r in rows]

    # -- stats ------------------------------------------------------------
    def stats(self, user_id: str | None = None) -> StorageStats:
        where, params = ("m.user_id = ?", [user_id]) if user_id is not None else ("1 = 1", [])
        rel_where = "WHERE user_id = ?" if user_id is not None else ""
        with self._guard() as conn:
            by_status = dict(
                conn.execute(
                    f"SELECT m.status, COUNT(*) FROM memories m WHERE {where} GROUP BY m.status",
                    params,
                ).fetchall()
            )
            by_type = dict(
                conn.execute(
                    f"SELECT m.memory_type, COUNT(*) FROM memories m WHERE {where} "
                    "GROUP BY m.memory_type ORDER BY m.memory_type",
                    params,
                ).fetchall()
            )
            users, first, last = conn.execute(
                "SELECT COUNT(DISTINCT m.user_id), MIN(m.created_at), MAX(m.created_at) "
                f"FROM memories m WHERE {where}",
                params,
            ).fetchone()
            relationships = conn.execute(
                f"SELECT COUNT(*) FROM relationships {rel_where}", params
            ).fetchone()[0]
            conflict_params = [*params, ConflictStatus.OPEN.value]
            open_conflicts = conn.execute(
                "SELECT COUNT(*) FROM conflicts WHERE "
                + ("user_id = ? AND " if user_id is not None else "")
                + "status = ?",
                conflict_params,
            ).fetchone()[0]
            vectors = conn.execute(
                "SELECT COUNT(*) FROM memory_vectors v JOIN memories m ON m.id = v.memory_id "
                f"WHERE {where}",
                params,
            ).fetchone()[0]
        return StorageStats(
            total=sum(by_status.values()),
            active=int(by_status.get(MemoryStatus.ACTIVE.value, 0)),
            users=int(users),
            by_type={str(k): int(v) for k, v in by_type.items()},
            by_status={str(k): int(v) for k, v in sorted(by_status.items())},
            relationships=int(relationships),
            open_conflicts=int(open_conflicts),
            vectors=int(vectors),
            storage_bytes=self._size_on_disk(),
            first_memory_at=_opt_ts(first),
            last_memory_at=_opt_ts(last),
        )

    def _size_on_disk(self) -> int:
        if self._path is None:
            return 0
        total = 0
        for suffix in ("", "-wal"):
            candidate = Path(str(self._path) + suffix)
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
        return total


def _check_sqlite_features(conn: sqlite3.Connection) -> None:
    """Refuse to run on SQLite builds that lack features the schema relies on."""
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        raise StorageError(
            f"SQLite {sqlite3.sqlite_version} is too old; HighHXPack needs "
            f"{'.'.join(map(str, MIN_SQLITE_VERSION))} or newer.",
            hint="Use a Python build that bundles a newer SQLite (e.g. from python.org).",
        )
    try:
        conn.execute("SELECT json_type('{}')").fetchone()
    except sqlite3.OperationalError as exc:
        raise StorageError(
            "This SQLite build lacks the JSON functions HighHXPack needs.",
            reason=str(exc),
            hint="Use a Python build whose SQLite includes JSON1 (standard since SQLite 3.38).",
        ) from exc


def _is_locked(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _retry_while_locked(operation: Callable[[], T], timeout: float) -> T:
    """Run ``operation``, retrying with backoff while the database is locked."""
    deadline = time.monotonic() + timeout
    delay = 0.005
    while True:
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if not _is_locked(exc) or time.monotonic() + delay > deadline:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 0.25)
