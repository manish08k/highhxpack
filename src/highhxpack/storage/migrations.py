"""Versioned SQLite schema migrations.

The schema version is stored in ``PRAGMA user_version``.  Migrations are
append-only: never edit a released migration, add a new one instead.  Each
migration runs inside its own transaction, so a failure leaves the database at
the previous version.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from highhxpack.exceptions import MigrationError


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        description="initial schema",
        statements=(
            """
            CREATE TABLE memories (
                id               TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                content          TEXT NOT NULL,
                content_hash     TEXT NOT NULL,
                memory_type      TEXT NOT NULL,
                source           TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'active',
                importance       REAL NOT NULL CHECK (importance >= 0 AND importance <= 1),
                confidence       REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                version          INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
                mention_count    INTEGER NOT NULL DEFAULT 1 CHECK (mention_count >= 1),
                access_count     INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
                term_count       INTEGER NOT NULL DEFAULT 0,
                claim_slot       TEXT,
                superseded_by    TEXT,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                last_accessed_at TEXT,
                expires_at       TEXT,
                metadata         TEXT NOT NULL DEFAULT '{}'
            )
            """,
            "CREATE INDEX idx_memories_user_status ON memories (user_id, status, created_at)",
            "CREATE INDEX idx_memories_user_hash ON memories (user_id, content_hash)",
            "CREATE INDEX idx_memories_user_slot ON memories (user_id, claim_slot) "
            "WHERE claim_slot IS NOT NULL",
            "CREATE INDEX idx_memories_type ON memories (memory_type)",
            """
            CREATE TABLE memory_terms (
                term      TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                tf        INTEGER NOT NULL,
                PRIMARY KEY (term, memory_id)
            ) WITHOUT ROWID
            """,
            "CREATE INDEX idx_terms_memory ON memory_terms (memory_id)",
            """
            CREATE TABLE memory_vectors (
                memory_id TEXT PRIMARY KEY REFERENCES memories (id) ON DELETE CASCADE,
                model     TEXT NOT NULL,
                dim       INTEGER NOT NULL,
                vector    BLOB NOT NULL
            )
            """,
            """
            CREATE TABLE memory_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id  TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                user_id    TEXT NOT NULL,
                kind       TEXT NOT NULL,
                version    INTEGER NOT NULL,
                content    TEXT NOT NULL,
                related_id TEXT,
                reason     TEXT,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX idx_events_memory ON memory_events (memory_id, id)",
            """
            CREATE TABLE relationships (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                source     TEXT NOT NULL COLLATE NOCASE,
                relation   TEXT NOT NULL,
                target     TEXT NOT NULL COLLATE NOCASE,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                memory_id  TEXT REFERENCES memories (id) ON DELETE CASCADE,
                metadata   TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE (user_id, source, relation, target)
            )
            """,
            "CREATE INDEX idx_rel_user_source ON relationships (user_id, source)",
            "CREATE INDEX idx_rel_user_target ON relationships (user_id, target)",
            "CREATE INDEX idx_rel_memory ON relationships (memory_id)",
            """
            CREATE TABLE conflicts (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                memory_id   TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                other_id    TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                kind        TEXT NOT NULL,
                status      TEXT NOT NULL,
                resolution  TEXT,
                created_at  TEXT NOT NULL,
                resolved_at TEXT
            )
            """,
            "CREATE INDEX idx_conflicts_user ON conflicts (user_id, status)",
            "CREATE INDEX idx_conflicts_memory ON conflicts (memory_id)",
            "CREATE INDEX idx_conflicts_other ON conflicts (other_id)",
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        ),
    ),
)

LATEST_VERSION = MIGRATIONS[-1].version


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Bring the schema up to :data:`LATEST_VERSION` and return applied versions.

    The connection must be in autocommit mode (``isolation_level=None``).
    """
    version = current_version(conn)
    _check_supported(version)
    applied: list[int] = []
    for migration in MIGRATIONS:
        if migration.version <= version:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Re-read under the write lock: another process may have migrated
            # the database while we were waiting for the lock.
            version = current_version(conn)
            _check_supported(version)
            if migration.version <= version:
                conn.execute("COMMIT")
                continue
            for statement in migration.statements:
                conn.execute(statement)
            # PRAGMA does not accept bound parameters; the value is an int we control.
            conn.execute(f"PRAGMA user_version = {int(migration.version)}")
            conn.execute("COMMIT")
        except BaseException as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if isinstance(exc, sqlite3.Error) and not _is_lock_error(exc):
                raise MigrationError(
                    f"Failed to apply schema migration {migration.version} "
                    f"({migration.description}).",
                    reason=str(exc),
                    hint="Back up the database file and report this issue.",
                ) from exc
            raise
        version = migration.version
        applied.append(migration.version)
    return applied


def _is_lock_error(exc: sqlite3.Error) -> bool:
    message = str(exc).lower()
    return isinstance(exc, sqlite3.OperationalError) and ("locked" in message or "busy" in message)


def _check_supported(version: int) -> None:
    if version > LATEST_VERSION:
        raise MigrationError(
            f"Database schema version {version} is newer than this version of "
            f"HighHXPack supports ({LATEST_VERSION}).",
            reason="The database was written by a newer HighHXPack release.",
            hint="Upgrade HighHXPack (`pip install -U highhxpack`) or use a different database.",
        )
