from __future__ import annotations

import os
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from highhxpack.exceptions import MigrationError, StorageError
from highhxpack.models.event import MemoryEvent
from highhxpack.models.memory import MemoryRecord, MemoryStatus
from highhxpack.models.relationship import Conflict, Relationship
from highhxpack.retrieval.filters import MemoryFilter, build_filter
from highhxpack.storage import vector as vec
from highhxpack.storage.base import IndexData
from highhxpack.storage.migrations import LATEST_VERSION
from highhxpack.storage.sqlite import SQLiteStorage, compile_filter
from highhxpack.utils.hashing import content_hash, new_id
from highhxpack.utils.text import term_frequencies
from support import file_mode

NOW = datetime(2025, 1, 1, tzinfo=UTC)


def record(content: str = "I use Python", user: str = "u", **changes: object) -> MemoryRecord:
    base = MemoryRecord(
        id=new_id(),
        user_id=user,
        content=content,
        memory_type="fact",
        source="api",
        importance=0.5,
        confidence=1.0,
        created_at=NOW,
        updated_at=NOW,
    )
    return base.with_changes(**changes)


def index(content: str, vector: list[float] | None = None, slot: str | None = None) -> IndexData:
    return IndexData(
        content_hash(content),
        term_frequencies(content),
        slot,
        ("test", vec.normalize(vector)) if vector else None,
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStorage:
    s = SQLiteStorage(tmp_path / "db" / "test.db")
    yield s
    s.close()


def add(store: SQLiteStorage, content: str = "I use Python", **kw: object) -> MemoryRecord:
    vector = kw.pop("vector", None)
    slot = kw.pop("slot", None)
    r = record(content, **kw)
    store.insert_memory(r, index(content, vector, slot))  # type: ignore[arg-type]
    return r


class TestSetup:
    def test_creates_schema_and_private_files(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "m.db"
        with SQLiteStorage(path) as s:
            assert s.schema_version == LATEST_VERSION
            assert s.check_integrity() == []
        if os.name == "posix":
            assert file_mode(path) == 0o600
            assert file_mode(path.parent) == 0o700

    def test_wal_mode_enabled(self, store: SQLiteStorage) -> None:
        conn = sqlite3.connect(store.location)
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()

    def test_reopen_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "m.db"
        with SQLiteStorage(path) as s:
            add(s)
        with SQLiteStorage(path) as s:
            assert s.count_memories(MemoryFilter()) == 1

    def test_directory_path_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(StorageError, match="directory"):
            SQLiteStorage(tmp_path)

    def test_corrupt_file_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.db"
        path.write_bytes(b"this is not a database" * 100)
        with pytest.raises(StorageError, match="not a usable"):
            SQLiteStorage(path)

    def test_newer_schema_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "future.db"
        conn = sqlite3.connect(path)
        conn.execute(f"PRAGMA user_version = {LATEST_VERSION + 1}")
        conn.close()
        with pytest.raises(MigrationError, match="newer"):
            SQLiteStorage(path)

    def test_closed_store_raises(self, tmp_path: Path) -> None:
        s = SQLiteStorage(tmp_path / "m.db")
        s.close()
        s.close()  # idempotent
        assert s.closed
        with pytest.raises(StorageError, match="closed"):
            s.get_memory("a" * 32)

    def test_wal_checkpointed_on_close(self, tmp_path: Path) -> None:
        path = tmp_path / "m.db"
        with SQLiteStorage(path) as s:
            add(s)
        wal = Path(str(path) + "-wal")
        assert not wal.exists() or wal.stat().st_size == 0


class TestMemories:
    def test_insert_get_update(self, store: SQLiteStorage) -> None:
        r = add(store, vector=[1.0, 0.0])
        loaded = store.get_memory(r.id)
        assert loaded is not None
        assert loaded.embedding_model == "test"
        assert loaded.with_changes(embedding_model=None) == r
        updated = r.with_changes(content="I use Rust", version=2, importance=0.9)
        store.update_memory(updated, index("I use Rust"))
        loaded = store.get_memory(r.id)
        assert loaded is not None
        assert loaded.content == "I use Rust"
        assert loaded.embedding_model is None  # vector replaced by the new index

    def test_update_missing_raises(self, store: SQLiteStorage) -> None:
        with pytest.raises(StorageError, match="does not exist"):
            store.update_memory(record())

    def test_transaction_rolls_back(self, store: SQLiteStorage) -> None:
        def failing_write() -> None:
            with store.transaction():
                add(store)
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            failing_write()
        assert store.count_memories(MemoryFilter()) == 0

    def test_nested_transactions_join(self, store: SQLiteStorage) -> None:
        with store.transaction():
            add(store)
            with store.transaction():
                add(store, "second")
        assert store.count_memories(MemoryFilter()) == 2

    def test_duplicate_id_is_storage_error(self, store: SQLiteStorage) -> None:
        r = add(store)
        with pytest.raises(StorageError):
            store.insert_memory(r, index(r.content))

    def test_list_order_and_paging(self, store: SQLiteStorage) -> None:
        for i in range(5):
            add(store, f"memory {i}", created_at=NOW + timedelta(minutes=i), importance=i / 10)
        newest = store.list_memories(MemoryFilter(), limit=2)
        assert [m.content for m in newest] == ["memory 4", "memory 3"]
        page = store.list_memories(MemoryFilter(), limit=2, offset=2, order="oldest")
        assert [m.content for m in page] == ["memory 2", "memory 3"]
        top = store.list_memories(MemoryFilter(), limit=1, order="importance")
        assert top[0].content == "memory 4"
        with pytest.raises(StorageError):
            store.list_memories(MemoryFilter(), limit=1, order="bogus")  # type: ignore[arg-type]

    def test_resolve_prefix_escapes_wildcards(self, store: SQLiteStorage) -> None:
        r = add(store)
        assert store.resolve_prefix(r.id[:6]) == [r.id]
        assert store.resolve_prefix("%") == []
        assert store.resolve_prefix("_") == []

    def test_delete_cascades(self, store: SQLiteStorage) -> None:
        a = add(store, vector=[1.0, 0.0])
        b = add(store, "other")
        store.add_event(MemoryEvent(a.id, "u", "created", 1, a.content, NOW))
        store.add_relationship(
            Relationship(new_id(), "u", "user", "uses", "Python", 1.0, NOW, a.id)
        )
        store.add_conflict(Conflict(new_id(), "u", a.id, b.id, "changed_value", "open", NOW))
        store.update_memory(b.with_changes(superseded_by=a.id))
        assert store.delete_memory(a.id)
        assert not store.delete_memory(a.id)
        assert store.get_events(a.id) == []
        assert store.list_relationships("u") == []
        assert store.list_conflicts("u") == []
        remaining = store.get_memory(b.id)
        assert remaining is not None
        assert remaining.superseded_by is None
        assert store.stats().vectors == 0

    def test_delete_user_and_all(self, store: SQLiteStorage) -> None:
        add(store, user="a")
        add(store, user="a")
        add(store, user="b")
        assert store.delete_user("a") == 2
        assert store.users() == [("b", 1)]
        assert store.delete_all() == 1
        assert store.users() == []

    def test_find_by_hash_and_slot_only_active(self, store: SQLiteStorage) -> None:
        a = add(store, "I live in Paris", slot="lives_in")
        add(store, "I live in Rome", slot="lives_in", status=MemoryStatus.SUPERSEDED.value)
        assert [m.id for m in store.find_by_hash("u", content_hash("i live in paris"))] == [a.id]
        assert [m.id for m in store.find_by_slot("u", "lives_in")] == [a.id]

    def test_record_access(self, store: SQLiteStorage) -> None:
        a = add(store)
        store.record_access([a.id], NOW + timedelta(days=1))
        store.record_access([], NOW)
        loaded = store.get_memory(a.id)
        assert loaded is not None
        assert loaded.access_count == 1
        assert loaded.last_accessed_at == NOW + timedelta(days=1)

    def test_get_memories_many(self, store: SQLiteStorage) -> None:
        ids = [add(store, f"memory number {i}").id for i in range(1_200)]
        assert set(store.get_memories(ids)) == set(ids)


class TestIndexes:
    def test_keyword_data(self, store: SQLiteStorage) -> None:
        a = add(store, "python python rust")
        add(store, "golang")
        add(store, "python", user="other")
        data = store.keyword_data(["python", "missing"], MemoryFilter(user_id="u"))
        assert data.total_documents == 2
        assert data.document_frequency == {"python": 1}
        assert data.postings == [(a.id, "python", 2, 3)]
        empty = store.keyword_data([], MemoryFilter(user_id="u"))
        assert empty.postings == []

    def test_vectors(self, store: SQLiteStorage) -> None:
        a = add(store, vector=[3.0, 4.0])
        b = add(store, "no vector")
        vectors = dict(store.iter_vectors(MemoryFilter(), "test"))
        assert vectors[a.id] == pytest.approx([0.6, 0.8])
        assert list(store.iter_vectors(MemoryFilter(), "other-model")) == []
        assert store.ids_missing_vector("test") == [b.id]
        store.set_vector(b.id, "test", [1.0, 0.0])
        assert store.ids_missing_vector("test", "u") == []
        assert set(store.ids_missing_vector("new-model")) == {a.id, b.id}


class TestGraphAndConflicts:
    def test_relationships_unique_and_case_insensitive(self, store: SQLiteStorage) -> None:
        first = store.add_relationship(
            Relationship(new_id(), "u", "user", "uses", "Python", 1.0, NOW)
        )
        again = store.add_relationship(
            Relationship(new_id(), "u", "User", "uses", "python", 1.0, NOW)
        )
        assert again.id == first.id
        assert len(store.list_relationships("u", node="PYTHON")) == 1
        assert store.list_relationships("u", node="python", direction="out") == []
        assert len(store.list_relationships("u", node="python", direction="in")) == 1
        assert len(store.list_relationships(None, relation="uses")) == 1
        assert store.delete_relationship(first.id)

    def test_conflicts(self, store: SQLiteStorage) -> None:
        a, b = add(store), add(store, "b")
        c = Conflict(new_id(), "u", a.id, b.id, "changed_value", "open", NOW)
        store.add_conflict(c)
        assert store.get_conflict(c.id) == c
        assert store.list_conflicts(status="open", memory_ids=[b.id]) == [c]
        assert store.list_conflicts(memory_ids=[]) == []
        resolved = Conflict(c.id, "u", a.id, b.id, c.kind, "resolved", NOW, "why", NOW)
        store.update_conflict(resolved)
        assert store.list_conflicts("u", status="open") == []
        assert store.stats("u").open_conflicts == 0


class TestFilters:
    def test_compile_filter_matches_reference_semantics(self, store: SQLiteStorage) -> None:
        now = NOW + timedelta(days=10)
        memories = [
            add(store, "a", memory_type="fact", metadata={"project": "x", "n": 1, "flag": True}),
            add(store, "b", memory_type="preference", metadata={"project": "y", "flag": False}),
            add(store, "c", source="cli", importance=0.9, metadata={"none": None}),
            add(store, "d", expires_at=NOW + timedelta(days=1)),
            add(store, "e", status=MemoryStatus.ARCHIVED.value, created_at=NOW + timedelta(days=2)),
            add(store, "f", user="other", confidence=0.2),
        ]
        filters = [
            MemoryFilter(),
            MemoryFilter(now=now),
            MemoryFilter(now=now, include_expired=True),
            MemoryFilter(user_id="u", statuses=()),
            MemoryFilter(memory_types=("preference",)),
            MemoryFilter(sources=("cli",)),
            MemoryFilter(min_importance=0.8),
            MemoryFilter(min_confidence=0.5),
            MemoryFilter(statuses=(), created_after=NOW + timedelta(days=1)),
            MemoryFilter(created_before=NOW + timedelta(days=1)),
            MemoryFilter(metadata={"project": "x"}),
            MemoryFilter(metadata={"n": 1}),
            MemoryFilter(metadata={"flag": True}),
            MemoryFilter(metadata={"flag": False}),
            MemoryFilter(metadata={"none": None}),
        ]
        for flt in filters:
            expected = {m.id for m in memories if flt.matches(m)}
            actual = {m.id for m in store.list_memories(flt, limit=100)}
            assert actual == expected, flt
            assert store.count_memories(flt) == len(expected)

    def test_compile_filter_binds_values(self) -> None:
        sql, params = compile_filter(MemoryFilter(user_id="x' OR 1=1 --"))
        assert "x'" not in sql
        assert params == ["x' OR 1=1 --", "active"]

    def test_build_filter_validation(self) -> None:
        from highhxpack.exceptions import ValidationError

        assert build_filter(status="any").statuses == ()
        assert build_filter(memory_type=["fact", "goal"]).memory_types == ("fact", "goal")
        with pytest.raises(ValidationError, match="status"):
            build_filter(status="deleted")
        with pytest.raises(ValidationError, match="key"):
            build_filter(metadata={'a"]': 1})
        with pytest.raises(ValidationError, match="values"):
            build_filter(metadata={"a": [1]})  # type: ignore[dict-item]


def test_failed_migration_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from highhxpack.storage import migrations

    path = tmp_path / "m.db"
    with SQLiteStorage(path) as s:
        add(s)
    broken = migrations.Migration(2, "broken", ("CREATE TABLE extra (x)", "NOT VALID SQL"))
    monkeypatch.setattr(migrations, "MIGRATIONS", (*migrations.MIGRATIONS, broken))
    monkeypatch.setattr(migrations, "LATEST_VERSION", 2)
    with pytest.raises(MigrationError, match="migration 2"):
        SQLiteStorage(path)
    monkeypatch.undo()
    with SQLiteStorage(path) as s:  # still usable at version 1, partial DDL rolled back
        assert s.schema_version == 1
        assert s.count_memories(MemoryFilter()) == 1
        conn = sqlite3.connect(path)
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        finally:
            conn.close()
        assert "extra" not in tables


def test_lock_timeout_is_reported_clearly(tmp_path: Path) -> None:
    path = tmp_path / "locked.db"
    SQLiteStorage(path).close()
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(StorageError) as info:
            SQLiteStorage(path, timeout=0.2)
        assert "Timed out" in info.value.message
        blocker.execute("COMMIT")
        store = SQLiteStorage(path, timeout=0.2)
        blocker.execute("BEGIN EXCLUSIVE")
        with pytest.raises(StorageError) as info:
            add(store)
        assert info.value.hint is not None
        assert "Another process" in info.value.hint
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    store.close()


def test_concurrent_threads(tmp_path: Path) -> None:
    store = SQLiteStorage(tmp_path / "m.db")
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(50):
                add(store, f"thread {n} memory {i}")
                store.list_memories(MemoryFilter(), limit=5)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.count_memories(MemoryFilter()) == 400
    store.close()


def test_two_connections_same_file(tmp_path: Path) -> None:
    path = tmp_path / "shared.db"
    with SQLiteStorage(path) as a, SQLiteStorage(path) as b:
        add(a, "from a")
        add(b, "from b")
        assert a.count_memories(MemoryFilter()) == 2
        assert b.count_memories(MemoryFilter()) == 2


def test_vector_helpers() -> None:
    assert vec.decode(vec.encode([1.0, 2.0]), 2) == [1.0, 2.0]
    assert vec.normalize([0.0, 0.0]) == [0.0, 0.0]
    from highhxpack.exceptions import EmbeddingError

    with pytest.raises(EmbeddingError):
        vec.decode(vec.encode([1.0]), 2)
    with pytest.raises(EmbeddingError):
        vec.normalize([float("nan")])
    ranked = vec.top_k(
        [1.0, 0.0],
        [("b", [1.0, 0.0]), ("a", [1.0, 0.0]), ("c", [0.0, 1.0])],
        k=5,
        min_similarity=0.5,
    )
    assert ranked == [("a", 1.0), ("b", 1.0)]
    assert vec.dot([1.0, 2.0], [3.0, 4.0]) == 11.0
    assert vec._dot_fallback([1.0, 2.0], [3.0, 4.0]) == 11.0  # Python 3.11 path


def test_unclosed_store_is_released_on_garbage_collection(tmp_path: Path) -> None:
    # Regression: an unclosed store used to emit "ResourceWarning: unclosed database".
    import gc
    import warnings

    from highhxpack import Memory

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        memory = Memory(tmp_path / "gc.db")
        memory.remember("u", "I use Python")
        del memory
        gc.collect()


def test_old_or_limited_sqlite_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 20, 0))
    with pytest.raises(StorageError, match="too old"):
        SQLiteStorage(tmp_path / "old.db")


def test_relationship_id_reuse_is_a_storage_error(store: SQLiteStorage) -> None:
    # Regression: this used to raise TypeError from inside the storage layer.
    rid = new_id()
    store.add_relationship(Relationship(rid, "u", "user", "uses", "Python", 1.0, NOW))
    with pytest.raises(StorageError, match="already used"):
        store.add_relationship(Relationship(rid, "other", "user", "uses", "Go", 1.0, NOW))
