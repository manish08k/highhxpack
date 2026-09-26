from __future__ import annotations

from pathlib import Path

import pytest

from highhxpack import (
    CallableEmbedding,
    CallableProvider,
    Config,
    ConfigurationError,
    ConflictNotFoundError,
    EmbeddingError,
    EmbeddingProvider,
    HighHXPackError,
    Memory,
    MemoryNotFoundError,
    MemoryStatus,
    RememberAction,
    ValidationError,
)
from highhxpack.consolidation import ExtractiveSummarizer
from support import FakeClock


class TestRemember:
    def test_spec_quickstart(self, memory: Memory) -> None:
        result = memory.remember(user_id="user_123", content="I prefer Python for AI development.")
        assert result.created
        assert result.memory.memory_type == "preference"
        results = memory.recall(user_id="user_123", query="What language do I prefer for AI?")
        assert results[0].id == result.id

    def test_defaults_and_options(self, memory: Memory, clock: FakeClock) -> None:
        result = memory.remember(
            "u",
            "  Deploys happen on Fridays  ",
            memory_type="project_note",
            source="slack:ops",
            importance=0.9,
            confidence=0.8,
            metadata={"team": "ops"},
            ttl="7d",
        )
        m = result.memory
        assert m.content == "Deploys happen on Fridays"
        assert (m.memory_type, m.source, m.importance, m.confidence) == (
            "project_note",
            "slack:ops",
            0.9,
            0.8,
        )
        assert m.metadata == {"team": "ops"}
        assert m.expires_at is not None
        assert m.embedding_model == "hashing-v1-384"
        assert m.version == 1
        assert memory.history(m.id)[0].kind == "created"

    def test_duplicates_reinforce(self, memory: Memory) -> None:
        first = memory.remember("u", "User likes Python.")
        second = memory.remember("u", "User likes Python.")
        third = memory.remember("u", "User prefers Python.")
        assert second.action == RememberAction.REINFORCED
        assert third.action == RememberAction.REINFORCED
        assert second.id == third.id == first.id
        assert memory.count("u") == 1
        stored = memory.get(first.id)
        assert stored.mention_count == 3
        assert stored.importance > first.memory.importance
        kinds = [e.kind for e in memory.history(first.id)]
        assert kinds == ["created", "reinforced", "reinforced"]

    def test_dedupe_is_per_user_and_optional(self, memory: Memory) -> None:
        memory.remember("a", "I use Python")
        assert memory.remember("b", "I use Python").created
        assert memory.remember("a", "I use Python", dedupe=False).created
        assert memory.count("a") == 2

    def test_archived_memory_is_not_reinforced(self, memory: Memory) -> None:
        first = memory.remember("u", "I use Python")
        memory.forget(first.id)
        assert memory.remember("u", "I use Python").created

    def test_conflict_supersedes_and_keeps_history(self, memory: Memory) -> None:
        java = memory.remember("u", "I prefer Java")
        python = memory.remember("u", "I prefer Python")
        assert python.superseded_ids == (java.id,)
        [conflict] = python.conflicts
        assert conflict.status == "resolved"
        assert conflict.resolution is not None
        assert "policy: supersede" in conflict.resolution
        old = memory.get(java.id)
        assert old.status == MemoryStatus.SUPERSEDED
        assert old.superseded_by == python.id
        assert [r.id for r in memory.recall("u", "prefer")] == [python.id]
        history = memory.search("prefer", user_id="u", status="any")
        assert {r.id for r in history} == {java.id, python.id}
        assert memory.history(java.id)[-1].kind == "superseded"

    def test_lower_confidence_does_not_supersede(self, memory: Memory) -> None:
        sure = memory.remember("u", "I live in Paris", confidence=1.0)
        unsure = memory.remember("u", "I live in Berlin", confidence=0.4)
        assert unsure.superseded_ids == ()
        assert unsure.conflicts[0].status == "open"
        assert memory.get(sure.id).is_active
        results = memory.recall("u", "where do I live")
        assert all(r.has_conflict for r in results)
        assert len(memory.conflicts("u")) == 1

    def test_keep_both_policy(self, tmp_path: Path, clock: FakeClock) -> None:
        config = Config(storage_path=tmp_path / "k.db", conflict_policy="keep_both")
        with Memory(config=config, clock=clock) as memory:
            memory.remember("u", "I prefer Java")
            result = memory.remember("u", "I prefer Python")
            assert result.superseded_ids == ()
            assert memory.count("u") == 2
            assert memory.history(result.id)[-1].kind == "conflict_detected"

    def test_detect_conflicts_can_be_disabled(self, memory: Memory) -> None:
        memory.remember("u", "I prefer Java")
        result = memory.remember("u", "I prefer Python", detect_conflicts=False)
        assert result.conflicts == ()
        assert memory.count("u") == 2

    def test_resolve_conflict(self, memory: Memory) -> None:
        paris = memory.remember("u", "I live in Paris")
        berlin = memory.remember("u", "I live in Berlin", confidence=0.5)
        [conflict] = memory.conflicts("u")
        resolved = memory.resolve_conflict(conflict.id, keep=berlin.id)
        assert resolved.status == "resolved"
        assert memory.get(paris.id).status == "superseded"
        assert memory.get(berlin.id).is_active
        assert memory.conflicts("u") == []
        assert len(memory.conflicts("u", status=None)) == 1
        # Reverse the decision: the superseded memory is restored.
        memory.resolve_conflict(conflict.id, keep=paris.id)
        assert memory.get(paris.id).is_active
        assert memory.get(berlin.id).status == "superseded"
        with pytest.raises(ValidationError):
            memory.resolve_conflict(conflict.id, keep="f" * 32)
        with pytest.raises(ConflictNotFoundError):
            memory.resolve_conflict("e" * 32, keep=paris.id)
        with pytest.raises(ValidationError):
            memory.conflicts(status="weird")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"user_id": "", "content": "x"}, "user_id"),
            ({"user_id": "u", "content": ""}, "content"),
            ({"user_id": "u", "content": "x", "importance": 1.5}, "importance"),
            ({"user_id": "u", "content": "x", "confidence": -1}, "confidence"),
            ({"user_id": "u", "content": "x", "memory_type": "Bad"}, "memory_type"),
            ({"user_id": "u", "content": "x", "source": "a b"}, "source"),
            ({"user_id": "u", "content": "x", "metadata": {"a": object()}}, "JSON"),
            ({"user_id": "u", "content": "x", "ttl": "forever"}, "duration"),
            ({"user_id": "u", "content": "x", "ttl": "1d", "expires_at": "2099-01-01"}, "either"),
            ({"user_id": "u", "content": "x", "expires_at": "2000-01-01"}, "past"),
        ],
    )
    def test_invalid_input(self, memory: Memory, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            memory.remember(**kwargs)  # type: ignore[arg-type]
        assert memory.count() == 0


class TestReadWrite:
    def test_get_update_history(self, memory: Memory) -> None:
        original = memory.remember("u", "I use Vim").memory
        updated = memory.update(original.id, content="I use Neovim", importance=0.8)
        assert updated.version == 2
        assert updated.content == "I use Neovim"
        assert updated.embedding_model is not None
        assert memory.recall("u", "neovim")[0].id == original.id
        assert memory.recall("u", "vim", strategy="keyword") == []
        events = memory.history(original.id)
        assert [(e.kind, e.content) for e in events] == [
            ("created", "I use Vim"),
            ("updated", "I use Neovim"),
        ]
        again = memory.update(original.id, metadata={"a": 1}, ttl="1d")
        assert again.metadata == {"a": 1}
        assert again.expires_at is not None
        cleared = memory.update(original.id, clear_expiry=True)
        assert cleared.expires_at is None

    def test_update_detects_conflicts(self, memory: Memory) -> None:
        paris = memory.remember("u", "I live in Paris")
        note = memory.remember("u", "Temporary note")
        memory.update(note.id, content="I live in Rome")
        assert memory.get(paris.id).status == "superseded"

    def test_update_validation(self, memory: Memory) -> None:
        mid = memory.remember("u", "x y z").id
        with pytest.raises(ValidationError, match="Nothing"):
            memory.update(mid)
        with pytest.raises(ValidationError):
            memory.update(mid, clear_expiry=True, ttl="1d")
        with pytest.raises(MemoryNotFoundError):
            memory.update("a" * 32, content="x")

    def test_forget_restore_delete(self, memory: Memory) -> None:
        mid = memory.remember("u", "I use Python").id
        archived = memory.forget(mid, reason="outdated")
        assert archived.status == "archived"
        assert memory.forget(mid).status == "archived"  # idempotent
        assert memory.recall("u", "python") == []
        assert memory.list("u") == []
        assert [m.id for m in memory.list("u", status="archived")] == [mid]
        assert memory.restore(mid).is_active
        assert memory.restore(mid).is_active
        memory.delete(mid)
        with pytest.raises(MemoryNotFoundError):
            memory.get(mid)
        with pytest.raises(MemoryNotFoundError):
            memory.delete(mid)
        with pytest.raises(MemoryNotFoundError):
            memory.history(mid)

    def test_list_count_users(self, memory: Memory) -> None:
        for i in range(5):
            memory.remember("a", f"note number {i}", memory_type="knowledge")
        memory.remember("b", "I prefer tea")
        assert [m.content for m in memory.list("a", limit=2)] == ["note number 4", "note number 3"]
        assert len(memory.list("a", offset=4)) == 1
        assert len(memory.list()) == 6
        assert len(memory.list(memory_type="preference")) == 1
        assert memory.list("a", order="oldest")[0].content == "note number 0"
        assert memory.count() == 6
        assert memory.users() == {"a": 5, "b": 1}
        with pytest.raises(ValidationError):
            memory.list(order="random")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            memory.list(limit=10_000)

    def test_clear(self, memory: Memory) -> None:
        memory.remember("a", "one thing")
        memory.remember("b", "two things")
        with pytest.raises(ValidationError):
            memory.clear()
        with pytest.raises(ValidationError):
            memory.clear("a", all_users=True)
        assert memory.clear("a") == 1
        assert memory.users() == {"b": 1}
        assert memory.clear(all_users=True) == 1
        assert memory.count() == 0

    def test_resolve_id(self, memory: Memory) -> None:
        mid = memory.remember("u", "x y z").id
        assert memory.resolve_id(mid[:6].upper()) == mid
        for bad in ["xyz!", "ab", 5]:
            with pytest.raises(ValidationError):
                memory.resolve_id(bad)  # type: ignore[arg-type]
        with pytest.raises(MemoryNotFoundError):
            memory.resolve_id("ffffffff" if not mid.startswith("ffffffff") else "00000000")

    def test_stats_and_inspect(self, memory: Memory) -> None:
        memory.ingest("u", "I prefer tea. I live in Oslo.")
        memory.remember("v", "x y z")
        stats = memory.stats()
        assert (stats.total, stats.active, stats.users, stats.relationships) == (3, 3, 2, 2)
        assert stats.vectors == 3
        assert stats.missing_vectors == 0
        assert stats.schema_version == 1
        assert stats.storage_bytes > 0
        user = memory.stats("u")
        assert user.total == 2
        summary = memory.inspect("u")
        assert summary.by_type == {"fact": 1, "preference": 1}
        assert summary.relationships == 2

    def test_repr_and_context_manager(self, db_path: Path) -> None:
        with Memory(db_path) as memory:
            assert "hashing" in repr(memory)
        with pytest.raises(HighHXPackError, match="closed"):
            memory.remember("u", "x")
        memory.close()  # idempotent


class TestEmbeddingFailures:
    def test_remember_survives_embedding_failure(self, tmp_path: Path, clock: FakeClock) -> None:
        calls = {"n": 0}

        def flaky(texts: list[str]) -> list[list[float]]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("provider down")
            return [[1.0, float(len(t))] for t in texts]

        embedder = CallableEmbedding(flaky, name="flaky")
        with Memory(tmp_path / "e.db", embedder=embedder, clock=clock) as memory:
            result = memory.remember("u", "I use Python")
            assert result.warnings
            assert "reindex" in result.warnings[0]
            assert result.memory.embedding_model is None
            assert memory.recall("u", "python", strategy="keyword")
            assert memory.stats().missing_vectors == 1
            assert memory.reindex() == 1
            assert memory.stats().missing_vectors == 0
            assert memory.reindex("u") == 0

    def test_reindex_after_model_change(self, db_path: Path, clock: FakeClock) -> None:
        with Memory(db_path, clock=clock) as memory:
            memory.remember("u", "I use Python")
        with Memory(
            db_path, embedder=CallableEmbedding(lambda t: [[1.0]] * len(t), name="new")
        ) as m:
            assert m.stats().missing_vectors == 1
            assert m.reindex() == 1

    def test_reindex_errors(self, tmp_path: Path) -> None:
        with (
            Memory(tmp_path / "n.db", embedder="none") as memory,
            pytest.raises(ConfigurationError),
        ):
            memory.reindex()

        def broken(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("down")

        with Memory(tmp_path / "b.db", embedder=CallableEmbedding(broken, name="b")) as memory:
            memory.remember("u", "x y z")
            with pytest.raises(HighHXPackError, match="Reindexing stopped"):
                memory.reindex()

    def test_invalid_provider_arguments(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            Memory(tmp_path / "x.db", embedder=42)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            Memory(tmp_path / "x.db", llm=42)  # type: ignore[arg-type]
        with Memory(":memory:", embedder="hashing") as memory:
            assert isinstance(memory.embedder, EmbeddingProvider)

    def test_embedding_error_type(self) -> None:
        assert issubclass(EmbeddingError, HighHXPackError)


class TestIngest:
    def test_ingest_extracts_and_dedupes(self, memory: Memory) -> None:
        results = memory.ingest(
            "u",
            "Hi! I've been using Python for ML for years, but I prefer C++ for contests. "
            "What do you think?",
            metadata={"session": "s1"},
        )
        assert [r.memory.memory_type for r in results] == ["fact", "preference"]
        assert all(r.memory.source == "conversation" for r in results)
        assert results[0].memory.metadata == {"session": "s1"}
        again = memory.ingest("u", "I prefer C++ for contests!")
        assert again[0].action == RememberAction.REINFORCED
        assert memory.ingest("u", "Thanks!") == []

    def test_ingest_validation(self, memory: Memory) -> None:
        with pytest.raises(ValidationError):
            memory.ingest("u", 42)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="too long"):
            memory.ingest("u", "x" * 300_000)

    def test_llm_extractor_from_config(self, tmp_path: Path) -> None:
        reply = '{"memories": [{"content": "User drinks tea", "memory_type": "preference"}]}'
        config = Config(storage_path=tmp_path / "l.db", llm_provider="ollama", extractor="llm")
        with Memory(config=config, llm=CallableProvider(lambda p: reply)) as memory:
            [result] = memory.ingest("u", "tea please, always tea")
            assert result.memory.content == "User drinks tea"

    def test_llm_extractor_requires_llm(self, tmp_path: Path) -> None:
        config = Config(storage_path=tmp_path / "l.db", llm_provider="ollama", extractor="llm")
        with Memory(config=config, llm="none") as memory, pytest.raises(ConfigurationError):
            memory.ingest("u", "I use Python")


class TestConsolidate:
    def test_merges_duplicates(self, memory: Memory) -> None:
        a = memory.remember("u", "User likes Python", dedupe=False)
        b = memory.remember("u", "I love Python", dedupe=False)
        c = memory.remember("u", "User prefers Python", dedupe=False)
        memory.remember("u", "I use Rust")
        preview = memory.consolidate("u", dry_run=True)
        assert len(preview.merged) == 2
        assert memory.count("u") == 4  # dry run changes nothing
        report = memory.consolidate("u")
        assert report.merged == preview.merged
        assert memory.count("u") == 2
        survivor = memory.get(a.id)
        assert survivor.mention_count == 3
        for loser in (b, c):
            merged = memory.get(loser.id)
            assert merged.status == "merged"
            assert merged.superseded_by == a.id
        assert not memory.consolidate("u").changed

    def test_archives_expired(self, memory: Memory, clock: FakeClock) -> None:
        temp = memory.remember("u", "Wifi password is hunter2", ttl="1h")
        clock.advance(hours=2)
        report = memory.consolidate("u")
        assert report.expired_archived == (temp.id,)
        assert memory.get(temp.id).status == "archived"
        assert memory.history(temp.id)[-1].reason == "expired"

    def test_detects_missed_conflicts(self, memory: Memory) -> None:
        old = memory.remember("u", "I live in Paris", detect_conflicts=False)
        new = memory.remember("u", "I live in Rome", detect_conflicts=False)
        report = memory.consolidate("u")
        assert report.superseded == ((new.id, old.id),)
        assert report.conflicts_detected == 1
        assert memory.get(old.id).status == "superseded"
        assert memory.consolidate("u").conflicts_detected == 0

    def test_conflicts_under_keep_both_are_flagged_once(self, tmp_path: Path) -> None:
        config = Config(storage_path=tmp_path / "k.db", conflict_policy="keep_both")
        with Memory(config=config) as memory:
            memory.remember("u", "I live in Paris", detect_conflicts=False)
            memory.remember("u", "I live in Rome", detect_conflicts=False)
            assert memory.consolidate("u").conflicts_detected == 1
            assert memory.consolidate("u").conflicts_detected == 0
            assert len(memory.conflicts("u")) == 1

    def test_summaries_are_additive_and_idempotent(self, memory: Memory) -> None:
        for text in ["I use Python at work", "Python is my favorite", "I teach Python"]:
            memory.remember("u", text)
        report = memory.consolidate("u", summarize=True)
        [summary_id] = report.summaries_created
        summary = memory.get(summary_id)
        assert summary.memory_type == "knowledge"
        assert summary.source == "consolidation"
        assert summary.content.startswith("About Python:")
        assert len(summary.metadata["summary_of"]) == 3
        assert memory.count("u") == 4
        assert memory.consolidate("u", summarize=True).summaries_created == ()
        custom = memory.consolidate(
            "u", summarize=True, summarizer=ExtractiveSummarizer(), dry_run=True
        )
        assert custom.summaries_created == ()

    def test_llm_summaries(self, tmp_path: Path) -> None:
        llm = CallableProvider(lambda prompt: "The user works with Python a lot.")
        with Memory(tmp_path / "s.db", llm=llm) as memory:
            for text in ["I use Python", "Python is great for AI", "I teach Python"]:
                memory.remember("u", text)
            [sid] = memory.consolidate("u", summarize=True).summaries_created
            assert memory.get(sid).content == "The user works with Python a lot."


def test_persistence_across_restart(db_path: Path) -> None:
    with Memory(db_path) as memory:
        mid = memory.remember("u", "I prefer Python for AI").id
        memory.ingest("u", "I live in Lisbon")
    with Memory(db_path) as memory:
        assert memory.get(mid).content == "I prefer Python for AI"
        assert memory.recall("u", "python")[0].id == mid
        assert memory.graph.edges("u")
        assert memory.count("u") == 2


def test_default_location_uses_home(isolated_env: Path) -> None:
    with Memory() as memory:
        memory.remember("u", "x y z")
        assert memory.location == str(isolated_env / "memory.db")
    assert (isolated_env / "memory.db").exists()


class TestAuditRegressions:
    def test_forget_keeps_superseded_link_restore_clears_it(self, memory: Memory) -> None:
        old = memory.remember("u", "I live in Paris")
        new = memory.remember("u", "I live in Rome")
        assert memory.forget(old.id).superseded_by == new.id
        assert memory.restore(old.id).superseded_by is None

    def test_hybrid_recall_survives_embedding_outage(self, tmp_path: Path) -> None:
        state = {"up": True}

        def embed(texts: list[str]) -> list[list[float]]:
            if not state["up"]:
                raise ConnectionError("provider down")
            return [[1.0, float(len(t))] for t in texts]

        with Memory(tmp_path / "o.db", embedder=CallableEmbedding(embed, name="e")) as memory:
            memory.remember("u", "I use Python")
            state["up"] = False
            [result] = memory.recall("u", "python")
            assert result.breakdown.semantic == 0.0
            assert result.breakdown.relevance == result.breakdown.keyword
            with pytest.raises(EmbeddingError):
                memory.recall("u", "python", strategy="semantic")

    def test_llm_is_created_lazily(self, tmp_path: Path) -> None:
        config = Config(storage_path=tmp_path / "l.db", llm_provider="openai")
        with Memory(config=config) as memory:
            memory.remember("u", "I use Python")  # no LLM needed: must work
            assert memory.count() == 1
            try:
                import openai  # noqa: F401
            except ImportError:
                with pytest.raises(HighHXPackError, match="openai"):
                    _ = memory.llm
        with Memory(tmp_path / "n.db", llm=CallableProvider(lambda p: "x")) as memory:
            assert memory.llm is not None
            assert memory.llm is memory.llm
