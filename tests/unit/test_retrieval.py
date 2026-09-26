from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from highhxpack import ConfigurationError, Memory, ScoringConfig, ValidationError
from highhxpack.models.memory import MemoryRecord
from highhxpack.retrieval.hybrid import merge_candidates
from highhxpack.retrieval.keyword import bm25_scores, idf
from highhxpack.retrieval.ranking import Ranker
from highhxpack.retrieval.retriever import Strategy, parse_strategy
from highhxpack.storage.base import KeywordData
from support import FakeClock

NOW = datetime(2025, 1, 1, tzinfo=UTC)


def rec(**changes: object) -> MemoryRecord:
    base = MemoryRecord(
        id="a" * 32,
        user_id="u",
        content="x",
        memory_type="fact",
        source="api",
        importance=0.5,
        confidence=1.0,
        created_at=NOW,
        updated_at=NOW,
    )
    return base.with_changes(**changes)


class TestRanking:
    def test_formula_is_documented_weighted_product(self) -> None:
        ranker = Ranker()
        b = ranker.score(rec(importance=1.0, confidence=1.0), keyword=1.0, semantic=1.0, now=NOW)
        cfg = ranker.config
        expected = (
            cfg.base_weight + cfg.importance_weight + cfg.recency_weight + cfg.confidence_weight
        ) / cfg.multiplier_total
        assert b.relevance == 1.0
        assert b.recency == 1.0
        assert b.frequency == 0.0
        assert b.final == pytest.approx(expected)

    def test_irrelevant_scores_zero(self) -> None:
        b = Ranker().score(rec(importance=1.0), keyword=0.0, semantic=0.0, now=NOW)
        assert b.final == 0.0

    def test_recency_half_life(self) -> None:
        ranker = Ranker(ScoringConfig(recency_half_life_days=10))
        assert ranker.recency(rec(), NOW + timedelta(days=10)) == pytest.approx(0.5)
        assert ranker.recency(rec(), NOW + timedelta(days=20)) == pytest.approx(0.25)

    def test_frequency_saturates(self) -> None:
        ranker = Ranker(ScoringConfig(frequency_saturation=4))
        assert ranker.frequency(rec(mention_count=1, access_count=0)) == 0.0
        assert ranker.frequency(rec(mention_count=3, access_count=2)) == 1.0
        mid = ranker.frequency(rec(mention_count=2))
        assert mid == pytest.approx(math.log1p(1) / math.log1p(4))

    def test_relevance_uses_only_enabled_channels(self) -> None:
        ranker = Ranker(ScoringConfig(keyword_weight=3, semantic_weight=1))
        assert ranker.relevance(0.8, 0.0, use_keyword=True, use_semantic=True) == pytest.approx(0.6)
        assert ranker.relevance(0.8, 0.0, use_keyword=True, use_semantic=False) == pytest.approx(
            0.8
        )
        assert ranker.relevance(0.8, 0.4, use_keyword=False, use_semantic=True) == pytest.approx(
            0.4
        )
        assert ranker.relevance(0.8, 0.4, use_keyword=False, use_semantic=False) == 0.0

    def test_signals_reorder_equally_relevant_memories(self) -> None:
        ranker = Ranker()
        important = ranker.score(rec(importance=0.9), keyword=0.5, semantic=0.5, now=NOW)
        trivial = ranker.score(rec(importance=0.1), keyword=0.5, semantic=0.5, now=NOW)
        assert important.final > trivial.final

    def test_from_mapping(self) -> None:
        assert ScoringConfig.from_mapping({"bm25_k1": 2}).bm25_k1 == 2
        assert ScoringConfig().to_dict()["base_weight"] == 0.6


class TestBM25:
    def test_idf_positive_and_decreasing(self) -> None:
        assert idf(10, 1) > idf(10, 5) > idf(10, 10) > 0

    def test_normalized_and_ignores_unknown_terms(self) -> None:
        data = KeywordData(
            total_documents=2,
            average_length=2.0,
            document_frequency={"python": 1},
            postings=[("m1", "python", 1, 2)],
        )
        scores = bm25_scores(["python", "unknownterm"], data, k1=1.2, b=0.75)
        assert scores == {"m1": pytest.approx(1.0)}

    def test_empty(self) -> None:
        assert bm25_scores(["a"], KeywordData(0, 0.0, {}), k1=1.2, b=0.75) == {}

    def test_merge_candidates(self) -> None:
        merged = merge_candidates({"b": 0.5, "a": 0.2}, {"a": 0.9, "c": 0.4})
        assert [(c.memory_id, c.keyword, c.semantic) for c in merged] == [
            ("a", 0.2, 0.9),
            ("b", 0.5, 0.0),
            ("c", 0.0, 0.4),
        ]


@pytest.fixture
def corpus(mem_memory: Memory) -> Memory:
    for text, mtype in [
        ("I prefer Python for AI development", "preference"),
        ("My favorite editor is Neovim", "preference"),
        ("I work at Acme Corp as a data engineer", "fact"),
        ("I went to PyCon in Pittsburgh last year", "event"),
        ("I want to learn Rust this year", "goal"),
    ]:
        mem_memory.remember("u", text, memory_type=mtype)
    mem_memory.remember("other", "I prefer Java for AI development")
    return mem_memory


class TestRecall:
    def test_spec_example(self, corpus: Memory) -> None:
        results = corpus.recall("u", "What language do I prefer for AI?")
        assert results[0].content == "I prefer Python for AI development"
        assert all(r.memory.user_id == "u" for r in results)

    def test_scores_sorted_and_bounded(self, corpus: Memory) -> None:
        results = corpus.recall("u", "python rust editor")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(0 < s <= 1 for s in scores)

    def test_filters(self, corpus: Memory) -> None:
        results = corpus.recall("u", "prefer favorite", memory_type="preference")
        assert {r.memory.memory_type for r in results} == {"preference"}
        assert corpus.recall("u", "python", min_score=0.99) == []
        assert corpus.recall("u", "python", limit=1)[0].content.startswith("I prefer Python")

    def test_strategies(self, corpus: Memory) -> None:
        for strategy in ("keyword", "semantic", "hybrid", Strategy.KEYWORD):
            results = corpus.recall("u", "Python AI development", strategy=strategy)
            assert results
            assert results[0].content == "I prefer Python for AI development"
        keyword = corpus.recall("u", "python", strategy="keyword")[0].breakdown
        assert keyword.semantic == 0.0

    def test_semantic_matches_morphology(self, corpus: Memory) -> None:
        # "engineering" shares sub-word features with "engineer" via character n-grams.
        results = corpus.recall("u", "engineering", strategy="semantic")
        assert results
        assert "engineer" in results[0].content

    def test_no_match(self, corpus: Memory) -> None:
        assert corpus.recall("u", "zzqx", strategy="keyword") == []

    def test_deterministic(self, corpus: Memory) -> None:
        first = [(r.id, r.score) for r in corpus.search("I", user_id="u")]
        second = [(r.id, r.score) for r in corpus.search("I", user_id="u")]
        assert first == second

    def test_recall_tracks_access_search_does_not(self, corpus: Memory) -> None:
        hit = corpus.search("Neovim", user_id="u")[0]
        assert corpus.get(hit.id).access_count == 0
        corpus.recall("u", "Neovim")
        assert corpus.get(hit.id).access_count == 1

    def test_search_across_users(self, corpus: Memory) -> None:
        users = {r.memory.user_id for r in corpus.search("prefer AI development")}
        assert users == {"u", "other"}

    def test_invalid_arguments(self, corpus: Memory) -> None:
        with pytest.raises(ValidationError):
            corpus.recall("u", "")
        with pytest.raises(ValidationError):
            corpus.recall("u", "x", strategy="magic")
        with pytest.raises(ValidationError):
            corpus.recall("u", "x", limit=0)
        with pytest.raises(ValidationError):
            corpus.recall("u", "x", min_score=2)
        with pytest.raises(ValidationError):
            parse_strategy("nope")

    def test_semantic_requires_embedder(self, clock: FakeClock) -> None:
        with Memory(":memory:", embedder="none", clock=clock) as memory:
            memory.remember("u", "I use Python")
            with pytest.raises(ConfigurationError, match="Semantic"):
                memory.recall("u", "python", strategy="semantic")
            # hybrid degrades to keyword-only
            assert memory.recall("u", "python")[0].breakdown.relevance == pytest.approx(
                memory.recall("u", "python", strategy="keyword")[0].breakdown.relevance
            )

    def test_recency_prefers_newer_when_equally_relevant(self, clock: FakeClock) -> None:
        with Memory(":memory:", clock=clock) as memory:
            old = memory.remember("u", "Standup meeting notes", memory_type="event")
            clock.advance(days=90)
            new = memory.remember("u", "Standup meeting notes today", memory_type="event")
            results = memory.recall("u", "standup meeting")
            assert results[0].id == new.id
            assert results[1].id == old.id

    def test_expired_memories_hidden(self, mem_memory: Memory, clock: FakeClock) -> None:
        mem_memory.remember("u", "Temporary door code is 1234", ttl="1h")
        assert mem_memory.recall("u", "door code")
        clock.advance(hours=2)
        assert mem_memory.recall("u", "door code") == []
        assert mem_memory.search("door code", include_expired=True)
