"""Candidate generation and ranking for recall and search."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from highhxpack.exceptions import ConfigurationError, HighHXPackError, ValidationError
from highhxpack.models.relationship import ConflictStatus
from highhxpack.models.result import RecallResult
from highhxpack.retrieval.hybrid import merge_candidates
from highhxpack.retrieval.keyword import keyword_search
from highhxpack.retrieval.ranking import Ranker, ScoringConfig
from highhxpack.retrieval.semantic import semantic_search
from highhxpack.utils.logging import get_logger

if TYPE_CHECKING:
    from highhxpack.embeddings.base import EmbeddingProvider
    from highhxpack.retrieval.filters import MemoryFilter
    from highhxpack.storage.base import StorageBackend


logger = get_logger("retrieval")


class Strategy(StrEnum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


def parse_strategy(value: str | Strategy) -> Strategy:
    try:
        return Strategy(str(getattr(value, "value", value)).lower())
    except ValueError as exc:
        raise ValidationError(
            f"Unknown retrieval strategy: {value!r}",
            hint="Use 'keyword', 'semantic' or 'hybrid'.",
        ) from exc


class Retriever:
    """Finds and ranks memories for a query.

    ``hybrid`` gracefully degrades to keyword-only retrieval when no embedding
    provider is configured; ``semantic`` requires one.
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider | None,
        scoring: ScoringConfig | None = None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.ranker = Ranker(scoring)

    @property
    def scoring(self) -> ScoringConfig:
        return self.ranker.config

    def retrieve(
        self,
        query: str,
        flt: MemoryFilter,
        *,
        now: datetime,
        limit: int,
        strategy: Strategy = Strategy.HYBRID,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        cfg = self.scoring
        use_keyword = strategy in (Strategy.KEYWORD, Strategy.HYBRID)
        use_semantic = (
            strategy in (Strategy.SEMANTIC, Strategy.HYBRID) and self.embedder is not None
        )
        if strategy == Strategy.SEMANTIC and self.embedder is None:
            raise ConfigurationError(
                "Semantic search needs an embedding provider, but embeddings are disabled.",
                hint="Set embedding_provider to 'hashing' (the default) or another provider.",
            )
        pool = max(limit * cfg.candidate_multiplier, cfg.min_candidates)
        keyword_scores = (
            keyword_search(self.storage, query, flt, k1=cfg.bm25_k1, b=cfg.bm25_b, limit=pool)
            if use_keyword
            else {}
        )
        semantic_scores: dict[str, float] = {}
        if use_semantic and self.embedder is not None:
            try:
                semantic_scores = semantic_search(
                    self.storage,
                    self.embedder,
                    query,
                    flt,
                    limit=pool,
                    min_similarity=cfg.min_semantic_similarity,
                )
            except HighHXPackError as exc:
                if strategy == Strategy.SEMANTIC:
                    raise
                # Hybrid retrieval stays available on keyword relevance alone.
                logger.warning("Semantic retrieval unavailable, using keyword only: %s", exc)
                use_semantic = False
        candidates = merge_candidates(keyword_scores, semantic_scores)
        if not candidates:
            return []
        records = self.storage.get_memories([c.memory_id for c in candidates])
        results: list[RecallResult] = []
        for candidate in candidates:
            memory = records.get(candidate.memory_id)
            if memory is None:  # deleted concurrently
                continue
            breakdown = self.ranker.score(
                memory,
                keyword=candidate.keyword,
                semantic=candidate.semantic,
                now=now,
                use_keyword=use_keyword,
                use_semantic=use_semantic,
            )
            if breakdown.final <= 0.0 or breakdown.final < min_score:
                continue
            results.append(RecallResult(memory=memory, score=breakdown.final, breakdown=breakdown))
        results.sort(
            key=lambda r: (-r.score, -r.breakdown.relevance, -r.memory.updated_at.timestamp(), r.id)
        )
        results = results[:limit]
        if not results:
            return results
        conflicted: set[str] = set()
        for conflict in self.storage.list_conflicts(
            status=ConflictStatus.OPEN.value, memory_ids=[r.id for r in results]
        ):
            conflicted.update((conflict.memory_id, conflict.other_id))
        if conflicted:
            results = [
                RecallResult(r.memory, r.score, r.breakdown, has_conflict=r.id in conflicted)
                for r in results
            ]
        return results
