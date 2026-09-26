"""Transparent, centrally configured ranking.

Every constant that influences retrieval order lives in :class:`ScoringConfig`.
The final score of a memory is::

    relevance = (keyword_weight * keyword + semantic_weight * semantic)
                / (keyword_weight + semantic_weight)          # over enabled channels

    final = relevance * (base_weight
                         + importance_weight * importance
                         + recency_weight    * recency
                         + confidence_weight * confidence
                         + frequency_weight  * frequency)

The five weights inside the parentheses are normalized to sum to 1, so
``final`` is in ``[0, 1]``.  Multiplying by ``relevance`` means a memory that
does not match the query scores 0 no matter how important or recent it is; the
other signals only re-order memories that *are* relevant.

* ``recency``    = ``0.5 ** (age_days / recency_half_life_days)`` using ``updated_at``.
* ``frequency``  = ``log1p(n) / log1p(frequency_saturation)`` capped at 1, where
  ``n`` is the number of repeated mentions plus recall accesses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

from highhxpack.exceptions import ConfigurationError
from highhxpack.models.memory import MemoryRecord
from highhxpack.models.result import ScoreBreakdown
from highhxpack.utils.timestamps import age_in_days


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """All tunable ranking parameters.  See the module docstring for the formula."""

    # Weights inside the final-score multiplier (normalized to sum to 1).
    base_weight: float = 0.60
    importance_weight: float = 0.15
    recency_weight: float = 0.10
    confidence_weight: float = 0.10
    frequency_weight: float = 0.05
    # Relevance channel weights for hybrid retrieval.
    keyword_weight: float = 0.5
    semantic_weight: float = 0.5
    # Signal shapes.
    recency_half_life_days: float = 30.0
    frequency_saturation: int = 20
    # BM25 parameters (standard Robertson/Sparck-Jones defaults).
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    # Vector matches below this cosine similarity are not considered relevant.
    min_semantic_similarity: float = 0.2
    # How many candidates each channel contributes before final ranking.
    candidate_multiplier: int = 5
    min_candidates: int = 50

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigurationError(f"scoring.{f.name} must be a number.")
            if math.isnan(value) or value < 0:
                raise ConfigurationError(f"scoring.{f.name} must be non-negative, got {value}.")
        if self.multiplier_total <= 0:
            raise ConfigurationError("At least one final-score weight must be positive.")
        if self.keyword_weight + self.semantic_weight <= 0:
            raise ConfigurationError("keyword_weight and semantic_weight cannot both be 0.")
        if self.recency_half_life_days <= 0:
            raise ConfigurationError("scoring.recency_half_life_days must be positive.")
        if self.frequency_saturation < 1:
            raise ConfigurationError("scoring.frequency_saturation must be at least 1.")
        if not 0 <= self.bm25_b <= 1:
            raise ConfigurationError("scoring.bm25_b must be between 0 and 1.")
        if self.min_semantic_similarity > 1:
            raise ConfigurationError("scoring.min_semantic_similarity must be at most 1.")
        if self.candidate_multiplier < 1 or self.min_candidates < 1:
            raise ConfigurationError("candidate_multiplier and min_candidates must be >= 1.")

    @property
    def multiplier_total(self) -> float:
        return (
            self.base_weight
            + self.importance_weight
            + self.recency_weight
            + self.confidence_weight
            + self.frequency_weight
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ScoringConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigurationError(
                f"Unknown scoring option: {unknown[0]!r}",
                hint=f"Valid options: {', '.join(sorted(known))}.",
            )
        return cls(**dict(data))

    def to_dict(self) -> dict[str, float | int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class Ranker:
    """Computes :class:`ScoreBreakdown` values according to a :class:`ScoringConfig`."""

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()

    def relevance(
        self, keyword: float, semantic: float, *, use_keyword: bool, use_semantic: bool
    ) -> float:
        cfg = self.config
        weight_kw = cfg.keyword_weight if use_keyword else 0.0
        weight_sem = cfg.semantic_weight if use_semantic else 0.0
        total = weight_kw + weight_sem
        if total <= 0:
            return 0.0
        return _clamp((weight_kw * keyword + weight_sem * semantic) / total)

    def recency(self, memory: MemoryRecord, now: datetime) -> float:
        age = age_in_days(memory.updated_at, now)
        return _clamp(0.5 ** (age / self.config.recency_half_life_days))

    def frequency(self, memory: MemoryRecord) -> float:
        events = max(memory.mention_count - 1, 0) + max(memory.access_count, 0)
        return _clamp(math.log1p(events) / math.log1p(self.config.frequency_saturation))

    def score(
        self,
        memory: MemoryRecord,
        *,
        keyword: float,
        semantic: float,
        now: datetime,
        use_keyword: bool = True,
        use_semantic: bool = True,
    ) -> ScoreBreakdown:
        cfg = self.config
        relevance = self.relevance(
            keyword, semantic, use_keyword=use_keyword, use_semantic=use_semantic
        )
        importance = _clamp(memory.importance)
        recency = self.recency(memory, now)
        confidence = _clamp(memory.confidence)
        frequency = self.frequency(memory)
        multiplier = (
            cfg.base_weight
            + cfg.importance_weight * importance
            + cfg.recency_weight * recency
            + cfg.confidence_weight * confidence
            + cfg.frequency_weight * frequency
        ) / cfg.multiplier_total
        return ScoreBreakdown(
            keyword=_clamp(keyword),
            semantic=_clamp(semantic),
            relevance=relevance,
            importance=importance,
            recency=recency,
            confidence=confidence,
            frequency=frequency,
            final=_clamp(relevance * multiplier),
        )


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value
