"""Deduplication, conflict handling, summarization and importance scoring."""

from highhxpack.consolidation.conflict import (
    Claim,
    ConflictKind,
    ConflictPolicy,
    detect_conflicts,
    parse_claim,
)
from highhxpack.consolidation.deduplication import (
    canonical_tokens,
    find_duplicate_pairs,
    is_near_duplicate,
)
from highhxpack.consolidation.importance import ImportanceConfig, estimate_importance, reinforce
from highhxpack.consolidation.summarization import (
    ExtractiveSummarizer,
    LLMSummarizer,
    Summarizer,
    group_by_entity,
)

__all__ = [
    "Claim",
    "ConflictKind",
    "ConflictPolicy",
    "ExtractiveSummarizer",
    "ImportanceConfig",
    "LLMSummarizer",
    "Summarizer",
    "canonical_tokens",
    "detect_conflicts",
    "estimate_importance",
    "find_duplicate_pairs",
    "group_by_entity",
    "is_near_duplicate",
    "parse_claim",
    "reinforce",
]
