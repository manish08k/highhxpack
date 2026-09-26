"""Lexical retrieval with Okapi BM25.

Scores are normalized to ``[0, 1]`` by dividing by the score a document would
get if it contained every *known* query term exactly once at average length
(``sum(idf)``).  Query terms that occur in no stored memory are ignored for
normalization, so a query with one unknown word is not penalized for it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from highhxpack.utils.text import tokenize

if TYPE_CHECKING:
    from highhxpack.retrieval.filters import MemoryFilter
    from highhxpack.storage.base import KeywordData, StorageBackend


def idf(total_documents: int, document_frequency: int) -> float:
    """BM25 inverse document frequency (always positive)."""
    return math.log(1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))


def bm25_scores(
    query_terms: Sequence[str], data: KeywordData, *, k1: float, b: float
) -> dict[str, float]:
    """Return normalized BM25 scores keyed by memory id (only ids with a match)."""
    terms = set(query_terms)
    known = {t: data.document_frequency[t] for t in terms if data.document_frequency.get(t)}
    if not known or data.total_documents == 0:
        return {}
    weights = {t: idf(data.total_documents, df) for t, df in known.items()}
    normalizer = sum(weights.values())
    average_length = data.average_length or 1.0
    raw: dict[str, float] = {}
    for memory_id, term, tf, length in data.postings:
        weight = weights.get(term)
        if weight is None:
            continue
        denominator = tf + k1 * (1.0 - b + b * length / average_length)
        raw[memory_id] = raw.get(memory_id, 0.0) + weight * tf * (k1 + 1.0) / denominator
    return {memory_id: min(score / normalizer, 1.0) for memory_id, score in raw.items()}


def keyword_search(
    storage: StorageBackend,
    query: str,
    flt: MemoryFilter,
    *,
    k1: float,
    b: float,
    limit: int,
) -> dict[str, float]:
    """Return the ``limit`` best keyword matches as ``{memory_id: score}``."""
    terms = tokenize(query)
    if not terms:
        return {}
    scores = bm25_scores(terms, storage.keyword_data(terms, flt), k1=k1, b=b)
    best = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return dict(best)
