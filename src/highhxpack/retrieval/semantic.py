"""Embedding-based retrieval (exact cosine similarity)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from highhxpack.embeddings.base import check_vectors
from highhxpack.storage import vector as vec

if TYPE_CHECKING:
    from highhxpack.embeddings.base import EmbeddingProvider
    from highhxpack.retrieval.filters import MemoryFilter
    from highhxpack.storage.base import StorageBackend


def semantic_search(
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    query: str,
    flt: MemoryFilter,
    *,
    limit: int,
    min_similarity: float,
) -> dict[str, float]:
    """Return up to ``limit`` ``{memory_id: similarity}`` pairs above ``min_similarity``.

    Only vectors produced by ``embedder`` (same ``name``) are compared.
    """
    query_vector = check_vectors([embedder.embed_query(query)], 1, provider=embedder.name)[0]
    matches = vec.top_k(
        query_vector,
        storage.iter_vectors(flt, embedder.name),
        k=limit,
        min_similarity=min_similarity,
    )
    return dict(matches)
