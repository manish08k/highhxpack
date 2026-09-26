"""Vector serialization and similarity helpers.

Vectors are stored as little-endian float32 blobs.  Similarity search is an
exact (brute-force) cosine scan, which is simple, deterministic and fast enough
for the per-user memory counts HighHXPack targets; see ``benchmarks/``.
"""

from __future__ import annotations

import math
import operator
import sys
from array import array
from collections.abc import Iterable, Sequence

from highhxpack.exceptions import EmbeddingError


def normalize(vector: Sequence[float]) -> list[float]:
    """Return ``vector`` scaled to unit length (a zero vector stays zero)."""
    values = [float(v) for v in vector]
    for value in values:
        if math.isnan(value) or math.isinf(value):
            raise EmbeddingError(
                "Embedding contains NaN or infinite values.",
                hint="Check the embedding provider; the vector was not stored.",
            )
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return values
    return [v / norm for v in values]


def encode(vector: Sequence[float]) -> bytes:
    """Serialize a vector to a float32 little-endian blob."""
    packed = array("f", vector)
    if sys.byteorder != "little":  # pragma: no cover - big-endian platforms are rare
        packed.byteswap()
    return packed.tobytes()


def decode(blob: bytes, dim: int) -> list[float]:
    """Deserialize a blob produced by :func:`encode`."""
    packed = array("f")
    packed.frombytes(blob)
    if sys.byteorder != "little":  # pragma: no cover
        packed.byteswap()
    if len(packed) != dim:
        raise EmbeddingError(
            f"Stored vector has {len(packed)} dimensions, expected {dim}.",
            hint="Run `memory.reindex()` to rebuild embeddings.",
        )
    return packed.tolist()


def _dot_fallback(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(map(operator.mul, a, b)))


# math.sumprod (Python 3.12+) is several times faster and uses extended precision.
_sumprod = getattr(math, "sumprod", _dot_fallback)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product; equals cosine similarity for unit-length vectors."""
    return float(_sumprod(a, b))


def top_k(
    query: Sequence[float],
    candidates: Iterable[tuple[str, Sequence[float]]],
    *,
    k: int,
    min_similarity: float,
) -> list[tuple[str, float]]:
    """Return up to ``k`` ``(id, similarity)`` pairs with similarity >= ``min_similarity``.

    ``query`` and candidates must be unit length.  Ties are broken by id so the
    result is deterministic.
    """
    scored = []
    for key, vector in candidates:
        similarity = dot(query, vector)
        if similarity >= min_similarity:
            scored.append((key, min(similarity, 1.0)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:k]
