"""Embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from highhxpack.exceptions import EmbeddingError
from highhxpack.storage.vector import normalize


class EmbeddingProvider(ABC):
    """Turns text into fixed-length vectors.

    ``name`` must uniquely identify the model *and* its configuration, because
    stored vectors are only compared with query vectors from the same ``name``.
    Changing the provider therefore never mixes incompatible vectors; run
    :meth:`highhxpack.Memory.reindex` to embed existing memories with a new model.
    """

    name: str = "embedding"

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.  Override for models with distinct query encoders."""
        return self.embed_one(text)


def check_vectors(vectors: object, expected: int, *, provider: str) -> list[list[float]]:
    """Validate provider output and return unit-length float vectors."""
    if not isinstance(vectors, list | tuple) or len(vectors) != expected:
        raise EmbeddingError(
            f"Embedding provider {provider!r} returned "
            f"{len(vectors) if isinstance(vectors, list | tuple) else 'no'} vectors "
            f"for {expected} texts."
        )
    result: list[list[float]] = []
    dim: int | None = None
    for vector in vectors:
        if not isinstance(vector, Sequence) or isinstance(vector, str | bytes) or not vector:
            raise EmbeddingError(f"Embedding provider {provider!r} returned a malformed vector.")
        if dim is None:
            dim = len(vector)
        elif len(vector) != dim:
            raise EmbeddingError(
                f"Embedding provider {provider!r} returned vectors of different lengths."
            )
        try:
            result.append(normalize([float(v) for v in vector]))
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(
                f"Embedding provider {provider!r} returned non-numeric values."
            ) from exc
    return result
