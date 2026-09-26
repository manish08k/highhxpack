"""Embeddings that run entirely on the local machine.

:class:`HashingEmbedding` is the default.  It needs no model download and no
third-party packages, and it is fully deterministic.  It is a *lexical*
embedding: it captures shared words, word stems and sub-word fragments (so
"programming" is close to "programmer"), but it does not know that "car" and
"automobile" are synonyms.  For true semantic similarity install a neural model
(``pip install "highhxpack[sentence-transformers]"``) or use Ollama.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from highhxpack.embeddings.base import EmbeddingProvider, check_vectors
from highhxpack.exceptions import EmbeddingError, ProviderNotAvailableError
from highhxpack.storage.vector import normalize
from highhxpack.utils.hashing import stable_hash64
from highhxpack.utils.text import tokenize

DEFAULT_HASHING_DIMENSION = 384
DEFAULT_SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"


class HashingEmbedding(EmbeddingProvider):
    """Deterministic feature-hashing embedding.

    Features (each hashed into ``dimension`` buckets with a sign bit to reduce
    collision bias):

    * stemmed, stop-word-filtered tokens (weight ``token_weight``),
    * adjacent token pairs, which capture short phrases (``bigram_weight``),
    * character trigrams of each token, which capture morphology (``trigram_weight``).

    Term counts are dampened with ``1 + log(count)``.
    """

    VERSION = 1

    def __init__(
        self,
        dimension: int = DEFAULT_HASHING_DIMENSION,
        *,
        token_weight: float = 1.0,
        bigram_weight: float = 0.5,
        trigram_weight: float = 0.25,
    ):
        if not 16 <= dimension <= 8192:
            raise ValueError("dimension must be between 16 and 8192")
        self.dimension = dimension
        self.token_weight = token_weight
        self.bigram_weight = bigram_weight
        self.trigram_weight = trigram_weight
        self.name = f"hashing-v{self.VERSION}-{dimension}"

    def _features(self, text: str) -> Counter[tuple[str, float]]:
        tokens = tokenize(text)
        features: Counter[tuple[str, float]] = Counter()
        for token in tokens:
            features[("t:" + token, self.token_weight)] += 1
            padded = f"<{token}>"
            for i in range(len(padded) - 2):
                features[("c:" + padded[i : i + 3], self.trigram_weight)] += 1
        for left, right in pairwise(tokens):
            features[(f"b:{left} {right}", self.bigram_weight)] += 1
        return features

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for (feature, weight), count in self._features(text).items():
            hashed = stable_hash64(feature)
            index = hashed % self.dimension
            sign = 1.0 if (hashed >> 63) & 1 == 0 else -1.0
            vector[index] += sign * weight * (1.0 + math.log(count))
        return normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Neural embeddings via the optional ``sentence-transformers`` package.

    The model runs locally; it is downloaded from the Hugging Face Hub the first
    time it is used unless it is already cached.
    """

    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER, *, model: Any = None):
        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self._model = model

    def _load(self) -> Any:  # pragma: no cover - requires the optional dependency
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ProviderNotAvailableError(
                    "The sentence-transformers embedding provider is not installed.",
                    hint='Install it with: pip install "highhxpack[sentence-transformers]"',
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load sentence-transformers model {self.model_name!r}.",
                    reason=str(exc),
                    hint="Check the model name and your network connection for the first download.",
                ) from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        try:
            raw = model.encode(list(texts), normalize_embeddings=True)
            vectors = [list(map(float, row)) for row in raw]
        except Exception as exc:
            raise EmbeddingError(
                "sentence-transformers failed to embed text.", reason=str(exc)
            ) from exc
        return check_vectors(vectors, len(texts), provider=self.name)
