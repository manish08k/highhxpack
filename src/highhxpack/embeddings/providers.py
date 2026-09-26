"""Remote and user-supplied embedding providers, and the provider factory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from highhxpack.embeddings.base import EmbeddingProvider, check_vectors
from highhxpack.embeddings.local import (
    DEFAULT_HASHING_DIMENSION,
    DEFAULT_SENTENCE_TRANSFORMER,
    HashingEmbedding,
    SentenceTransformerEmbedding,
)
from highhxpack.exceptions import ConfigurationError, EmbeddingError
from highhxpack.providers.base import DEFAULT_TIMEOUT_SECONDS, post_json, validate_base_url
from highhxpack.providers.ollama import DEFAULT_OLLAMA_URL
from highhxpack.providers.openai import load_openai_client
from highhxpack.utils.logging import redact

DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDERS = ("hashing", "none", "sentence-transformers", "ollama", "openai")


class OllamaEmbedding(EmbeddingProvider):
    """Embeddings from a local Ollama server (``/api/embed``)."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.model = model
        self.base_url = validate_base_url(base_url, provider="Ollama")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = post_json(
            f"{self.base_url}/api/embed",
            {"model": self.model, "input": list(texts)},
            timeout=self.timeout,
            provider="Ollama",
            unreachable_hint=(
                f"Start Ollama (`ollama serve`) and pull the model (`ollama pull {self.model}`)."
            ),
        )
        return check_vectors(data.get("embeddings"), len(texts), provider=self.name)


class OpenAIEmbedding(EmbeddingProvider):
    """Embeddings from the OpenAI API.  Requires ``highhxpack[openai]``."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Any = None,
    ):
        self.model = model
        self.name = f"openai:{model}"
        self._client = client or load_openai_client(api_key, base_url, timeout)

    def __repr__(self) -> str:
        return f"OpenAIEmbedding(model={self.model!r})"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self.model, input=list(texts))
            vectors = [list(item.embedding) for item in response.data]
        except Exception as exc:
            raise EmbeddingError(
                "OpenAI embedding request failed.",
                reason=redact(f"{type(exc).__name__}: {exc}"),
                hint="Check the API key, model name and network connectivity.",
            ) from exc
        return check_vectors(vectors, len(texts), provider=self.name)


class CallableEmbedding(EmbeddingProvider):
    """Adapt ``fn(texts) -> list[list[float]]`` to :class:`EmbeddingProvider`.

    ``name`` must identify the underlying model, because vectors from different
    names are never compared.
    """

    def __init__(self, fn: Callable[[list[str]], Sequence[Sequence[float]]], *, name: str):
        if not callable(fn):
            raise TypeError("fn must be callable")
        if not name:
            raise ValueError("name must identify the embedding model")
        self._fn = fn
        self.name = name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._fn(list(texts))
        except Exception as exc:
            raise EmbeddingError(
                f"Custom embedding provider {self.name!r} raised {type(exc).__name__}.",
                reason=str(exc),
            ) from exc
        return check_vectors(
            [list(v) for v in vectors] if isinstance(vectors, list | tuple) else vectors,
            len(texts),
            provider=self.name,
        )


def create_embedding_provider(
    name: str,
    *,
    model: str | None = None,
    ollama_url: str | None = None,
    openai_base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> EmbeddingProvider | None:
    """Build an embedding provider by name; ``"none"`` disables embeddings."""
    key = name.strip().lower()
    t = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
    if key == "none":
        return None
    if key == "hashing":
        dimension = DEFAULT_HASHING_DIMENSION
        if model:
            try:
                dimension = int(model)
            except ValueError as exc:
                raise ConfigurationError(
                    f"Invalid hashing embedding dimension: {model!r}",
                    hint="For the hashing provider, embedding_model is the vector size, e.g. 384.",
                ) from exc
        try:
            return HashingEmbedding(dimension)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
    if key == "sentence-transformers":
        return SentenceTransformerEmbedding(model or DEFAULT_SENTENCE_TRANSFORMER)
    if key == "ollama":
        return OllamaEmbedding(
            model or DEFAULT_OLLAMA_EMBEDDING_MODEL,
            base_url=ollama_url or DEFAULT_OLLAMA_URL,
            timeout=t,
        )
    if key == "openai":
        return OpenAIEmbedding(
            model or DEFAULT_OPENAI_EMBEDDING_MODEL,
            api_key=api_key,
            base_url=openai_base_url,
            timeout=t,
        )
    raise ConfigurationError(
        f"Unknown embedding provider: {name!r}",
        hint=f"Use one of: {', '.join(EMBEDDING_PROVIDERS)}, or pass an EmbeddingProvider.",
    )


__all__ = [
    "EMBEDDING_PROVIDERS",
    "CallableEmbedding",
    "OllamaEmbedding",
    "OpenAIEmbedding",
    "create_embedding_provider",
]
