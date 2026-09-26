"""Optional embedding providers; the default :class:`HashingEmbedding` is local."""

from highhxpack.embeddings.base import EmbeddingProvider
from highhxpack.embeddings.local import HashingEmbedding, SentenceTransformerEmbedding
from highhxpack.embeddings.providers import (
    EMBEDDING_PROVIDERS,
    CallableEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    create_embedding_provider,
)

__all__ = [
    "EMBEDDING_PROVIDERS",
    "CallableEmbedding",
    "EmbeddingProvider",
    "HashingEmbedding",
    "OllamaEmbedding",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    "create_embedding_provider",
]
