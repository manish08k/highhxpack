"""HighHXPack: a local-first memory and context engine for AI applications.

Quick start::

    from highhxpack import Memory

    memory = Memory()
    memory.remember("user_123", "I prefer Python for AI development.")
    for result in memory.recall("user_123", "What language do I prefer for AI?"):
        print(result.score, result.content)

Everything importable from this module is public API and follows semantic
versioning.  Submodules are internal unless documented otherwise.
"""

from highhxpack.__version__ import __version__
from highhxpack.consolidation.conflict import ConflictPolicy
from highhxpack.core.config import Config
from highhxpack.core.memory import Memory
from highhxpack.embeddings.base import EmbeddingProvider
from highhxpack.embeddings.local import HashingEmbedding
from highhxpack.embeddings.providers import CallableEmbedding
from highhxpack.exceptions import (
    ConfigurationError,
    ConflictNotFoundError,
    EmbeddingError,
    ExportError,
    ExtractionError,
    HighHXPackError,
    MemoryImportError,
    MemoryNotFoundError,
    MigrationError,
    ProviderError,
    ProviderNotAvailableError,
    StorageError,
    ValidationError,
)
from highhxpack.models import (
    Conflict,
    ConflictStatus,
    ConsolidationReport,
    EventKind,
    ExportReport,
    ImportReport,
    MemoryEvent,
    MemoryRecord,
    MemoryStats,
    MemoryStatus,
    MemoryType,
    RecallResult,
    Relationship,
    RememberAction,
    RememberResult,
    ScoreBreakdown,
    UserSummary,
)
from highhxpack.providers.base import LLMProvider
from highhxpack.providers.custom import CallableProvider
from highhxpack.retrieval.ranking import ScoringConfig
from highhxpack.retrieval.retriever import Strategy
from highhxpack.storage.base import StorageBackend

__all__ = [
    "CallableEmbedding",
    "CallableProvider",
    "Config",
    "ConfigurationError",
    "Conflict",
    "ConflictNotFoundError",
    "ConflictPolicy",
    "ConflictStatus",
    "ConsolidationReport",
    "EmbeddingError",
    "EmbeddingProvider",
    "EventKind",
    "ExportError",
    "ExportReport",
    "ExtractionError",
    "HashingEmbedding",
    "HighHXPackError",
    "ImportReport",
    "LLMProvider",
    "Memory",
    "MemoryEvent",
    "MemoryImportError",
    "MemoryNotFoundError",
    "MemoryRecord",
    "MemoryStats",
    "MemoryStatus",
    "MemoryType",
    "MigrationError",
    "ProviderError",
    "ProviderNotAvailableError",
    "RecallResult",
    "Relationship",
    "RememberAction",
    "RememberResult",
    "ScoreBreakdown",
    "ScoringConfig",
    "StorageBackend",
    "StorageError",
    "Strategy",
    "UserSummary",
    "ValidationError",
    "__version__",
]
