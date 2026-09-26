"""Exception hierarchy for HighHXPack.

Every exception raised deliberately by the package derives from
:class:`HighHXPackError`, so applications can catch a single type.  Each error
carries three pieces of information:

* ``message`` - what went wrong,
* ``reason``  - why it happened (optional),
* ``hint``    - how to fix it (optional).

``str(error)`` renders all three so that the CLI and log output are helpful
without extra formatting code.
"""

from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "ConflictNotFoundError",
    "EmbeddingError",
    "ExportError",
    "ExtractionError",
    "HighHXPackError",
    "MemoryImportError",
    "MemoryNotFoundError",
    "MigrationError",
    "ProviderError",
    "ProviderNotAvailableError",
    "StorageError",
    "ValidationError",
]


class HighHXPackError(Exception):
    """Base class for all HighHXPack errors."""

    def __init__(self, message: str, *, reason: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.hint = hint

    def __str__(self) -> str:
        parts = [self.message]
        if self.reason:
            parts.append(f"Reason: {self.reason}")
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        return "\n".join(parts)


class ConfigurationError(HighHXPackError):
    """Invalid or inconsistent configuration."""


class ValidationError(HighHXPackError, ValueError):
    """User-supplied input failed validation."""


class StorageError(HighHXPackError):
    """The storage backend failed to read or write data."""


class MigrationError(StorageError):
    """A schema migration could not be applied."""


class MemoryNotFoundError(HighHXPackError, LookupError):
    """No memory exists with the requested identifier."""

    def __init__(self, memory_id: str, *, hint: str | None = None):
        super().__init__(
            f"Memory not found: {memory_id!r}",
            hint=hint or "List memories with `memory.list(user_id)` or `highhxpack memories`.",
        )
        self.memory_id = memory_id


class ConflictNotFoundError(HighHXPackError, LookupError):
    """No conflict exists with the requested identifier."""

    def __init__(self, conflict_id: str):
        super().__init__(
            f"Conflict not found: {conflict_id!r}",
            hint="List conflicts with `memory.conflicts()` or `highhxpack conflicts`.",
        )
        self.conflict_id = conflict_id


class ProviderError(HighHXPackError):
    """An LLM or embedding provider failed."""


class ProviderNotAvailableError(ProviderError):
    """An optional provider was requested but its dependency is missing."""


class EmbeddingError(ProviderError):
    """Embedding generation failed or returned malformed vectors."""


class ExtractionError(HighHXPackError):
    """Automatic memory extraction failed."""


class ExportError(HighHXPackError):
    """Exporting memories failed."""


class MemoryImportError(HighHXPackError):
    """Importing memories failed.

    Named ``MemoryImportError`` rather than ``ImportError`` so that it never
    shadows Python's built-in :class:`ImportError`.
    """
