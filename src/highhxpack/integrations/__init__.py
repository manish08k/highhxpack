"""Integrations with AI frameworks.

:mod:`~highhxpack.integrations.generic` has no dependencies.  The LangChain and
LlamaIndex adapters import their frameworks lazily and raise
:class:`~highhxpack.exceptions.ProviderNotAvailableError` with install
instructions when the optional dependency is missing.
"""

from highhxpack.integrations.generic import ChatMemory, format_memories

__all__ = ["ChatMemory", "format_memories"]
