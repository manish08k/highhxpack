"""LangChain retriever adapter.

Requires ``pip install "highhxpack[langchain]"`` (``langchain-core``)::

    from highhxpack.integrations.langchain import create_retriever

    retriever = create_retriever(memory, user_id="alice", k=5)
    docs = retriever.invoke("What does the user prefer?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from highhxpack.exceptions import ProviderNotAvailableError

if TYPE_CHECKING:
    from highhxpack.core.memory import Memory
    from highhxpack.models.result import RecallResult


def _require() -> Any:
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:
        raise ProviderNotAvailableError(
            "The LangChain integration requires 'langchain-core'.",
            hint='Install it with: pip install "highhxpack[langchain]"',
        ) from exc
    return Document, BaseRetriever


def to_metadata(result: RecallResult) -> dict[str, Any]:
    m = result.memory
    return {
        "id": m.id,
        "user_id": m.user_id,
        "memory_type": m.memory_type,
        "score": result.score,
        "importance": m.importance,
        "confidence": m.confidence,
        "created_at": m.created_at.isoformat(),
        "has_conflict": result.has_conflict,
    }


def create_retriever(memory: Memory, user_id: str, *, k: int = 5, min_score: float = 0.0) -> Any:
    """Return a ``langchain_core`` ``BaseRetriever`` backed by ``memory.recall``."""
    document_cls, base_cls = _require()

    class HighHXPackRetriever(base_cls):  # type: ignore[misc, valid-type]
        """Recalls a user's memories as LangChain documents."""

        model_config = {"arbitrary_types_allowed": True}  # noqa: RUF012 - pydantic config
        memory_store: Any
        user: str
        top_k: int = 5
        threshold: float = 0.0

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
            results = self.memory_store.recall(
                self.user, query, limit=self.top_k, min_score=self.threshold
            )
            return [document_cls(page_content=r.content, metadata=to_metadata(r)) for r in results]

    return HighHXPackRetriever(memory_store=memory, user=user_id, top_k=k, threshold=min_score)
