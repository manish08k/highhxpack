"""LlamaIndex retriever adapter.

Requires ``pip install "highhxpack[llamaindex]"`` (``llama-index-core``)::

    from highhxpack.integrations.llamaindex import create_retriever

    retriever = create_retriever(memory, user_id="alice", k=5)
    nodes = retriever.retrieve("What does the user prefer?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from highhxpack.exceptions import ProviderNotAvailableError
from highhxpack.integrations.langchain import to_metadata

if TYPE_CHECKING:
    from highhxpack.core.memory import Memory


def _require() -> Any:
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, TextNode
    except ImportError as exc:
        raise ProviderNotAvailableError(
            "The LlamaIndex integration requires 'llama-index-core'.",
            hint='Install it with: pip install "highhxpack[llamaindex]"',
        ) from exc
    return BaseRetriever, NodeWithScore, TextNode


def create_retriever(memory: Memory, user_id: str, *, k: int = 5, min_score: float = 0.0) -> Any:
    """Return a LlamaIndex ``BaseRetriever`` backed by ``memory.recall``."""
    base_cls, node_with_score_cls, text_node_cls = _require()

    class HighHXPackRetriever(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self._memory = memory
            self._user = user_id

        def _retrieve(self, query_bundle: Any) -> list[Any]:
            results = self._memory.recall(
                self._user, query_bundle.query_str, limit=k, min_score=min_score
            )
            return [
                node_with_score_cls(
                    node=text_node_cls(text=r.content, id_=r.id, metadata=to_metadata(r)),
                    score=r.score,
                )
                for r in results
            ]

    return HighHXPackRetriever()
