"""Framework-agnostic helpers for putting memories into prompts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from highhxpack.models.result import RecallResult, RememberResult
from highhxpack.utils.validation import MAX_QUERY_LENGTH

if TYPE_CHECKING:
    from highhxpack.core.memory import Memory

DEFAULT_CONTEXT_CHARS = 2_000
DEFAULT_HEADER = "Relevant memories about the user:"


def format_memories(
    results: Sequence[RecallResult],
    *,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
    header: str = DEFAULT_HEADER,
    include_type: bool = True,
) -> str:
    """Render recalled memories as a bullet list that fits in ``max_chars``.

    Memories with an unresolved conflict are marked so the model does not treat
    them as settled facts.  Returns ``""`` when there is nothing to include.
    """
    if not results:
        return ""
    lines = [header]
    used = len(header)
    for result in results:
        prefix = f"[{result.memory.memory_type}] " if include_type else ""
        note = " (conflicts with another memory)" if result.has_conflict else ""
        line = f"- {prefix}{result.content}{note}"
        if used + 1 + len(line) > max_chars:
            break
        lines.append(line)
        used += 1 + len(line)
    return "\n".join(lines) if len(lines) > 1 else ""


class ChatMemory:
    """Minimal memory loop for chatbots.

    Example::

        chat = ChatMemory(Memory(), user_id="alice")
        system_prompt = base_prompt + "\\n\\n" + chat.context(user_message)
        ...
        chat.observe(user_message)   # extract and store anything worth remembering
    """

    def __init__(
        self,
        memory: Memory,
        user_id: str,
        *,
        limit: int = 8,
        max_chars: int = DEFAULT_CONTEXT_CHARS,
        min_score: float = 0.0,
    ):
        self.memory = memory
        self.user_id = user_id
        self.limit = limit
        self.max_chars = max_chars
        self.min_score = min_score

    def context(self, query: str) -> str:
        """Prompt-ready text with the memories most relevant to ``query``."""
        query = query.strip()[:MAX_QUERY_LENGTH]  # long messages must not break the chat
        if not query:
            return ""
        results = self.memory.recall(
            self.user_id, query, limit=self.limit, min_score=self.min_score
        )
        return format_memories(results, max_chars=self.max_chars)

    def observe(self, message: str, *, source: str = "conversation") -> list[RememberResult]:
        """Extract memorable statements from a user message and store them."""
        if not message.strip():
            return []
        return self.memory.ingest(self.user_id, message, source=source)
