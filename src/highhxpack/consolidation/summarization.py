"""Summaries of related memories.

Summaries are *additional* memories (type ``knowledge``); the memories they
summarize are never modified or removed.  Without an LLM the
:class:`ExtractiveSummarizer` joins the original statements verbatim, so no
information is paraphrased or invented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from highhxpack.exceptions import ProviderError
from highhxpack.extraction.entities import extract_entities
from highhxpack.extraction.patterns import clean_clause
from highhxpack.models.memory import MemoryRecord
from highhxpack.providers.base import LLMProvider
from highhxpack.utils.hashing import content_hash
from highhxpack.utils.validation import MAX_CONTENT_LENGTH

DEFAULT_MIN_GROUP_SIZE = 3
DEFAULT_MAX_SUMMARY_CHARS = 2_000


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, topic: str, contents: Sequence[str]) -> str: ...


class ExtractiveSummarizer(Summarizer):
    """Concatenate distinct statements, most important first."""

    def __init__(self, max_chars: int = DEFAULT_MAX_SUMMARY_CHARS):
        self.max_chars = min(max_chars, MAX_CONTENT_LENGTH)

    def summarize(self, topic: str, contents: Sequence[str]) -> str:
        seen: set[str] = set()
        parts: list[str] = []
        for text in contents:
            key = content_hash(text)
            if key in seen:
                continue
            seen.add(key)
            parts.append(clean_clause(text))
        summary = f"About {topic}: " + "; ".join(parts) + "."
        if len(summary) > self.max_chars:
            summary = summary[: self.max_chars - 3].rstrip() + "..."
        return summary


class LLMSummarizer(Summarizer):
    """Ask an LLM to summarize; the prompt forbids adding information."""

    _SYSTEM = (
        "Summarize the user's memories about the given topic in one or two sentences. "
        "Use only the information given. Do not add facts, opinions or guesses."
    )

    def __init__(self, provider: LLMProvider, max_chars: int = DEFAULT_MAX_SUMMARY_CHARS):
        self.provider = provider
        self.max_chars = min(max_chars, MAX_CONTENT_LENGTH)

    def summarize(self, topic: str, contents: Sequence[str]) -> str:
        bullet_list = "\n".join(f"- {c}" for c in contents)
        text = self.provider.complete(
            f"Topic: {topic}\nMemories:\n{bullet_list}", system=self._SYSTEM
        ).strip()
        if not text:
            raise ProviderError(f"{self.provider.name} returned an empty summary.")
        return text[: self.max_chars]


@dataclass(frozen=True, slots=True)
class MemoryGroup:
    topic: str
    memories: tuple[MemoryRecord, ...]


def group_by_entity(
    memories: Sequence[MemoryRecord], *, min_size: int = DEFAULT_MIN_GROUP_SIZE
) -> list[MemoryGroup]:
    """Group memories that mention the same entity.

    A memory can belong to several groups.  Groups are ordered by size (largest
    first) and then by topic for determinism; memories within a group are ordered
    by importance, then age.
    """
    by_entity: dict[str, tuple[str, list[MemoryRecord]]] = {}
    for memory in memories:
        for entity in extract_entities(memory.content):
            _, members = by_entity.setdefault(entity.key, (entity.name, []))
            members.append(memory)
    groups = [
        MemoryGroup(
            topic=name,
            memories=tuple(sorted(members, key=lambda m: (-m.importance, m.created_at, m.id))),
        )
        for name, members in by_entity.values()
        if len(members) >= min_size
    ]
    groups.sort(key=lambda g: (-len(g.memories), g.topic.casefold()))
    return groups


def summary_key(memory_ids: Sequence[str]) -> str:
    """Stable identifier of a set of memories, used to avoid duplicate summaries."""
    return content_hash(" ".join(sorted(memory_ids)))
