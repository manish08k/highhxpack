"""Turn free text into candidate memories.

Two extractors are provided:

* :class:`RuleBasedExtractor` (default) - deterministic, offline and fast.  It
  splits text into clauses and keeps only clauses that match a known pattern
  for facts, preferences, goals, decisions, relationships or events.  Anything
  else (questions, greetings, small talk) is dropped, so not every sentence is
  stored.
* :class:`LLMExtractor` - asks a configured :class:`LLMProvider` for structured
  output and validates it strictly.  Invalid output raises
  :class:`ExtractionError`; nothing is invented to fill gaps.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from highhxpack.exceptions import ExtractionError, HighHXPackError
from highhxpack.extraction.entities import Entity, extract_entities
from highhxpack.extraction.events import match_event
from highhxpack.extraction.facts import match_decision, match_fact, match_goal, match_relationship
from highhxpack.extraction.patterns import ClauseMatch, clean_clause, expand_contractions
from highhxpack.extraction.preferences import match_preference
from highhxpack.providers.base import LLMProvider
from highhxpack.utils.text import words
from highhxpack.utils.validation import (
    validate_content,
    validate_memory_type,
    validate_unit_interval,
)

#: The graph node that represents the memory owner in extracted relationships.
USER_NODE = "user"
MAX_EXTRACTED_PER_CALL = 50
_MIN_WORDS = 3

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|;\s*")
_CLAUSE_SPLIT_RE = re.compile(
    r",?\s+(?:but|although|though|whereas|however)\s+(?=(?:i|we|my|user|the\s+user)\b)"
    r"|,?\s+and\s+(?=(?:i|we|my)\b)"
    r"|,\s+(?=(?:i|we|my)\s)",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^(?:what|who|whom|whose|where|when|why|how|which|is|are|am|do|does|did|can|could|would"
    r"|should|will|shall|may|might)\b",
    re.IGNORECASE,
)
_FILLER = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "sure",
        "yes",
        "no",
        "cool",
        "great",
        "nice",
        "bye",
        "goodbye",
        "lol",
        "hmm",
        "got it",
        "sounds good",
    }
)
_MATCHERS = (
    match_relationship,
    match_goal,
    match_decision,
    match_preference,
    match_event,
    match_fact,
)


@dataclass(frozen=True, slots=True)
class ExtractedRelation:
    source: str
    relation: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExtractedMemory:
    """A candidate memory produced by an extractor (not yet stored)."""

    content: str
    memory_type: str
    confidence: float
    importance: float | None = None
    relations: tuple[ExtractedRelation, ...] = ()
    entities: tuple[Entity, ...] = ()


class Extractor(ABC):
    """Interface for anything that turns text into :class:`ExtractedMemory` objects."""

    @abstractmethod
    def extract(self, text: str) -> list[ExtractedMemory]: ...


def split_clauses(text: str) -> list[str]:
    """Split text into sentences, then into independent first-person clauses."""
    clauses: list[str] = []
    for raw_sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        is_question = sentence.rstrip().endswith("?")
        for part in _CLAUSE_SPLIT_RE.split(sentence):
            cleaned = clean_clause(part)
            if cleaned:
                clauses.append(cleaned + ("?" if is_question else ""))
    return clauses


def is_memorable_candidate(clause: str) -> bool:
    """Cheap pre-filter: reject questions, fillers and fragments."""
    stripped = clause.strip()
    if stripped.endswith("?") or _QUESTION_START_RE.match(stripped):
        return False
    lowered = stripped.lower().strip(" !.")
    if lowered in _FILLER:
        return False
    return len(words(stripped)) >= _MIN_WORDS


def classify(content: str) -> ClauseMatch | None:
    """Return the first matching pattern for ``content``, or ``None``."""
    clause = expand_contractions(clean_clause(content))
    for matcher in _MATCHERS:
        result = matcher(clause)
        if result is not None:
            return result
    return None


def classify_memory_type(content: str, default: str = "fact") -> str:
    """Best-effort memory type for explicitly stored content."""
    match = classify(content)
    return match.memory_type if match else default


def relations_for(match: ClauseMatch) -> tuple[ExtractedRelation, ...]:
    if match.relation is None or match.obj is None:
        return ()
    metadata = {"context": match.obj.context} if match.obj.context else {}
    return (ExtractedRelation(USER_NODE, match.relation, match.obj.target, metadata),)


class RuleBasedExtractor(Extractor):
    """Deterministic pattern-based extractor.  See the module docstring."""

    def extract(self, text: str) -> list[ExtractedMemory]:
        results: list[ExtractedMemory] = []
        seen: set[str] = set()
        for clause in split_clauses(text):
            if not is_memorable_candidate(clause):
                continue
            match = classify(clause)
            if match is None:
                continue
            content = clause[0].upper() + clause[1:]
            key = content.casefold()
            if key in seen:
                continue
            seen.add(key)
            results.append(
                ExtractedMemory(
                    content=content,
                    memory_type=match.memory_type,
                    confidence=match.confidence,
                    relations=relations_for(match),
                    entities=tuple(extract_entities(content)),
                )
            )
            if len(results) >= MAX_EXTRACTED_PER_CALL:
                break
        return results


_LLM_SYSTEM_PROMPT = """You extract long-term memories about a user from text they wrote.
Return ONLY a JSON object: {"memories": [...]}. Each memory is an object with:
- "content": one short self-contained statement, in the user's own words where possible
- "memory_type": one of fact, preference, event, goal, relationship, decision, knowledge
- "confidence": number 0-1, how explicitly the text states it
- "importance": number 0-1, how useful it is to remember long-term
- "relations": optional list of {"source": "user", "relation": "snake_case_verb", "target": "..."}
Only include information that is explicitly stated and worth remembering. Never guess.
Skip greetings, questions and small talk. Return {"memories": []} if nothing qualifies."""


class LLMExtractor(Extractor):
    """Extract memories with an LLM.  Output is validated field by field."""

    def __init__(self, provider: LLMProvider, *, max_input_chars: int = 16_000):
        self.provider = provider
        self.max_input_chars = max_input_chars

    def extract(self, text: str) -> list[ExtractedMemory]:
        if len(text) > self.max_input_chars:
            raise ExtractionError(
                f"Text is too long for LLM extraction ({len(text)} characters).",
                hint=f"Split it into chunks of at most {self.max_input_chars} characters.",
            )
        try:
            raw = self.provider.complete(text, system=_LLM_SYSTEM_PROMPT, json_output=True)
        except HighHXPackError as exc:
            raise ExtractionError(
                f"LLM extraction failed using {self.provider.name}.", reason=str(exc)
            ) from exc
        return parse_llm_output(raw)


def parse_llm_output(raw: str) -> list[ExtractedMemory]:
    """Validate an LLM JSON response.  Invalid items are skipped; invalid JSON raises."""
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ExtractionError(
            "The LLM did not return valid JSON.",
            reason=str(exc),
            hint="Use a model that supports JSON output, or the rule-based extractor.",
        ) from exc
    items = data.get("memories") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ExtractionError("The LLM response has no 'memories' list.")
    results: list[ExtractedMemory] = []
    for item in items[:MAX_EXTRACTED_PER_CALL]:
        memory = _parse_item(item)
        if memory is not None:
            results.append(memory)
    return results


def _parse_item(item: object) -> ExtractedMemory | None:
    if not isinstance(item, dict):
        return None
    try:
        content = validate_content(item.get("content"))
        memory_type = validate_memory_type(str(item.get("memory_type", "fact")).lower())
        confidence = validate_unit_interval(item.get("confidence", 0.7), field="confidence")
        importance_raw = item.get("importance")
        importance = (
            validate_unit_interval(importance_raw, field="importance")
            if importance_raw is not None
            else None
        )
    except HighHXPackError:
        return None
    relations: list[ExtractedRelation] = []
    raw_relations = item.get("relations") or []
    if isinstance(raw_relations, list):
        for rel in raw_relations[:10]:
            if not isinstance(rel, dict):
                continue
            source, relation, target = rel.get("source"), rel.get("relation"), rel.get("target")
            if all(isinstance(v, str) and v.strip() for v in (source, relation, target)):
                relations.append(
                    ExtractedRelation(
                        str(source).strip(), str(relation).strip(), str(target).strip()
                    )
                )
    return ExtractedMemory(
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        importance=importance,
        relations=tuple(relations),
        entities=tuple(extract_entities(content)),
    )
