"""Shared building blocks for the rule-based extraction patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass

from highhxpack.extraction.entities import extract_entities

#: Grammatical subjects that refer to the person whose memory is stored.
SUBJECT = r"(?:i|we|user|the\s+user)"

_CONTEXT_SPLIT_RE = re.compile(
    r"\s+(?:for|when|while|because|since|as|in\s+order\s+to)\s+", re.IGNORECASE
)
_HEAD_STOP_RE = re.compile(
    r"\s+(?:for|to|in|at|on|with|from|by|about|because|since|and|but|or|so|as|when|while"
    r"|every|each|all|over|than|that|which)\s+",
    re.IGNORECASE,
)
_TRAILING_RE = re.compile(r"[\s,;:!?.]+$")
#: Words that qualify *when* something is true, not *what*; "I prefer Rust now"
#: makes the same claim as "I prefer Rust".
_TRAILING_FILLER_RE = re.compile(
    r"(?:[\s,]+(?:now|anymore|any\s+more|nowadays|these\s+days|currently|instead|too|as\s+well"
    r"|though|actually|really|lately|again))+$",
    re.IGNORECASE,
)
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the|some|my|our)\s+", re.IGNORECASE)
_MAX_TARGET_WORDS = 6


@dataclass(frozen=True, slots=True)
class ObjectPhrase:
    """The object of a statement, e.g. ``C++`` in "I prefer C++ for contests".

    ``context`` is the qualifier after "for"/"when"/... (``"contests"``).
    """

    target: str
    context: str | None = None


@dataclass(frozen=True, slots=True)
class ClauseMatch:
    """A clause recognized as memorable, with an optional ``user --relation--> target`` edge."""

    memory_type: str
    confidence: float
    relation: str | None = None
    obj: ObjectPhrase | None = None


_CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in (
        (r"\b(i|we|you|they)'ve\b", r"\1 have"),
        (r"\bi'm\b", "i am"),
        (r"\b(we|you|they)'re\b", r"\1 are"),
        (r"\b(i|we|you|they)'d\b", r"\1 would"),
        (r"\b(i|we|you|they)'ll\b", r"\1 will"),
        (r"\bcan't\b", "cannot"),
        (r"\bwon't\b", "will not"),
        (r"\b(do|does|did|is|are|was|were|have|has)n't\b", r"\1 not"),
    )
)


def expand_contractions(text: str) -> str:
    """Expand common English contractions ("I've" -> "i have") for pattern matching."""
    text = text.replace("\u2019", "'")
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def clean_clause(text: str) -> str:
    """Strip surrounding whitespace and trailing punctuation (keeping ``+``/``#``)."""
    return _TRAILING_RE.sub("", text.strip())


def object_phrase(text: str) -> ObjectPhrase | None:
    """Split an object phrase into its head (graph target) and qualifying context."""
    text = clean_clause(_TRAILING_FILLER_RE.sub("", clean_clause(text)))
    if not text:
        return None
    parts = _CONTEXT_SPLIT_RE.split(text, maxsplit=1)
    head = parts[0]
    context = (
        clean_clause(_TRAILING_FILLER_RE.sub("", clean_clause(parts[1])))
        if len(parts) > 1
        else None
    )
    entities = extract_entities(head)
    if entities:
        target = entities[0].name
    else:
        head = _HEAD_STOP_RE.split(head, maxsplit=1)[0]
        head = _LEADING_ARTICLE_RE.sub("", head)
        target = " ".join(head.split()[:_MAX_TARGET_WORDS])
    target = clean_clause(target)
    if not target:
        return None
    return ObjectPhrase(target=target, context=context or None)
