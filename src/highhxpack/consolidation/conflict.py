"""Detection and resolution of contradictory memories.

A *claim* is the part of a statement that can be contradicted: a **slot**
(what the statement is about) and a **value**.  Two active memories of the same
user conflict when they fill the same slot with different values, e.g.

* ``prefers:*``           Java   vs  Python        ("I prefer Java" / "I prefer Python")
* ``prefers:web``         Django vs  Flask         (qualified preferences only clash
                                                    within the same context)
* ``sentiment:python``    positive vs negative     ("I like Python" / "I hate Python")
* ``lives_in``            Paris vs Berlin
* ``attr:favorite_color`` blue vs green

Resolution is governed by an explicit :class:`ConflictPolicy`.  HighHXPack never
guesses which statement is true: under ``supersede`` the *newer* statement
becomes the current belief only when its confidence is at least that of the
older one, the older memory is kept with status ``superseded``, and a resolved
conflict record explains why.  Otherwise both stay active and an *open*
conflict is recorded for the application or user to resolve.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from highhxpack.exceptions import ValidationError
from highhxpack.extraction.patterns import SUBJECT, clean_clause, expand_contractions, object_phrase
from highhxpack.models.memory import MemoryRecord
from highhxpack.utils.text import normalize_for_hash, stem, words


class ConflictPolicy(StrEnum):
    #: Newer, at-least-as-confident information supersedes older information.
    SUPERSEDE = "supersede"
    #: Keep both memories active and record an open conflict.
    KEEP_BOTH = "keep_both"


def parse_policy(value: str | ConflictPolicy) -> ConflictPolicy:
    try:
        return ConflictPolicy(str(getattr(value, "value", value)))
    except ValueError as exc:
        raise ValidationError(
            f"Unknown conflict policy: {value!r}", hint="Use 'supersede' or 'keep_both'."
        ) from exc


class ConflictKind(StrEnum):
    #: Same slot, different value ("prefers Java" -> "prefers Python").
    CHANGED_VALUE = "changed_value"
    #: Same subject, opposite polarity ("likes X" -> "dislikes X").
    CONTRADICTION = "contradiction"


@dataclass(frozen=True, slots=True)
class Claim:
    slot: str
    value: str
    #: True when the slot encodes polarity (value is "positive"/"negative").
    polar: bool = False


_ADVERB = (
    r"(?:(?:really|strongly|generally|usually|mostly|definitely|also|now|actually|still"
    r"|currently)\s+)*"
)
_FAVORITE_RE = re.compile(
    r"^(?:my|the\s+user's|user's)\s+(?:all[-\s]time\s+)?favou?rite\s+(?P<attr>[\w\s\-]{1,40}?)\s+"
    r"(?:is|are)\s+(?P<obj>.+)$",
    re.IGNORECASE,
)
_PREFER_RE = re.compile(rf"^{SUBJECT}\s+{_ADVERB}prefers?\s+(?P<obj>.+)$", re.IGNORECASE)
_NEGATIVE_SENTIMENT_RE = re.compile(
    rf"^{SUBJECT}\s+{_ADVERB}(?:dislikes?|hates?|cannot\s+stand|detests?|(?:do|does)\s+not\s+"
    r"(?:like|love|enjoy)|no\s+longer\s+(?:like|likes|love|loves|enjoy|enjoys))\s+(?P<obj>.+)$",
    re.IGNORECASE,
)
_POSITIVE_SENTIMENT_RE = re.compile(
    rf"^{SUBJECT}\s+{_ADVERB}(?:likes?|loves?|enjoys?|adores?)\s+(?P<obj>.+)$", re.IGNORECASE
)
_NEGATIVE_USE_RE = re.compile(
    rf"^{SUBJECT}\s+(?:(?:do|does)\s+not\s+use|no\s+longer\s+uses?|stopped\s+using|never\s+uses?"
    r"|quit\s+using)\s+(?P<obj>.+)$",
    re.IGNORECASE,
)
_POSITIVE_USE_RE = re.compile(
    rf"^{SUBJECT}\s+{_ADVERB}(?:uses?|am\s+using|is\s+using|are\s+using)\s+(?P<obj>.+)$",
    re.IGNORECASE,
)
_SINGLE_VALUED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (slot, re.compile(pattern, re.IGNORECASE))
    for slot, pattern in (
        (
            "lives_in",
            rf"^{SUBJECT}\s+{_ADVERB}(?:live|lives|am\s+living|reside|resides)\s+in\s+(?P<obj>.+)$",
        ),
        (
            "works_at",
            rf"^{SUBJECT}\s+{_ADVERB}(?:work|works|am\s+working)\s+(?:at|for)\s+(?P<obj>.+)$",
        ),
        ("name", r"^(?:my\s+name\s+is|call\s+me)\s+(?P<obj>.+)$"),
        ("age", rf"^{SUBJECT}\s+(?:am|is)\s+(?P<obj>\d{{1,3}})\s+years?\s+old$"),
    )
)
_ATTRIBUTE_RE = re.compile(
    r"^(?:my|the\s+user's|user's)\s+(?P<attr>[a-z][\w\-]*(?:\s+[a-z][\w\-]*){0,2})\s+"
    r"(?:is|are)\s+(?P<obj>(?!a\s|an\s|the\s|not\s|very\s|so\s).+)$",
    re.IGNORECASE,
)


def _key(text: str) -> str:
    return " ".join(stem(w) for w in words(text)) or normalize_for_hash(text)


def _target(text: str) -> tuple[str, str | None] | None:
    phrase = object_phrase(text)
    if phrase is None:
        return None
    return _key(phrase.target), (_key(phrase.context) if phrase.context else None)


def parse_claim(content: str) -> Claim | None:
    """Extract the contradictable claim of a statement, if it has one."""
    clause = expand_contractions(clean_clause(content))
    if m := _FAVORITE_RE.match(clause):
        target = _target(m["obj"])
        return Claim(f"favorite:{_key(m['attr'])}", target[0]) if target else None
    if m := _PREFER_RE.match(clause):
        target = _target(m["obj"])
        if target is None:
            return None
        return Claim(f"prefers:{target[1] or '*'}", target[0])
    for regex, polarity, prefix in (
        (_NEGATIVE_SENTIMENT_RE, "negative", "sentiment"),
        (_POSITIVE_SENTIMENT_RE, "positive", "sentiment"),
        (_NEGATIVE_USE_RE, "negative", "uses"),
        (_POSITIVE_USE_RE, "positive", "uses"),
    ):
        if m := regex.match(clause):
            target = _target(m["obj"])
            return Claim(f"{prefix}:{target[0]}", polarity, polar=True) if target else None
    for slot, regex in _SINGLE_VALUED:
        if m := regex.match(clause):
            target = _target(m["obj"])
            return Claim(slot, target[0]) if target else None
    if m := _ATTRIBUTE_RE.match(clause):
        target = _target(m["obj"])
        return Claim(f"attr:{_key(m['attr'])}", target[0]) if target else None
    return None


@dataclass(frozen=True, slots=True)
class DetectedConflict:
    existing: MemoryRecord
    kind: ConflictKind


def detect_conflicts(claim: Claim, existing: Iterable[MemoryRecord]) -> list[DetectedConflict]:
    """Return memories among ``existing`` whose claim on ``claim.slot`` differs."""
    found: list[DetectedConflict] = []
    for memory in existing:
        other = parse_claim(memory.content)
        if other is None or other.slot != claim.slot or other.value == claim.value:
            continue
        kind = ConflictKind.CONTRADICTION if claim.polar else ConflictKind.CHANGED_VALUE
        found.append(DetectedConflict(memory, kind))
    return found


def should_supersede(
    policy: ConflictPolicy, *, new_confidence: float, old_confidence: float
) -> bool:
    """Whether the newer memory replaces the older one as the current belief."""
    return policy == ConflictPolicy.SUPERSEDE and new_confidence >= old_confidence
