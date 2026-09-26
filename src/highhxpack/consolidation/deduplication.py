"""Duplicate detection.

Two levels:

1. **Exact** - identical :func:`~highhxpack.utils.hashing.content_hash` (case,
   punctuation and whitespace are ignored).
2. **Near** - the Jaccard similarity of *canonical* token sets is at least
   ``threshold``.  Canonicalization stems words, drops subject words and maps
   preference verbs to one token, so "User likes Python", "I love Python" and
   "I prefer Python" are duplicates.  Statements with opposite polarity
   ("I like X" / "I don't like X") are never duplicates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from highhxpack.utils.text import jaccard, stem, words

_SYNONYMS: dict[str, str] = {}
for _canonical, _variants in {
    "prefer": "prefer prefers preferred like likes liked love loves loved enjoy enjoys enjoyed "
    "favor favors favour favours favorite favourite fond adore adores",
    "dislike": "dislike dislikes disliked hate hates hated detest detests loathe loathes",
    "use": "use uses used using utilize utilizes utilise",
    "work": "work works worked working",
}.items():
    for _variant in _variants.split():
        _SYNONYMS[_variant] = _canonical

_SUBJECT_WORDS = frozenset({"user", "user's", "i", "me", "my", "we", "our", "the"})
_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "can't",
        "cannot",
        "nor",
        "neither",
        "without",
        "longer",
        "dislike",
        "dislikes",
        "disliked",
        "hate",
        "hates",
        "hated",
        "stopped",
    }
)
_FILLER = frozenset(
    "a an and are as at be been being by for from has have i'm i've in is it of on or really "
    "so that the this to very was were will with".split()
)

DEFAULT_THRESHOLD = 0.85


def canonical_tokens(text: str) -> frozenset[str]:
    """Token set used for near-duplicate comparison."""
    result: set[str] = set()
    for word in words(text):
        if word in _SUBJECT_WORDS or word in _FILLER:
            continue
        canonical = _SYNONYMS.get(word)
        if canonical is None and word not in _NEGATIONS:
            canonical = stem(word)
        elif canonical is None:
            continue
        result.add(canonical)
    return frozenset(result)


def is_negated(text: str) -> bool:
    return any(w in _NEGATIONS for w in words(text))


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    memory_id: str
    similarity: float


def similarity(a: str, b: str) -> float:
    """Canonical token Jaccard similarity; 0 when polarity differs."""
    if is_negated(a) != is_negated(b):
        return 0.0
    return jaccard(canonical_tokens(a), canonical_tokens(b))


def is_near_duplicate(a: str, b: str, *, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Whether two texts are near-duplicates at ``threshold``."""
    return similarity(a, b) >= threshold


def find_duplicate_pairs(
    items: Sequence[tuple[str, str]], *, threshold: float = DEFAULT_THRESHOLD
) -> list[tuple[str, str, float]]:
    """Find near-duplicate pairs among ``(id, content)`` items.

    Uses *prefix filtering*: with tokens ordered from rarest to most common, two
    sets with Jaccard >= t must share a token within the first
    ``len - ceil(t * len) + 1`` tokens of each.  Only such pairs are compared,
    which avoids the quadratic cost on large collections.
    """
    tokenized = [(item_id, canonical_tokens(text), is_negated(text)) for item_id, text in items]
    frequency: dict[str, int] = {}
    for _, tokens, _ in tokenized:
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
    index: dict[str, list[int]] = {}
    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[int, int]] = set()
    for position, (item_id, tokens, negated) in enumerate(tokenized):
        if not tokens:
            continue
        ordered = sorted(tokens, key=lambda t: (frequency[t], t))
        prefix_length = len(ordered) - math.ceil(threshold * len(ordered)) + 1
        for token in ordered[:prefix_length]:
            for other in index.get(token, ()):
                if (other, position) in seen:
                    continue
                seen.add((other, position))
                other_id, other_tokens, other_negated = tokenized[other]
                if other_negated != negated:
                    continue
                score = jaccard(tokens, other_tokens)
                if score >= threshold:
                    pairs.append((other_id, item_id, score))
            index.setdefault(token, []).append(position)
    return pairs


def best_duplicate(
    content: str,
    candidates: Iterable[tuple[str, str]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> DuplicateMatch | None:
    """Return the most similar near-duplicate among ``(id, content)`` candidates."""
    best: DuplicateMatch | None = None
    for memory_id, other in candidates:
        score = similarity(content, other)
        if score >= threshold and (best is None or score > best.similarity):
            best = DuplicateMatch(memory_id, score)
    return best
