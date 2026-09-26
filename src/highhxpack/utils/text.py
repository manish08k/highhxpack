"""Text normalization and tokenization shared by indexing, deduplication,
embeddings and extraction.

The tokenizer is deliberately simple and deterministic: lower-casing, a
Unicode-aware word pattern that keeps technical tokens such as ``c++``,
``c#``, ``node.js`` and ``gpt-4`` intact, an English stop-word list, and a light
suffix-stripping stemmer.  It is not a linguistic parser; its job is to make
"prefer", "prefers" and "preferred" match each other.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_TOKEN_RE = re.compile(r"[^\W_]+(?:[.\-'][^\W_]+)*[+#]*")
_HASH_STRIP_RE = re.compile(r"[^\w+#]+")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all also am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have haven't
    having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself just let's me more most mustn't my myself no
    nor not of off on once only or other ought our ours ourselves out over own same shan't she
    she'd she'll she's should shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're they've this those through
    to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when
    when's where where's which while who who's whom why why's will with won't would wouldn't you
    you'd you'll you're you've your yours yourself yourselves really quite
    """.split()
)


def normalize_text(text: str) -> str:
    """Apply Unicode NFKC normalization, unify apostrophes and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'")
    return " ".join(text.split())


def normalize_for_hash(text: str) -> str:
    """Normalize text for exact-duplicate detection.

    Case, punctuation (except ``+`` and ``#``) and whitespace differences are
    ignored, so "I use Python." and "i use python" hash identically.
    """
    lowered = normalize_text(text).casefold()
    return " ".join(_HASH_STRIP_RE.sub(" ", lowered).split())


def words(text: str) -> list[str]:
    """Split text into lower-cased word tokens without removing stop words."""
    return _TOKEN_RE.findall(normalize_text(text).casefold())


def stem(word: str) -> str:
    """Strip common English inflectional suffixes.

    Tokens containing digits or symbols (``c++``, ``gpt-4``) are returned
    unchanged.  The stemmer is intentionally conservative.
    """
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) <= 2 or not word.isalpha():
        return word
    for suffix, replacement in (("ies", "y"), ("ied", "y")):
        if word.endswith(suffix) and len(word) > 4:
            return word[: -len(suffix)] + replacement
    stemmed = word
    if stemmed.endswith("ing") and len(stemmed) > 4 and _has_vowel(stemmed[:-3]):
        stemmed = stemmed[:-3]
    elif (
        (
            stemmed.endswith("ed")
            and not stemmed.endswith("eed")
            and len(stemmed) > 3
            and _has_vowel(stemmed[:-2])
        )
        or stemmed.endswith("sses")
        or (stemmed.endswith("es") and len(stemmed) > 4 and stemmed[-3] in "sxz")
    ):
        stemmed = stemmed[:-2]
    elif stemmed.endswith("s") and not stemmed.endswith(("ss", "us", "is")) and len(stemmed) > 3:
        stemmed = stemmed[:-1]
    if stemmed.endswith("e") and len(stemmed) > 2:
        stemmed = stemmed[:-1]
    if (
        len(stemmed) > 3
        and stemmed[-1] == stemmed[-2]
        and stemmed[-1] not in "aeioulsz"
        and stemmed != word
    ):
        stemmed = stemmed[:-1]
    return stemmed


def _has_vowel(text: str) -> bool:
    return any(ch in "aeiouy" for ch in text)


def tokenize(text: str) -> list[str]:
    """Return stemmed, stop-word-filtered index terms in document order."""
    return [stem(token) for token in words(text) if token not in STOPWORDS]


def term_frequencies(text: str) -> dict[str, int]:
    """Return a mapping of index term to its frequency in ``text``."""
    return dict(Counter(tokenize(text)))


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    """Return the Jaccard similarity of two sets (0.0 when both are empty)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
