"""Recognition of likes, dislikes and preferences."""

from __future__ import annotations

import re

from highhxpack.extraction.patterns import SUBJECT, ClauseMatch, object_phrase

_ADVERB = (
    r"(?:(?:really|strongly|generally|usually|mostly|definitely|also|now|actually|still"
    r"|currently)\s+)*"
)

_PREFER_RE = re.compile(
    rf"^{SUBJECT}\s+{_ADVERB}(?:prefers?|would\s+rather\s+use|'d\s+rather\s+use)\s+(?P<obj>.+)$",
    re.IGNORECASE,
)
_LIKE_RE = re.compile(
    rf"^{SUBJECT}\s+{_ADVERB}(?:likes?|loves?|enjoys?|adores?|favou?rs?)\s+(?P<obj>.+)$",
    re.IGNORECASE,
)
_FAN_RE = re.compile(
    rf"^{SUBJECT}\s+(?:am|'m|is|are)\s+(?:a\s+(?:big\s+|huge\s+)?fan\s+of|into|"
    r"interested\s+in|passionate\s+about)\s+(?P<obj>.+)$",
    re.IGNORECASE,
)
_DISLIKE_RE = re.compile(
    rf"^{SUBJECT}\s+{_ADVERB}(?:dislikes?|hates?|can't\s+stand|cannot\s+stand|detests?|"
    r"(?:do|does)\s+not\s+(?:like|enjoy)|(?:don't|doesn't)\s+(?:like|enjoy)|avoids?)\s+"
    r"(?P<obj>.+)$",
    re.IGNORECASE,
)
_FAVORITE_RE = re.compile(
    r"^(?:my|the\s+user's|user's)\s+(?:all[-\s]time\s+)?favou?rite\s+(?P<attr>[\w\s\-]+?)\s+"
    r"(?:is|are)\s+(?P<obj>.+)$",
    re.IGNORECASE,
)


def match_preference(clause: str) -> ClauseMatch | None:
    """Classify ``clause`` as a preference, returning a graph relation if possible."""
    if m := _FAVORITE_RE.match(clause):
        return ClauseMatch(
            "preference", 0.9, "favorite_" + _slug(m["attr"]), object_phrase(m["obj"])
        )
    if m := _PREFER_RE.match(clause):
        return ClauseMatch("preference", 0.9, "prefers", object_phrase(m["obj"]))
    if m := _DISLIKE_RE.match(clause):
        return ClauseMatch("preference", 0.85, "dislikes", object_phrase(m["obj"]))
    if m := _LIKE_RE.match(clause):
        return ClauseMatch("preference", 0.85, "likes", object_phrase(m["obj"]))
    if m := _FAN_RE.match(clause):
        return ClauseMatch("preference", 0.8, "interested_in", object_phrase(m["obj"]))
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "thing"
