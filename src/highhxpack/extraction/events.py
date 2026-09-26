"""Recognition of events: things that happened (or will happen) at a point in time."""

from __future__ import annotations

import re

from highhxpack.extraction.patterns import SUBJECT, ClauseMatch, object_phrase

_TIME_MARKER_RE = re.compile(
    r"\b(?:yesterday|today|tonight|tomorrow|this\s+(?:morning|afternoon|evening|week|month|year)"
    r"|last\s+(?:night|week|weekend|month|year|monday|tuesday|wednesday|thursday|friday|saturday"
    r"|sunday)|next\s+(?:week|weekend|month|year)|on\s+(?:monday|tuesday|wednesday|thursday"
    r"|friday|saturday|sunday)|in\s+(?:19|20)\d\d|\d{4}-\d{2}-\d{2}|\d+\s+(?:days?|weeks?|"
    r"months?|years?)\s+ago)\b",
    re.IGNORECASE,
)
_PAST_ACTION_RE = re.compile(
    rf"^{SUBJECT}\s+(?:just\s+|recently\s+|finally\s+)?(?:went|visited|attended|started|began|"
    r"finished|completed|joined|left|moved|graduated|met|launched|shipped|released|published|"
    r"got|bought|travel(?:l)?ed|won|lost|passed|failed|presented|married|signed|quit|"
    r"retired|received|adopted|deployed|migrated|switched)\b\s*(?P<obj>.*)$",
    re.IGNORECASE,
)
_FUTURE_EVENT_RE = re.compile(
    rf"^{SUBJECT}\s+(?:have|has|'ve\s+got)\s+(?:a|an|my|the)\s+(?P<obj>(?:meeting|interview|"
    r"appointment|exam|deadline|flight|trip|presentation|call|conference|wedding|birthday)\b.*)$",
    re.IGNORECASE,
)


def has_time_marker(clause: str) -> bool:
    return _TIME_MARKER_RE.search(clause) is not None


def match_event(clause: str) -> ClauseMatch | None:
    if m := _PAST_ACTION_RE.match(clause):
        return ClauseMatch("event", 0.8, None, object_phrase(m["obj"]) if m["obj"] else None)
    if m := _FUTURE_EVENT_RE.match(clause):
        return ClauseMatch("event", 0.8, None, object_phrase(m["obj"]))
    if has_time_marker(clause) and re.match(rf"^{SUBJECT}\b", clause, re.IGNORECASE):
        return ClauseMatch("event", 0.7)
    return None
