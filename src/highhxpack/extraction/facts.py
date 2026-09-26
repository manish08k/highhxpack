"""Recognition of facts about the user, goals, decisions and personal relationships."""

from __future__ import annotations

import re

from highhxpack.extraction.patterns import SUBJECT, ClauseMatch, ObjectPhrase, object_phrase

_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str | None, float], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), relation, confidence)
    for pattern, relation, confidence in (
        (
            rf"^{SUBJECT}\s+(?:use|uses|am\s+using|'m\s+using|have\s+been\s+using|'ve\s+been\s+"
            r"using|mainly\s+use|mostly\s+use|work\s+with|works\s+with|code\s+in|program\s+in"
            r"|write\s+(?:code\s+)?in)\s+(?P<obj>.+)$",
            "uses",
            0.9,
        ),
        (
            rf"^{SUBJECT}\s+(?:work|works|am\s+working|'m\s+working)\s+(?:at|for)\s+(?P<obj>.+)$",
            "works_at",
            0.9,
        ),
        (
            rf"^{SUBJECT}\s+(?:am|'m|is|are)\s+(?:working\s+on|building|developing|creating)"
            r"\s+(?P<obj>.+)$",
            "works_on",
            0.85,
        ),
        (
            rf"^{SUBJECT}\s+(?:work|works)\s+on\s+(?P<obj>.+)$",
            "works_on",
            0.85,
        ),
        (
            rf"^{SUBJECT}\s+(?:live|lives|am\s+living|'m\s+living|reside|resides)\s+in\s+"
            r"(?P<obj>.+)$",
            "lives_in",
            0.9,
        ),
        (rf"^{SUBJECT}\s+(?:am|'m|is|come|comes)\s+from\s+(?P<obj>.+)$", "from", 0.9),
        (
            r"^(?:my\s+name\s+is|call\s+me|i\s+am\s+called|i'm\s+called)\s+(?P<obj>.+)$",
            "named",
            0.95,
        ),
        (rf"^{SUBJECT}\s+(?:am|'m|is)\s+allergic\s+to\s+(?P<obj>.+)$", "allergic_to", 0.95),
        (
            rf"^{SUBJECT}\s+(?:speak|speaks|know|knows|studied|studies|am\s+learning|'m\s+"
            r"learning|is\s+learning)\s+(?P<obj>.+)$",
            "knows",
            0.8,
        ),
        (rf"^{SUBJECT}\s+(?:have|has|own|owns)\s+(?P<obj>.+)$", "has", 0.75),
        (rf"^{SUBJECT}\s+(?:am|'m|is)\s+(?:a|an)\s+(?P<obj>.+)$", "is_a", 0.85),
        (rf"^{SUBJECT}\s+(?:am|'m|is)\s+(?P<obj>\d{{1,3}}\s+years?\s+old)$", None, 0.9),
        (
            r"^(?:my|the\s+user's|user's)\s+(?P<attr>[\w\s\-]+?)\s+(?:is|are)\s+(?P<obj>.+)$",
            None,
            0.85,
        ),
    )
)

_GOAL_RE = re.compile(
    rf"^(?:{SUBJECT}\s+(?:want|wants|plan|plans|hope|hopes|intend|intends|aim|aims|would\s+"
    r"like|'d\s+like|need|needs|am\s+planning|'m\s+planning|am\s+trying|'m\s+trying|am\s+hoping"
    r"|'m\s+hoping|am\s+going|'m\s+going)\s+to|(?:my|our)\s+(?:goal|plan|aim|objective)\s+"
    r"(?:is|are)\s+(?:to\s+)?)\s*(?P<obj>.+)$",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    rf"^{SUBJECT}\s+(?:have\s+|'ve\s+|has\s+)?(?:decided|chose|chosen|agreed|settled)\s+"
    r"(?:to\s+|on\s+|that\s+)?(?P<obj>.+)$"
    rf"|^{SUBJECT}\s+(?:will|'ll|are\s+going\s+to|'re\s+going\s+to)\s+(?:go\s+with|use|switch\s+to|"
    r"adopt|migrate\s+to)\s+(?P<obj2>.+)$",
    re.IGNORECASE,
)
_ROLES = (
    "wife|husband|partner|girlfriend|boyfriend|spouse|son|daughter|child|kid|brother|sister|"
    "sibling|mother|mom|mum|father|dad|parent|grandmother|grandfather|aunt|uncle|cousin|friend|"
    "best\\s+friend|manager|boss|colleague|coworker|co-worker|mentor|teammate|roommate|dog|cat|pet"
)
_RELATIONSHIP_RE = re.compile(
    rf"^(?:my|the\s+user's|user's)\s+(?P<role>{_ROLES})(?:'s\s+name\s+is|\s+is\s+(?:called|named)"
    rf"|\s+is|,)?\s+(?P<name>[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)\b",
)


def match_fact(clause: str) -> ClauseMatch | None:
    for pattern, relation, confidence in _FACT_PATTERNS:
        if m := pattern.match(clause):
            obj = object_phrase(m["obj"])
            if relation is None and "attr" in pattern.groupindex:
                attr = re.sub(r"[^a-z0-9]+", "_", m["attr"].lower()).strip("_")
                return ClauseMatch("fact", confidence, attr or None, obj)
            return ClauseMatch("fact", confidence, relation, obj)
    return None


def match_goal(clause: str) -> ClauseMatch | None:
    if m := _GOAL_RE.match(clause):
        return ClauseMatch("goal", 0.8, "wants_to", ObjectPhrase(m["obj"].strip()))
    return None


def match_decision(clause: str) -> ClauseMatch | None:
    if m := _DECISION_RE.match(clause):
        text = m["obj"] or m["obj2"]
        return ClauseMatch("decision", 0.85, "decided", object_phrase(text))
    return None


def match_relationship(clause: str) -> ClauseMatch | None:
    # Case-sensitive on the name so that "my dog is hungry" is not a relationship.
    if m := _RELATIONSHIP_RE.match(clause[:1].lower() + clause[1:]):
        role = re.sub(r"\s+", "_", m["role"].lower())
        return ClauseMatch("relationship", 0.9, role, ObjectPhrase(m["name"]))
    return None
