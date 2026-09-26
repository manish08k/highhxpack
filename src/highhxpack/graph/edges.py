"""Graph edges (stored as :class:`~highhxpack.models.Relationship`)."""

from __future__ import annotations

import re

from highhxpack.exceptions import ValidationError
from highhxpack.models.relationship import Relationship

#: Edges are relationships; the alias documents intent in graph code.
Edge = Relationship

_RELATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def normalize_relation(relation: object) -> str:
    """Return ``relation`` as a snake_case identifier ("Works On" -> "works_on")."""
    if not isinstance(relation, str):
        raise ValidationError("relation must be a string.")
    value = re.sub(r"[\s\-]+", "_", relation.strip().lower())
    if not _RELATION_RE.match(value):
        raise ValidationError(
            f"Invalid relation: {relation!r}",
            hint="Use letters, digits and underscores, starting with a letter (e.g. 'works_on').",
        )
    return value
