"""Graph nodes.

Nodes are not stored separately: a node exists while at least one edge touches
it.  Node names are compared case-insensitively ("Python" == "python").
"""

from __future__ import annotations

from dataclasses import dataclass

from highhxpack.exceptions import ValidationError
from highhxpack.utils.text import normalize_text

MAX_NODE_LENGTH = 200


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    degree: int


def normalize_node(name: object) -> str:
    """Validate a node name and collapse internal whitespace."""
    if not isinstance(name, str):
        raise ValidationError(f"Graph node names must be strings, got {type(name).__name__}.")
    value = normalize_text(name)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValidationError("Graph node names must be valid Unicode text.") from exc
    if not value:
        raise ValidationError("Graph node names must not be empty.")
    if len(value) > MAX_NODE_LENGTH:
        raise ValidationError(f"Graph node names must be at most {MAX_NODE_LENGTH} characters.")
    return value
