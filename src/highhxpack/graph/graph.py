"""A lightweight per-user knowledge graph.

The graph is optional: memories work without it.  Edges are added explicitly
through :meth:`KnowledgeGraph.add` or automatically by :meth:`Memory.ingest`
when extraction recognizes a relationship ("I prefer C++" ->
``user --prefers--> C++``).  Each auto-generated edge references the memory it
came from and is deleted together with that memory.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from highhxpack.exceptions import ValidationError
from highhxpack.graph.edges import normalize_relation
from highhxpack.graph.nodes import Node, normalize_node
from highhxpack.models.memory import MemoryRecord
from highhxpack.models.relationship import Relationship
from highhxpack.storage.base import Direction, StorageBackend
from highhxpack.utils.hashing import new_id
from highhxpack.utils.validation import (
    validate_limit,
    validate_memory_id,
    validate_metadata,
    validate_unit_interval,
    validate_user_id,
)

MAX_TRAVERSAL_DEPTH = 5


class KnowledgeGraph:
    """Relationship queries over one storage backend.  Obtain it via ``memory.graph``."""

    def __init__(self, storage: StorageBackend, clock: Callable[[], datetime]):
        self._storage = storage
        self._clock = clock

    def add(
        self,
        user_id: str,
        source: str,
        relation: str,
        target: str,
        *,
        confidence: float = 1.0,
        memory_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Relationship:
        """Add ``source --relation--> target``; returns the existing edge if already present."""
        edge = Relationship(
            id=new_id(),
            user_id=validate_user_id(user_id),
            source=normalize_node(source),
            relation=normalize_relation(relation),
            target=normalize_node(target),
            confidence=validate_unit_interval(confidence, field="confidence"),
            created_at=self._clock(),
            memory_id=validate_memory_id(memory_id) if memory_id is not None else None,
            metadata=validate_metadata(metadata),
        )
        return self._storage.add_relationship(edge)

    def edges(
        self,
        user_id: str,
        *,
        node: str | None = None,
        relation: str | None = None,
        direction: Direction = "both",
    ) -> list[Relationship]:
        """Edges of a user, optionally only those touching ``node``."""
        return self._storage.list_relationships(
            validate_user_id(user_id),
            node=normalize_node(node) if node is not None else None,
            relation=normalize_relation(relation) if relation is not None else None,
            direction=direction,
        )

    def neighbors(
        self,
        user_id: str,
        node: str,
        *,
        relation: str | None = None,
        direction: Direction = "out",
    ) -> list[str]:
        """Names of nodes directly connected to ``node`` (sorted, case-insensitively unique)."""
        name = normalize_node(node)
        found: dict[str, str] = {}
        for edge in self.edges(user_id, node=name, relation=relation, direction=direction):
            other = edge.target if edge.source.casefold() == name.casefold() else edge.source
            found.setdefault(other.casefold(), other)
        return sorted(found.values(), key=str.casefold)

    def nodes(self, user_id: str) -> list[Node]:
        """All nodes of a user with their degree, most connected first."""
        degree: dict[str, int] = {}
        names: dict[str, str] = {}
        for edge in self.edges(user_id):
            for name in (edge.source, edge.target):
                key = name.casefold()
                names.setdefault(key, name)
                degree[key] = degree.get(key, 0) + 1
        return sorted(
            (Node(names[k], d) for k, d in degree.items()),
            key=lambda n: (-n.degree, n.name.casefold()),
        )

    def traverse(self, user_id: str, start: str, *, depth: int = 2) -> list[Relationship]:
        """Edges reachable from ``start`` within ``depth`` hops (breadth-first, both directions)."""
        if (
            isinstance(depth, bool)
            or not isinstance(depth, int)
            or not 1 <= depth <= MAX_TRAVERSAL_DEPTH
        ):
            raise ValidationError(f"depth must be an integer between 1 and {MAX_TRAVERSAL_DEPTH}.")
        uid = validate_user_id(user_id)
        visited = {normalize_node(start).casefold()}
        frontier = deque([(normalize_node(start), 0)])
        seen_edges: dict[str, Relationship] = {}
        while frontier:
            node, level = frontier.popleft()
            if level >= depth:
                continue
            for edge in self._storage.list_relationships(uid, node=node, direction="both"):
                seen_edges.setdefault(edge.id, edge)
                for neighbor in (edge.source, edge.target):
                    if neighbor.casefold() not in visited:
                        visited.add(neighbor.casefold())
                        frontier.append((neighbor, level + 1))
        return sorted(
            seen_edges.values(),
            key=lambda e: (e.source.casefold(), e.relation, e.target.casefold()),
        )

    def memories_about(self, user_id: str, node: str, *, limit: int = 50) -> list[MemoryRecord]:
        """Active memories linked to edges touching ``node`` (graph-based retrieval)."""
        validate_limit(limit)
        ids = [e.memory_id for e in self.edges(user_id, node=node) if e.memory_id is not None]
        records = self._storage.get_memories(ids)
        result = [m for m in (records.get(i) for i in dict.fromkeys(ids)) if m and m.is_active]
        result.sort(key=lambda m: (-m.importance, m.created_at, m.id))
        return result[:limit]

    def remove(self, edge_id: str) -> bool:
        """Delete one edge; returns ``False`` when it did not exist."""
        return self._storage.delete_relationship(validate_memory_id(edge_id))
