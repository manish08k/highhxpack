"""Optional lightweight knowledge graph."""

from highhxpack.graph.edges import Edge, normalize_relation
from highhxpack.graph.graph import KnowledgeGraph
from highhxpack.graph.nodes import Node, normalize_node

__all__ = ["Edge", "KnowledgeGraph", "Node", "normalize_node", "normalize_relation"]
