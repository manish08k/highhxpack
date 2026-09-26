from __future__ import annotations

import pytest

from highhxpack import Memory, ValidationError
from highhxpack.graph import normalize_node, normalize_relation


def test_normalization() -> None:
    assert normalize_relation("Works On") == "works_on"
    assert normalize_relation("interested-in") == "interested_in"
    assert normalize_node("  New   York ") == "New York"
    for bad in ["", "1abc", "a" * 70, 5]:
        with pytest.raises(ValidationError):
            normalize_relation(bad)
    for bad_node in ["", "x" * 201, None]:
        with pytest.raises(ValidationError):
            normalize_node(bad_node)


def test_add_query_remove(mem_memory: Memory) -> None:
    graph = mem_memory.graph
    edge = graph.add("u", "User", "prefers", "Python")
    assert graph.add("u", "user", "Prefers", "python").id == edge.id  # idempotent
    graph.add("u", "user", "works_on", "HighHXPack")
    graph.add("u", "user", "uses", "Flutter")
    graph.add("u", "HighHXPack", "written_in", "Python")
    graph.add("other", "user", "prefers", "Java")

    assert graph.neighbors("u", "user") == ["Flutter", "HighHXPack", "Python"]
    assert graph.neighbors("u", "python", direction="in") == ["HighHXPack", "User"]
    assert graph.neighbors("u", "user", relation="prefers") == ["Python"]
    assert [(n.name, n.degree) for n in graph.nodes("u")][:2] == [("User", 3), ("HighHXPack", 2)]
    assert len(graph.traverse("u", "Flutter", depth=1)) == 1
    assert len(graph.traverse("u", "Flutter", depth=3)) == 4
    with pytest.raises(ValidationError):
        graph.traverse("u", "Flutter", depth=0)
    assert graph.remove(edge.id)
    assert not graph.remove(edge.id)
    assert graph.neighbors("u", "user", relation="prefers") == []


def test_ingest_builds_graph_and_memories_about(mem_memory: Memory) -> None:
    mem_memory.ingest("u", "I prefer C++ for competitive programming. I use Python at work.")
    edges = {(e.relation, e.target) for e in mem_memory.graph.edges("u")}
    assert edges == {("prefers", "C++"), ("uses", "Python")}
    about = mem_memory.graph.memories_about("u", "python")
    assert [m.content for m in about] == ["I use Python at work"]


def test_edges_deleted_with_memory(mem_memory: Memory) -> None:
    [result] = mem_memory.ingest("u", "I live in Berlin")
    assert mem_memory.graph.edges("u")
    mem_memory.delete(result.id)
    assert mem_memory.graph.edges("u") == []


def test_invalid_edge_inputs(mem_memory: Memory) -> None:
    with pytest.raises(ValidationError):
        mem_memory.graph.add("u", "a", "rel", "b", confidence=2)
    with pytest.raises(ValidationError):
        mem_memory.graph.add("u", "a", "rel", "b", memory_id="nope")
