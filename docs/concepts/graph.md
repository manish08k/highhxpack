# Knowledge graph

The graph is optional and lightweight: edges are stored in the same database, and
nodes exist while an edge touches them. Node names compare case-insensitively.

```python
graph = memory.graph
graph.add("alice", "user", "works_on", "HighHXPack")
graph.add("alice", "HighHXPack", "written_in", "Python")

graph.neighbors("alice", "user")  # ['HighHXPack', ...]
graph.neighbors("alice", "Python", direction="in")  # ['HighHXPack']
graph.edges("alice", node="user", relation="prefers")
graph.nodes("alice")  # [Node(name, degree), ...]
graph.traverse("alice", "Python", depth=2)  # edges within two hops
graph.memories_about("alice", "Python")  # memories linked to those edges
graph.remove(edge_id)
```

`ingest()` adds edges automatically from recognized statements, linked to the memory
they came from; deleting the memory deletes those edges. Relations are normalized to
snake_case (`"Works On"` → `works_on`).
