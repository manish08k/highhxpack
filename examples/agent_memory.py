"""Long-term memory for an agent: metadata, TTLs, the knowledge graph,
consolidation and export.

Run:  python examples/agent_memory.py
"""

import tempfile
from pathlib import Path

from highhxpack import Memory

with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "agent.db"
    with Memory(db) as memory:
        agent = "agent:research-bot"

        # Decisions and tool results, tagged with metadata for later filtering.
        memory.remember(
            agent,
            "We decided to use Postgres for the analytics backend",
            memory_type="decision",
            metadata={"project": "analytics"},
        )
        memory.remember(
            agent,
            "The staging API token rotates every Monday",
            metadata={"project": "analytics"},
            ttl="7d",  # short-lived knowledge expires automatically
        )
        memory.ingest(agent, "I work on HighHXPack. I use Flutter for the mobile client.")

        # Explicit graph edges complement the ones extracted automatically.
        memory.graph.add(agent, "HighHXPack", "written_in", "Python")
        print("Graph neighbours of 'user':", memory.graph.neighbors(agent, "user"))
        print("Edges within two hops of Flutter:")
        for edge in memory.graph.traverse(agent, "Flutter", depth=2):
            print(f"  {edge.source} -{edge.relation}-> {edge.target}")

        # Retrieval restricted by metadata and type.  The default embedder is
        # lexical, so queries should share words (or word stems) with the memory;
        # configure a neural embedder to match purely by meaning.
        print("\nAnalytics decisions:")
        for r in memory.recall(
            agent, "analytics backend", memory_type="decision", metadata={"project": "analytics"}
        ):
            print(f"  {r.score:.3f} {r.content}")

        # Near-duplicates stored without deduplication are merged by consolidation.
        memory.remember(agent, "User likes Python", dedupe=False)
        memory.remember(agent, "User loves Python", dedupe=False)
        report = memory.consolidate(agent, dry_run=True)
        print("\nConsolidation plan:", *report.actions, sep="\n  ")
        memory.consolidate(agent)

        backup = Path(tmp) / "agent-backup.json"
        print("\nExported:", memory.export(backup, user_id=agent).to_dict())

    # Data survives restarts.
    with Memory(db) as memory:
        print("Memories after restart:", memory.count(agent))
