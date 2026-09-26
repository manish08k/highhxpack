"""The five-minute tour: store, recall, update, forget.

Run:  python examples/basic.py

``Memory()`` with no arguments stores data in your platform data directory.
This example uses a temporary file so it leaves nothing behind.
"""

import tempfile
from pathlib import Path

from highhxpack import Memory

with tempfile.TemporaryDirectory() as tmp, Memory(Path(tmp) / "memory.db") as memory:
    # Store memories. The memory type is inferred when not given.
    memory.remember("user_123", "I prefer Python for AI development.")
    memory.remember("user_123", "I work at Acme Corp as a data engineer.")
    memory.remember("user_123", "My favorite editor is Neovim.")

    # Repeating information reinforces the existing memory instead of duplicating it.
    again = memory.remember("user_123", "User prefers Python for AI development")
    print(f"Second mention: {again.action.value} (mentions: {again.memory.mention_count})")

    # Recall ranks by relevance, importance, recency, confidence and frequency.
    print("\nWhat language do I prefer for AI?")
    for result in memory.recall("user_123", "What language do I prefer for AI?", limit=3):
        print(f"  {result.score:.3f}  [{result.memory.memory_type}] {result.content}")

    # Changing information: the older statement is kept as history.
    memory.remember("user_123", "I work at Globex as a data engineer.")
    print("\nCurrent job:", memory.recall("user_123", "where do I work", limit=1)[0].content)

    # Update, inspect history, then forget.
    editor = memory.recall("user_123", "editor", limit=1)[0].memory
    memory.update(editor.id, content="My favorite editor is Helix.")
    print("\nHistory of the editor memory:")
    for event in memory.history(editor.id):
        print(f"  v{event.version} {event.kind}: {event.content}")

    memory.forget(editor.id)  # archived: hidden from recall, restorable
    print("\nActive memories:", memory.count("user_123"))
    print(memory.stats().by_status)
