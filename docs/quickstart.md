# Quickstart

## Python

```python
from highhxpack import Memory

memory = Memory()  # SQLite file in the platform data directory; Memory("x.db") for a custom path

memory.remember("user_123", "I prefer Python for AI development.")
memory.remember("user_123", "I work at Acme Corp.", importance=0.7, metadata={"source_app": "crm"})

for result in memory.recall("user_123", "What language do I prefer for AI?", limit=5):
    print(f"{result.score:.2f}", result.content)
    print("   ", result.breakdown)  # keyword, semantic, importance, recency, ...

memory.close()  # or use `with Memory() as memory:`
```

### Learn from conversations

```python
results = memory.ingest(
    "user_123",
    "I've been using Python for ML for three years, but I prefer C++ for competitive "
    "programming. Anyway, how's it going?",
)
for r in results:
    print(r.action, r.memory.memory_type, r.memory.content)
# created fact I've been using Python for ML for three years
# created preference I prefer C++ for competitive programming
```

The question at the end is ignored: only statements that match a known pattern for
facts, preferences, goals, decisions, relationships or events are stored.

### Information that changes

```python
memory.remember("u", "I live in Paris")
result = memory.remember("u", "I live in Berlin")
result.superseded_ids  # the Paris memory is now "superseded", not deleted
memory.history(result.superseded_ids[0])  # created -> superseded, with the reason
```

### Put memories into a prompt

```python
from highhxpack.integrations import ChatMemory

chat = ChatMemory(memory, user_id="user_123")
system_prompt = "You are helpful.\n\n" + chat.context(user_message)
chat.observe(user_message)  # remember anything worth remembering
```

## CLI

```bash
highhxpack init
highhxpack remember "I prefer Python for AI development"
highhxpack recall "programming language" --explain
highhxpack memories --json
```

See the [CLI reference](api/cli.md).
