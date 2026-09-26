# HighHXPack documentation

HighHXPack is a local-first memory and context engine for AI applications, agents
and developer tools. It stores memories in a local SQLite file, retrieves the most
relevant ones for a query, and keeps them tidy: duplicates are merged, contradictions
are tracked with full history, and expired information is archived.

- [Installation](installation.md)
- [Quickstart](quickstart.md)

## Concepts

- [Architecture](concepts/architecture.md)
- [Memories: types, statuses, deduplication, conflicts, history](concepts/memories.md)
- [Retrieval and ranking](concepts/retrieval.md)
- [Extraction](concepts/extraction.md)
- [Consolidation](concepts/consolidation.md)
- [Knowledge graph](concepts/graph.md)
- [Storage](concepts/storage.md)
- [Providers: embeddings and LLMs](concepts/providers.md)
- [Configuration](concepts/configuration.md)
- [Privacy and security](concepts/privacy.md)

## API reference

- [`Memory`](api/memory.md)
- [Models and results](api/models.md)
- [Extension interfaces](api/extending.md)
- [Exceptions](api/exceptions.md)
- [Command-line interface](api/cli.md)
