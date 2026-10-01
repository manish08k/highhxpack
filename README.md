# HighHXPackk

**A local-first memory and context engine for AI applications, agents and developer tools.**

HighHXPack gives your chatbot, agent or tool a long-term memory: it stores what users
tell it, recognizes facts and preferences, avoids duplicates, keeps track of
information that changes over time, and retrieves the most relevant memories for a
prompt — all on the local machine, in a single SQLite file, with **no required
dependencies, API keys, or cloud services**.

```python
from highhxpack import Memory

memory = Memory()
memory.remember(user_id="user_123", content="I prefer Python for AI development.")

results = memory.recall(user_id="user_123", query="What language do I prefer for AI?")
print(results[0].content)  # I prefer Python for AI development.
```

> Status: **alpha (0.1.0)**. The public API is documented and tested, but may
> still change before 1.0; changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Why HighHXPack?

LLM applications forget everything between sessions. Common fixes send user data to a
hosted memory service or require a vector database. HighHXPack is a library instead:

- **Local and private by default** — data lives in a SQLite file you control.
  Nothing leaves the machine unless you configure a remote provider.
- **Zero required dependencies** — the core uses only the Python standard library.
- **More than a vector store** — deduplication, conflict handling with history,
  importance/recency ranking, extraction, consolidation and a small knowledge graph.
- **Transparent ranking** — every result carries a score breakdown; all weights live
  in one configuration object.
- **Pluggable** — bring your own embedding model, LLM, or storage backend.

## Installation

```bash
pip install highhxpack
```

Requires Python 3.11+. Optional extras add third-party providers:

```bash
pip install "highhxpack[openai]"                 # OpenAI embeddings / LLM
pip install "highhxpack[sentence-transformers]"  # local neural embeddings
pip install "highhxpack[langchain]"              # LangChain retriever
pip install "highhxpack[llamaindex]"             # LlamaIndex retriever
pip install "highhxpack[ollama]"                 # no extra packages; Ollama uses HTTP
pip install "highhxpack[all]"
```

## Five-minute quickstart

```python
from highhxpack import Memory

with Memory() as memory:  # ~/.local/share/highhxpack/memory.db on Linux
    memory.remember("alice", "I prefer Python for AI development.")
    memory.remember("alice", "My favorite editor is Neovim.", importance=0.8)
    memory.remember("alice", "Temporary door code is 4321", ttl="1d")  # expires

    # Duplicates reinforce the existing memory instead of piling up.
    result = memory.remember("alice", "User prefers Python for AI development")
    print(result.action)  # reinforced

    # Changing information: the old statement is superseded, not deleted.
    memory.remember("alice", "I live in Paris")
    update = memory.remember("alice", "I live in Berlin")
    print(update.superseded_ids)  # ('<id of the Paris memory>',)

    # Ranked retrieval with an explanation of every score.
    for r in memory.recall("alice", "where do I live?", limit=5):
        print(f"{r.score:.2f} {r.content}  relevance={r.breakdown.relevance:.2f}")

    # Extract memorable statements from free text (offline, rule-based).
    memory.ingest(
        "alice",
        "I've been using Python for ML for years, "
        "but I prefer C++ for competitive programming. How are you?",
    )
    print(memory.graph.neighbors("alice", "user"))  # ['C++', 'Python']

    # Everything else
    memory.search("python", user_id="alice", status="any")  # includes history
    memory.list("alice", memory_type="preference")
    memory.history(update.id)  # every version/change
    memory.consolidate("alice", dry_run=True)  # merge dupes, apply policy
    memory.export("alice.json", user_id="alice")
    memory.stats()
```

The full public API: `remember`, `ingest`, `recall`, `search`, `get`, `update`,
`forget`, `restore`, `delete`, `list`, `count`, `clear`, `stats`, `inspect`,
`users`, `history`, `conflicts`, `resolve_conflict`, `consolidate`, `reindex`,
`export`, `import_`, and the `graph` property. See [docs/api](docs/api/memory.md).

## CLI

```bash
highhxpack init                         # data directory, config file, database
highhxpack remember "I use Python"
highhxpack remember --extract "I prefer Rust, but I use Go at work."
highhxpack recall "programming language" --explain
highhxpack search "python" --status any --json
highhxpack memories                     # list (alias: list)
highhxpack inspect alice                # summary + knowledge graph of a user
highhxpack history 1a2b3c4d             # a memory and its change history
highhxpack forget 1a2b3c4d              # archive (restorable); --permanent to erase
highhxpack conflicts                    # contradictions awaiting a decision
highhxpack consolidate --dry-run
highhxpack stats
highhxpack export backup.json
highhxpack import backup.json
highhxpack config                       # effective configuration (secrets masked)
highhxpack --help / --version
```

Every command accepts `--user`, `--db`, `--config`, `--json` and `--no-color`.
Memory ids can be abbreviated to a unique prefix. Exit codes: `0` success, `1` error,
`2` invalid input, `3` not found, `130` interrupted. Errors are printed as a message,
reason and hint — never as a traceback.

## Architecture

```text
Memory (public API)
 ├── MemoryManager ─ write path: validation → dedup → conflicts → index → history
 │     ├── consolidation/  deduplication · conflict · importance · summarization
 │     └── extraction/     rule-based or LLM extraction of facts, preferences, …
 ├── Retriever ─ keyword (BM25) + semantic (cosine) → hybrid ranking
 ├── KnowledgeGraph ─ user --prefers--> C++
 ├── EmbeddingProvider ─ hashing (default) · sentence-transformers · Ollama · OpenAI · custom
 ├── LLMProvider ─ none (default) · Ollama · OpenAI · custom
 └── StorageBackend ─ SQLiteStorage (default, WAL, migrations) · your own
```

Details: [docs/concepts/architecture.md](docs/concepts/architecture.md).

## How it works

**Memories** have a type (`fact`, `preference`, `event`, `goal`, `relationship`,
`knowledge`, `conversation`, `decision`, or any lower-case identifier you choose),
importance, confidence, source, metadata, optional expiry, a version and a status:
`active` (current belief), `superseded` (replaced by newer information),
`merged` (folded into a duplicate), or `archived` (forgotten, restorable).

**Deduplication**: identical content (ignoring case/punctuation) and near-duplicates
(canonical token overlap, e.g. "User likes Python" ≈ "User prefers Python") reinforce
the existing memory. Statements with opposite polarity are never merged.

**Conflicts**: statements that fill the same *slot* with different values
("I prefer Java" → "I prefer Python", "I live in Paris" → "I live in Berlin") are
detected. Under the default `supersede` policy the newer statement becomes current
only if it is at least as confident; the older one is kept with status `superseded`
and a conflict record explains why. Otherwise both stay active and the conflict is
left open for you to resolve. HighHXPack never guesses which statement is true.

**Ranking** (all constants in `ScoringConfig`):

```text
relevance = weighted mean of keyword (BM25, normalized) and semantic (cosine) scores
final     = relevance × (0.60 + 0.15·importance + 0.10·recency + 0.10·confidence + 0.05·frequency)
```

A memory that does not match the query scores 0 however important it is. See
[docs/concepts/retrieval.md](docs/concepts/retrieval.md).

## Storage behavior

- One SQLite file, by default in the platform data directory
  (`~/.local/share/highhxpack/memory.db`, `~/Library/Application Support/highhxpack/`
  on macOS, `%LOCALAPPDATA%\highhxpack\` on Windows), or `Memory("path/to/file.db")`.
- WAL mode, explicit transactions, versioned migrations, indexes, and a clean
  checkpoint on close. Safe for several threads and several processes.
- The database directory is created with mode `0700` and the file `0600` (POSIX).
- `Memory(":memory:")` gives a throw-away in-memory store.

## Embeddings and LLM providers

Embeddings and LLMs are **optional**.

| Setting | Default | Options |
|---|---|---|
| `embedding_provider` | `hashing` (local, lexical, no downloads) | `none`, `sentence-transformers`, `ollama`, `openai` |
| `llm_provider` | `none` | `ollama`, `openai` |

The default hashing embedding is deterministic and offline. It captures shared words,
stems and sub-word fragments, but not synonyms; for meaning-based search use
`sentence-transformers` or Ollama. LLMs are used only for `extractor = "llm"` and
LLM-written summaries. Custom providers:

```python
from highhxpack import CallableEmbedding, CallableProvider, Memory

memory = Memory(
    embedder=CallableEmbedding(my_embed_fn, name="my-model-v1"),  # fn(list[str]) -> vectors
    llm=CallableProvider(my_complete_fn),  # fn(prompt) -> str
)
```

Missing optional packages produce an error that says exactly what to install.

## Configuration

Precedence (lowest → highest): built-in defaults → config file
(`<home>/config.toml` or `$HIGHHXPACK_CONFIG`) → `HIGHHXPACK_*` environment variables →
explicit Python arguments / CLI options. `highhxpack config` prints the result.
API keys are **never** read from the config file; set `OPENAI_API_KEY` instead.
See [docs/concepts/configuration.md](docs/concepts/configuration.md).

## Privacy and security

- All data stays on your machine unless you configure `ollama` on a remote host or
  `openai`; only then is the text being embedded/processed sent to that endpoint.
- Secrets are never logged or printed; the config file cannot contain API keys.
- Input is validated; control characters are stripped; SQL uses bound parameters.
- Imports accept only HighHXPack's JSON format (never pickle), validate every record,
  and enforce a size limit. Exports are written atomically with mode `0600`.
- `delete()`/`clear()` permanently erase memories including their history.

See [SECURITY.md](SECURITY.md) and [docs/concepts/privacy.md](docs/concepts/privacy.md).

## Examples

- [examples/basic.py](examples/basic.py) — store, recall, update, forget
- [examples/chatbot.py](examples/chatbot.py) — memory loop for a chatbot
- [examples/agent_memory.py](examples/agent_memory.py) — metadata, TTL, graph, consolidation, export
- [examples/local_llm.py](examples/local_llm.py) — Ollama extraction and summaries

## Development

```bash
git clone https://github.com/highhxpack/highhxpack
cd highhxpack
uv sync                     # or: python -m venv .venv && pip install -e . --group dev
uv run pytest --cov         # unit + integration tests, 90% coverage floor
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run python benchmarks/benchmark_retrieval.py
```

Benchmarks print measurements for your hardware; no numbers are claimed here.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately as described
in [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
