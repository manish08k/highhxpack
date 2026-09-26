# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0.0, minor releases may contain
breaking changes; they are always listed under **Changed** or **Removed**.

## [Unreleased]

## [0.1.0] - 2026-09-26

First public release.

### Added

- `Memory` API: `remember`, `ingest`, `recall`, `search`, `get`, `update`, `forget`,
  `restore`, `delete`, `list`, `count`, `clear`, `stats`, `inspect`, `users`,
  `history`, `conflicts`, `resolve_conflict`, `consolidate`, `reindex`, `export`,
  `import_`, `resolve_id`, and the `graph` property.
- SQLite storage with WAL mode, transactions, versioned migrations, owner-only file
  permissions and safe concurrent first-time initialization.
- Keyword (BM25), semantic (cosine) and hybrid retrieval with a transparent,
  centrally configured ranking (`ScoringConfig`) and per-result score breakdowns.
- Embedding providers: deterministic local hashing embedding (default),
  sentence-transformers, Ollama, OpenAI and callables.
- LLM providers: Ollama, OpenAI and callables (all optional).
- Rule-based and LLM-based extraction of facts, preferences, goals, decisions,
  relationships and events; entity recognition.
- Exact and near-duplicate detection with reinforcement; claim-based conflict
  detection with `supersede`/`keep_both` policies and full history.
- Consolidation: expiry archiving, duplicate merging, missed-conflict resolution and
  optional summaries.
- Lightweight per-user knowledge graph.
- JSON export/import with strict validation.
- `highhxpack` CLI: `init`, `config`, `remember`, `recall`, `search`, `memories`,
  `inspect`, `history`, `forget`, `restore`, `conflicts`, `resolve`, `consolidate`,
  `stats`, `export`, `import`, with `--json` output and documented exit codes.
- Integrations: `ChatMemory`/`format_memories`, LangChain and LlamaIndex retrievers.
- Graceful degradation: hybrid retrieval falls back to keyword relevance if the
  embedding provider fails; LLM providers are created on first use.
- Configuration via file, environment variables and arguments with documented
  precedence.

[Unreleased]: https://github.com/highhxpack/highhxpack/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/highhxpack/highhxpack/releases/tag/v0.1.0
