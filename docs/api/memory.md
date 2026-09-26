# `highhxpack.Memory`

```text
Memory(storage_path=None, *, config=None, storage=None, embedder=None, llm=None, clock=None)
```

| Argument | Meaning |
|---|---|
| `storage_path` | database file or `":memory:"`; default from configuration |
| `config` | a `Config`; when omitted, `Config.load()` reads the file and environment |
| `storage` | a custom `StorageBackend` (overrides `storage_path`) |
| `embedder` | `EmbeddingProvider` instance or provider name (`"none"` disables) |
| `llm` | `LLMProvider` instance or provider name |
| `clock` | function returning the current UTC `datetime` (tests) |

`Memory` is thread-safe and a context manager. Call `close()` (or use `with`) to
checkpoint and release the database.

Properties: `config`, `storage`, `embedder`, `llm`, `graph`, `location`.

## Writing

| Method | Returns |
|---|---|
| `remember(user_id, content, *, memory_type=None, source="api", importance=None, confidence=1.0, metadata=None, ttl=None, expires_at=None, dedupe=True, detect_conflicts=True)` | `RememberResult` |
| `ingest(user_id, text, *, source="conversation", extractor=None, metadata=None)` | `list[RememberResult]` |
| `update(memory_id, *, content=None, memory_type=None, importance=None, confidence=None, metadata=None, ttl=None, expires_at=None, clear_expiry=False)` | `MemoryRecord` (version + 1) |
| `forget(memory_id, *, reason=None)` | `MemoryRecord` (archived) |
| `restore(memory_id)` | `MemoryRecord` (active) |
| `delete(memory_id)` | `None`; permanent |
| `clear(user_id=None, *, all_users=False)` | number of memories erased |

## Reading

| Method | Returns |
|---|---|
| `recall(user_id, query, *, limit=10, memory_type=None, min_score=0.0, strategy="hybrid", source=None, metadata=None, created_after=None, created_before=None)` | `list[RecallResult]` |
| `search(query, *, user_id=None, limit=10, strategy="hybrid", memory_type=None, status="active", min_score=0.0, include_expired=False, source=None, metadata=None)` | `list[RecallResult]` |
| `get(memory_id)` | `MemoryRecord` (any status) |
| `list(user_id=None, *, memory_type=None, status="active", limit=50, offset=0, order="newest", include_expired=False)` | `list[MemoryRecord]` |
| `count(user_id=None, *, memory_type=None, status="active", include_expired=False)` | `int` |
| `history(memory_id)` | `list[MemoryEvent]` |
| `users()` | `dict[str, int]` |
| `inspect(user_id)` | `UserSummary` |
| `stats(user_id=None)` | `MemoryStats` |
| `resolve_id(prefix)` | full id for a unique prefix of ≥ 4 hex characters |

`status` accepts `"active"`, `"superseded"`, `"merged"`, `"archived"`, a list of them,
or `"any"`. `order` is `"newest"`, `"oldest"`, `"importance"` or `"updated"`.

## Maintenance

| Method | Returns |
|---|---|
| `conflicts(user_id=None, *, status="open")` | `list[Conflict]` (`status=None` for all) |
| `resolve_conflict(conflict_id, *, keep)` | resolved `Conflict` |
| `consolidate(user_id, *, dry_run=False, summarize=False, summarizer=None)` | `ConsolidationReport` |
| `reindex(user_id=None)` | number of vectors computed |
| `export(path, *, user_id=None, overwrite=False)` | `ExportReport` |
| `import_(path, *, user_id=None, on_existing="skip")` | `ImportReport` (`"replace"` overwrites same-id memories) |

## Errors

Invalid arguments raise `ValidationError`; unknown ids raise `MemoryNotFoundError`
or `ConflictNotFoundError`; see [exceptions](exceptions.md).
