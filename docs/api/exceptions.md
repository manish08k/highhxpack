# Exceptions

Every error raised deliberately derives from `HighHXPackError` and carries `message`
(what went wrong), `reason` (why) and `hint` (how to fix it); `str(error)` shows all
three.

```text
HighHXPackError
├── ConfigurationError           invalid settings, missing API key
├── ValidationError (ValueError) invalid arguments or input data
├── StorageError                 database cannot be opened/read/written, lock timeout
│   └── MigrationError           schema upgrade failed or database is from a newer release
├── MemoryNotFoundError (LookupError)
├── ConflictNotFoundError (LookupError)
├── ProviderError                LLM/embedding provider failed
│   ├── ProviderNotAvailableError   optional dependency missing (names the extra)
│   └── EmbeddingError              failed or malformed embeddings
├── ExtractionError              LLM extraction output unusable
├── ExportError
└── MemoryImportError            named so it never shadows the built-in ImportError
```
