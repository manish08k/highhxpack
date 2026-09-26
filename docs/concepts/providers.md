# Providers

## Embeddings

| `embedding_provider` | Runs | Notes |
|---|---|---|
| `hashing` (default) | locally | deterministic feature hashing of stems, word pairs and character trigrams; no downloads; lexical, not synonym-aware. `embedding_model` sets the dimension (default 384) |
| `none` | – | keyword retrieval only |
| `sentence-transformers` | locally | neural; needs `highhxpack[sentence-transformers]`; downloads the model (default `all-MiniLM-L6-v2`) on first use |
| `ollama` | Ollama server | default model `nomic-embed-text`, `ollama_url` default `http://localhost:11434` |
| `openai` | OpenAI API | needs `highhxpack[openai]` and `OPENAI_API_KEY`; default `text-embedding-3-small` |

Any object implementing `EmbeddingProvider` (a unique `name` and
`embed(texts) -> list[list[float]]`) can be passed as `Memory(embedder=...)`;
`CallableEmbedding(fn, name=...)` wraps a function. Vectors are validated (count,
dimension, finite values) and normalized.

If an embedding call fails while storing, the memory is still stored (keyword-searchable)
and `RememberResult.warnings` explains how to fix it; `memory.reindex()` fills in
missing vectors later.

## LLMs

LLMs are only used for `extractor = "llm"` and LLM summaries in `consolidate`. The LLM
is created on first use, so a missing package or API key only affects those features.
Embedding providers, by contrast, are created with `Memory(...)` because retrieval needs
them; to work with a database while a remote embedder is unavailable, set
`HIGHHXPACK_EMBEDDING_PROVIDER=none` (keyword retrieval only).

| `llm_provider` | Notes |
|---|---|
| `none` (default) | |
| `ollama` | default model `llama3.2`; standard-library HTTP client, no extra package |
| `openai` | needs `highhxpack[openai]` and `OPENAI_API_KEY`; default `gpt-4o-mini`; `openai_base_url` supports compatible servers |

Implement `LLMProvider.complete(prompt, *, system=None, json_output=False,
temperature=0.0) -> str`, or wrap a function with `CallableProvider(fn)`.

## Errors

- Missing optional package → `ProviderNotAvailableError` naming the extra to install.
- Unreachable server, HTTP errors, malformed responses → `ProviderError` /
  `EmbeddingError` with the reason and a hint (for Ollama: `ollama serve`,
  `ollama pull <model>`). Secrets are redacted from error messages.
- Provider base URLs must be plain `http(s)://` URLs without embedded credentials.
