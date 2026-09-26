# Privacy and security

## What is stored, and where

Everything lives in one SQLite database on your machine (see [storage](storage.md)):
memory text, metadata you attach, timestamps, access counts, history of every change
(including previous content), graph edges, conflict records and embedding vectors.
The configuration file contains settings only — never API keys.

## What leaves the machine

Nothing, with the default configuration (`hashing` embeddings, no LLM). Data is sent
to a third party only when you configure it:

- `embedding_provider = "openai"` / `llm_provider = "openai"`: the text being
  embedded, extracted or summarized is sent to the OpenAI API (or `openai_base_url`).
- `ollama`: text is sent to `ollama_url` (localhost by default).
- `sentence-transformers` downloads a model from the Hugging Face Hub on first use;
  your data is processed locally.

## Protections

- **Files**: data directory `0700`; database, config and exports `0600` (POSIX),
  applied at creation time. Exports are written atomically.
- **Secrets**: API keys are only read from the environment or explicit arguments,
  masked in `repr`, `highhxpack config` and `to_dict()`, and redacted from provider
  error messages and logs.
- **Input validation**: ids, types, sources, scores, metadata (JSON-only, ≤ 64 KiB),
  content length (≤ 32,000 characters) and paths are validated at the API boundary.
  Control characters are stripped, so stored text cannot inject terminal escape codes.
- **SQL**: every value is a bound parameter.
- **Imports**: JSON only (never pickle or code); `NaN`/`Infinity` rejected; size limit
  (256 MiB); every record re-validated; invalid records skipped and reported.
- **Providers**: base URLs must be `http(s)` without embedded credentials; responses
  are size-limited and validated.
- **No telemetry.**

## Erasing data

`memory.delete(id)` erases a memory with its history, vector and edges;
`memory.clear(user_id)` erases a user; `memory.clear(all_users=True)` erases
everything. Note that SQLite may keep freed pages in the file until it is reused or
`VACUUM`ed; delete the database file to guarantee removal.

Report vulnerabilities as described in [SECURITY.md](../../SECURITY.md).
