# Configuration

## Precedence

From lowest to highest; later sources override earlier ones:

1. built-in defaults
2. the configuration file
3. `HIGHHXPACK_*` environment variables
4. explicit arguments — `Memory(...)`/`Config.load(...)` keyword arguments or CLI options

`Memory(config=Config(...))` uses exactly the given `Config` (no file or environment).
`highhxpack config` prints the effective configuration with secrets masked.

## Configuration file

`$HIGHHXPACK_CONFIG` if set (it must exist), otherwise `<home>/config.toml` if it
exists. `highhxpack init` writes a commented template with mode `0600`.

```toml
storage_path = "memory.db"           # relative to this file's directory
default_user = "default"
embedding_provider = "hashing"       # hashing | none | sentence-transformers | ollama | openai
# embedding_model = "nomic-embed-text"
llm_provider = "none"                # none | ollama | openai
# llm_model = "llama3.2"
ollama_url = "http://localhost:11434"
# openai_base_url = "https://api.openai.com/v1"
request_timeout = 60.0
dedup_threshold = 0.85
conflict_policy = "supersede"        # supersede | keep_both
extractor = "rules"                  # rules | llm
track_access = true

[scoring]
recency_half_life_days = 30.0
# see docs/concepts/retrieval.md for every scoring option
```

Unknown keys, wrong types and invalid values are rejected with a message naming the
key. **API keys are not allowed in the file**; this keeps it safe to share.

## Environment variables

| Variable | Setting |
|---|---|
| `HIGHHXPACK_HOME` | data directory (database and config file) |
| `HIGHHXPACK_CONFIG` | configuration file path |
| `HIGHHXPACK_STORAGE_PATH` | `storage_path` |
| `HIGHHXPACK_USER` | `default_user` |
| `HIGHHXPACK_EMBEDDING_PROVIDER` / `HIGHHXPACK_EMBEDDING_MODEL` | embeddings |
| `HIGHHXPACK_LLM_PROVIDER` / `HIGHHXPACK_LLM_MODEL` | LLM |
| `HIGHHXPACK_OLLAMA_URL` / `HIGHHXPACK_OPENAI_BASE_URL` | endpoints |
| `HIGHHXPACK_REQUEST_TIMEOUT` | seconds |
| `HIGHHXPACK_DEDUP_THRESHOLD` | 0–1 |
| `HIGHHXPACK_CONFLICT_POLICY` | `supersede` / `keep_both` |
| `HIGHHXPACK_EXTRACTOR` | `rules` / `llm` |
| `HIGHHXPACK_TRACK_ACCESS` | `true` / `false` |
| `OPENAI_API_KEY` (or `HIGHHXPACK_OPENAI_API_KEY`) | OpenAI key, read only by the OpenAI provider |

## Python

```python
from pathlib import Path

from highhxpack import Config, Memory, ScoringConfig

config = Config.load(conflict_policy="keep_both")  # file + env + overrides
memory = Memory(config=config)

memory = Memory(
    config=Config(
        storage_path=Path("agent.db"),
        embedding_provider="none",
        scoring=ScoringConfig(recency_half_life_days=7),
    )
)
```
