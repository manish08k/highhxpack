# Installation

HighHXPack requires **Python 3.11 or newer** and has **no required third-party
dependencies**.

```bash
pip install highhxpack
```

This installs the `highhxpack` Python package and the `highhxpack` command.

## Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `openai` | `openai` | `OpenAIProvider`, `OpenAIEmbedding` |
| `sentence-transformers` | `sentence-transformers` | local neural embeddings |
| `langchain` | `langchain-core` | `highhxpack.integrations.langchain` |
| `llamaindex` | `llama-index-core` | `highhxpack.integrations.llamaindex` |
| `ollama` | nothing | Ollama is reached over HTTP with the standard library |
| `all` | all of the above | |

```bash
pip install "highhxpack[sentence-transformers]"
```

If you configure a provider whose package is missing, HighHXPack raises
`ProviderNotAvailableError` with the exact `pip install` command to run.

## Verify

```bash
highhxpack --version
highhxpack init
highhxpack remember "I use Python"
highhxpack memories
```

## From source

```bash
git clone https://github.com/highhxpack/highhxpack
cd highhxpack
uv sync            # creates .venv with the package and dev tools
uv run pytest
```

Without uv: `python -m venv .venv && . .venv/bin/activate && pip install -e . --group dev`
(`--group` needs pip 25.1+).
