# Extension interfaces

## Embedding provider

```python
from highhxpack import EmbeddingProvider, Memory


class MyEmbedding(EmbeddingProvider):
    name = "my-model-v1"  # must identify model + configuration

    def embed(self, texts):
        return [my_model.encode(t) for t in texts]  # list[list[float]], same length


memory = Memory(embedder=MyEmbedding())
```

Override `embed_query` for models with a separate query encoder. Or wrap a function:
`CallableEmbedding(fn, name="my-model-v1")`.

## LLM provider

```python
from highhxpack import LLMProvider


class MyLLM(LLMProvider):
    name = "my-llm"

    def complete(self, prompt, *, system=None, json_output=False, temperature=0.0):
        return my_client.generate(prompt, system=system)
```

Or `CallableProvider(lambda prompt: my_client.generate(prompt))`.

## Extractor and summarizer

`highhxpack.extraction.Extractor.extract(text) -> list[ExtractedMemory]` and
`highhxpack.consolidation.Summarizer.summarize(topic, contents) -> str`; pass them as
`memory.ingest(..., extractor=...)` and `memory.consolidate(..., summarizer=...)`.

## Storage backend

Subclass `highhxpack.StorageBackend` and implement its abstract methods; see
[storage](../concepts/storage.md#custom-backends). Use the SQLite backend and its
tests (`tests/unit/test_storage.py`) as the reference.

## Framework adapters

```python
from highhxpack.integrations.langchain import create_retriever  # needs [langchain]

retriever = create_retriever(memory, user_id="alice", k=5)
docs = retriever.invoke("What does the user prefer?")

from highhxpack.integrations.llamaindex import create_retriever  # needs [llamaindex]

nodes = create_retriever(memory, user_id="alice").retrieve("preferences")

from highhxpack.integrations import ChatMemory, format_memories  # no dependencies
```
