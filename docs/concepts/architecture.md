# Architecture

```text
highhxpack.Memory                     public facade: validation, defaults, public API
 ├── core.manager.MemoryManager       write path and consolidation
 │     ├── consolidation.deduplication    exact + near-duplicate detection
 │     ├── consolidation.conflict         claims, contradiction detection, policy
 │     ├── consolidation.importance       initial importance, reinforcement
 │     └── consolidation.summarization    extractive / LLM summaries
 ├── extraction                       RuleBasedExtractor, LLMExtractor, entities
 ├── retrieval.Retriever              candidates (BM25 + cosine) → Ranker → results
 │     ├── retrieval.keyword / semantic / hybrid
 │     ├── retrieval.ranking.ScoringConfig   every ranking constant
 │     └── retrieval.filters.MemoryFilter    backend-agnostic filters
 ├── graph.KnowledgeGraph             relationships between entities
 ├── embeddings.EmbeddingProvider     hashing (default) · sentence-transformers · Ollama · OpenAI · callable
 ├── providers.LLMProvider            Ollama · OpenAI · callable (optional)
 └── storage.StorageBackend           SQLiteStorage (default) · migrations · vector encoding
```

## Write path (`remember`)

1. **Validate** user id, content, type, scores, metadata, expiry.
2. **Classify** the memory type (if not given) and **estimate importance**.
3. **Deduplicate**: exact content hash, then near-duplicates among the top keyword
   candidates. A duplicate *reinforces* the existing memory (mention count,
   importance) and records a `reinforced` history event.
4. **Index**: term frequencies for BM25, a *claim slot* for conflict detection, and
   an embedding (a failing embedding provider produces a warning, not an error).
5. **Store** the memory and a `created` event in one transaction.
6. **Conflicts**: active memories with the same claim slot but a different value are
   resolved by the configured policy (see [memories](memories.md)).

## Read path (`recall` / `search`)

1. Keyword candidates via BM25 over stored postings; semantic candidates via exact
   cosine similarity over vectors from the *same* embedding model.
2. Union the candidates and score each with `Ranker` (see [retrieval](retrieval.md)).
3. Sort deterministically, apply `min_score` and `limit`, flag open conflicts.
4. `recall` records access statistics; `search` does not.

## Design choices

- **Standard library only** in the core, so installation never fails on native wheels.
- **Deterministic**: the default tokenizer, embedding, extractor and ranking are pure
  functions of their inputs (plus the clock for recency).
- **History over deletion**: updates, supersessions and merges keep prior content in
  the event log; only `delete`/`clear` erase data.
- **Pluggable backends**: `Memory(storage=MyBackend())` accepts any `StorageBackend`.
