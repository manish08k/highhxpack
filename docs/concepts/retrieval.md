# Retrieval and ranking

```text
memory.recall(user_id, query, limit=10, memory_type=None, min_score=0.0,
              strategy="hybrid", source=None, metadata=None,
              created_after=None, created_before=None)
memory.search(query, user_id=None, limit=10, strategy="hybrid", memory_type=None,
              status="active", min_score=0.0, include_expired=False, source=None, metadata=None)
```

`recall` is scoped to one user, only returns active unexpired memories, and records
access statistics. `search` can span users and statuses and never changes data.

## Strategies

| Strategy | Candidates | Relevance |
|---|---|---|
| `keyword` | BM25 over stemmed terms | keyword score |
| `semantic` | cosine similarity ≥ `min_semantic_similarity` | semantic score |
| `hybrid` (default) | union of both | weighted mean of both |

`semantic` requires an embedding provider; `hybrid` falls back to keyword-only when
embeddings are disabled, and also when the provider fails at query time (for example an
unreachable Ollama server) — a warning is logged on the `highhxpack` logger. Only vectors produced by the *current* provider (same `name`)
are compared; run `memory.reindex()` after switching providers.

**Keyword scores** are BM25 (k1 = 1.2, b = 0.75) normalized to [0, 1] by the score a
document containing each known query term once would get. Query words that occur in
no memory do not lower scores.

## Scoring

All constants live in `ScoringConfig` (config file section `[scoring]`):

```text
relevance = (keyword_weight·keyword + semantic_weight·semantic) / (sum of enabled weights)

final = relevance × (base_weight + importance_weight·importance + recency_weight·recency
                     + confidence_weight·confidence + frequency_weight·frequency)
        / (base_weight + importance_weight + recency_weight + confidence_weight + frequency_weight)

recency   = 0.5 ^ (age_in_days(updated_at) / recency_half_life_days)
frequency = min(1, log1p(mentions − 1 + accesses) / log1p(frequency_saturation))
```

| Option | Default |
|---|---|
| `base_weight` | 0.60 |
| `importance_weight` | 0.15 |
| `recency_weight` | 0.10 |
| `confidence_weight` | 0.10 |
| `frequency_weight` | 0.05 |
| `keyword_weight` / `semantic_weight` | 0.5 / 0.5 |
| `recency_half_life_days` | 30 |
| `frequency_saturation` | 20 |
| `bm25_k1` / `bm25_b` | 1.2 / 0.75 |
| `min_semantic_similarity` | 0.2 |
| `candidate_multiplier` / `min_candidates` | 5 / 50 |

Because relevance multiplies everything, an unrelated memory scores 0 regardless of
importance. Each `RecallResult.breakdown` shows every component. Ties are broken by
relevance, then recency, then id, so results are deterministic.

## Filters

`memory_type` and `source` accept a string or a list. `metadata={"project": "x"}`
matches top-level metadata keys exactly (str, int, float, bool or None values).
`created_after`/`created_before` accept ISO-8601 strings or datetimes.

## Performance

Keyword search uses an indexed postings table; semantic search is an exact scan over
the user's vectors. Measure on your hardware with
`python benchmarks/benchmark_retrieval.py --sizes 1000 10000`.
