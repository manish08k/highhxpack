# Memories

A memory (`MemoryRecord`) is an immutable record:

| Field | Meaning |
|---|---|
| `id` | 32 hex characters; the CLI accepts unique prefixes |
| `user_id` | owner; every query is scoped by it |
| `content` | the text, stored verbatim (whitespace-trimmed, control characters removed) |
| `memory_type` | `fact`, `preference`, `event`, `goal`, `relationship`, `knowledge`, `conversation`, `decision`, or any lower-case identifier |
| `source` | where it came from (`api`, `cli`, `conversation`, `slack:ops`, …) |
| `importance`, `confidence` | 0–1 |
| `status` | `active`, `superseded`, `merged`, `archived` |
| `version` | incremented by `update()` |
| `mention_count`, `access_count` | reinforcement and recall statistics |
| `created_at`, `updated_at`, `last_accessed_at`, `expires_at` | UTC timestamps |
| `superseded_by` | the memory that replaced this one |
| `embedding_model` | name of the model that produced its vector, if any |
| `metadata` | JSON object (≤ 64 KiB) |

## Types are open-ended

`MemoryType` lists the built-in names, but any identifier matching
`[a-z][a-z0-9_]{0,31}` works (`memory_type="project_note"`). When omitted, the type is
inferred from the text by the rule-based classifier (default `fact`).

## Statuses and history

| Status | Set by | Recalled? |
|---|---|---|
| `active` | new memories, `restore()` | yes |
| `superseded` | conflict resolution | no (visible with `search(status="any")`) |
| `merged` | `consolidate()` | no |
| `archived` | `forget()`, expiry during `consolidate()` | no |

Every change appends a `MemoryEvent` (`created`, `updated`, `reinforced`,
`superseded`, `merged`, `archived`, `restored`, `conflict_detected`, `imported`) with
the content at that time: `memory.history(memory_id)`.

## Deduplication

1. **Exact**: content hashes ignore case, punctuation (except `+`/`#`) and whitespace.
2. **Near**: Jaccard similarity of *canonical* tokens ≥ `dedup_threshold` (0.85).
   Canonicalization stems words, drops subject words ("I", "user") and maps preference
   verbs (like/love/prefer/enjoy) to one token. Opposite polarity ("I like X" /
   "I don't like X") is never a duplicate.

A duplicate increments `mention_count`, raises importance
(`importance += (1 - importance) × 0.1`), keeps the higher confidence and refreshes
`updated_at`. Pass `dedupe=False` to store a copy anyway.

## Conflicts

A *claim* is a slot plus a value, parsed from statements such as:

| Statement | Slot | Value |
|---|---|---|
| I prefer Java | `prefers:*` | java |
| I prefer Django for web | `prefers:web` | django |
| I like / hate Python | `sentiment:python` | positive / negative |
| I use / no longer use Vim | `uses:vim` | positive / negative |
| I live in Paris · I work at Acme · My name is Ada · I'm 30 years old | `lives_in` · `works_at` · `name` · `age` | … |
| My favorite color is blue · My manager is Bob | `favorite:color` · `attr:manager` | … |

Two active memories of one user with the same slot and different values conflict.
Temporal fillers ("now", "these days") are ignored. The `conflict_policy` decides:

- **`supersede`** (default): the newer memory becomes current **only if its confidence
  is at least the older one's**. The older memory becomes `superseded` and a
  *resolved* `Conflict` records why. Otherwise both stay active and the conflict is
  *open*.
- **`keep_both`**: both stay active; an open conflict is recorded.

Open conflicts are flagged in results (`RecallResult.has_conflict`) and listed by
`memory.conflicts()`. Resolve them with `memory.resolve_conflict(conflict_id,
keep=memory_id)`. HighHXPack never decides which statement is true on its own.

## Expiry

`remember(..., ttl="30d")` or `expires_at=...`. Expired memories disappear from
`recall`, `search` and `list` immediately (pass `include_expired=True` to see them);
`consolidate()` archives them. Nothing is deleted automatically.

## Deleting

- `forget(id)` archives (reversible with `restore(id)`). An archived memory keeps its
  `superseded_by` link; `restore(id)` clears it. `restore` does not re-run conflict
  resolution: to switch which of two conflicting memories is current, use
  `resolve_conflict(conflict_id, keep=...)`.
- `delete(id)` permanently erases the memory, its history, vector, and derived edges.
- `clear(user_id)` erases everything of one user; `clear(all_users=True)` everything.
