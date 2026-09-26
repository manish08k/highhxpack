# Models and results

All models are frozen dataclasses with a `to_dict()` method returning JSON-compatible
data. Import them from `highhxpack`.

- **`MemoryRecord`** — see [memories](../concepts/memories.md) for fields.
  Also `is_active`, `short_id`, `is_expired(now)`, `with_changes(**fields)`,
  `from_dict(data)`.
- **`RememberResult`** — `memory`, `action` (`RememberAction.CREATED` or
  `RememberAction.REINFORCED`), `superseded_ids`, `conflicts`, `warnings`; properties
  `id`, `created`.
- **`RecallResult`** — `memory`, `score`, `breakdown`, `has_conflict`; properties `id`,
  `content`.
- **`ScoreBreakdown`** — `keyword`, `semantic`, `relevance`, `importance`, `recency`,
  `confidence`, `frequency`, `final`.
- **`MemoryEvent`** — `memory_id`, `user_id`, `kind` (`EventKind`), `version`,
  `content`, `created_at`, `related_id`, `reason`, `id`.
- **`Relationship`** — `id`, `user_id`, `source`, `relation`, `target`, `confidence`,
  `created_at`, `memory_id`, `metadata`.
- **`Conflict`** — `id`, `user_id`, `memory_id` (newer), `other_id` (older), `kind`
  (`changed_value` or `contradiction`), `status` (`ConflictStatus.OPEN`/`RESOLVED`),
  `created_at`, `resolution`, `resolved_at`.
- **`MemoryStats`** — `total`, `active`, `users`, `by_type`, `by_status`,
  `relationships`, `open_conflicts`, `vectors`, `missing_vectors`, `storage_path`,
  `storage_bytes`, `schema_version`, `embedding_model`, `user_id`.
- **`UserSummary`** — `user_id`, `total`, `active`, `by_type`, `by_status`,
  `open_conflicts`, `relationships`, `first_memory_at`, `last_memory_at`.
- **`ConsolidationReport`** — `user_id`, `dry_run`, `merged` (survivor, merged) pairs,
  `superseded` (newer, older) pairs, `conflicts_detected`, `expired_archived`,
  `summaries_created`, `actions`, property `changed`.
- **`ExportReport`** / **`ImportReport`** — counts; `ImportReport.errors` lists rejected
  records.
- Enums: `MemoryType`, `MemoryStatus`, `EventKind`, `ConflictStatus`, `ConflictPolicy`,
  `RememberAction`, `Strategy`.
