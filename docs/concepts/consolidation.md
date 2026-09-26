# Consolidation

`memory.consolidate(user_id, dry_run=False, summarize=False)` tidies one user's
memories and returns a `ConsolidationReport`. Run it periodically or after imports.

1. **Expired memories** (explicit `ttl`/`expires_at`) are archived.
2. **Near-duplicates** that were stored with `dedupe=False` (or imported) are merged.
   The survivor (most mentions, then oldest) receives the combined mention and access
   counts and the highest importance and confidence; the others become `merged`.
   Candidate pairs are found with prefix filtering, so large collections are not
   compared pairwise.
3. **Contradictions** not caught at write time (for example with
   `detect_conflicts=False`) are resolved with the configured policy. Pairs that already
   have a conflict record are skipped.
4. With `summarize=True`, groups of three or more memories about the same entity get an
   additional `knowledge` memory summarizing them — verbatim statements by default, or
   an LLM summary when an LLM is configured. Originals are untouched and the summary
   is not recreated for the same set of memories.

`dry_run=True` returns the planned actions without changing anything. Consolidation
never deletes data: every change is recorded in history and can be reversed with
`restore()`.
