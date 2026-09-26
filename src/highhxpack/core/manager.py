"""Write-path orchestration: indexing, deduplication, conflicts and consolidation.

:class:`MemoryManager` is internal.  It assumes its inputs were validated by
:class:`highhxpack.Memory`, and it performs every multi-step change inside a
storage transaction so a failure never leaves half-applied state.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from highhxpack.consolidation.conflict import (
    Claim,
    ConflictKind,
    ConflictPolicy,
    detect_conflicts,
    parse_claim,
    should_supersede,
)
from highhxpack.consolidation.deduplication import best_duplicate, find_duplicate_pairs
from highhxpack.consolidation.importance import ImportanceConfig, reinforce
from highhxpack.consolidation.summarization import Summarizer, group_by_entity, summary_key
from highhxpack.core.lifecycle import Clock
from highhxpack.embeddings.base import EmbeddingProvider, check_vectors
from highhxpack.exceptions import HighHXPackError, MemoryNotFoundError, ValidationError
from highhxpack.extraction.extractor import ExtractedRelation
from highhxpack.graph.graph import KnowledgeGraph
from highhxpack.models.event import EventKind, MemoryEvent
from highhxpack.models.memory import MemoryRecord, MemoryStatus
from highhxpack.models.relationship import Conflict, ConflictStatus
from highhxpack.models.result import ConsolidationReport, RememberAction, RememberResult
from highhxpack.retrieval.filters import MemoryFilter
from highhxpack.retrieval.keyword import keyword_search
from highhxpack.retrieval.ranking import ScoringConfig
from highhxpack.storage.base import IndexData, StorageBackend
from highhxpack.utils.hashing import content_hash, new_id
from highhxpack.utils.logging import get_logger
from highhxpack.utils.text import term_frequencies

logger = get_logger("manager")

#: Number of keyword candidates inspected for near-duplicates on each write.
DEDUP_CANDIDATES = 20
#: Page size used when scanning all memories of a user.
SCAN_PAGE_SIZE = 1_000
#: Upper bound on summaries created by one consolidation pass.
MAX_SUMMARIES_PER_PASS = 20
SUMMARY_SOURCE = "consolidation"


@dataclass(frozen=True, slots=True)
class NewMemory:
    """Validated input for :meth:`MemoryManager.store`."""

    user_id: str
    content: str
    memory_type: str
    source: str
    importance: float
    confidence: float
    metadata: dict[str, Any]
    expires_at: datetime | None = None


class MemoryManager:
    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider | None,
        graph: KnowledgeGraph,
        *,
        clock: Clock,
        scoring: ScoringConfig,
        conflict_policy: ConflictPolicy,
        dedup_threshold: float,
        importance: ImportanceConfig | None = None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.graph = graph
        self.clock = clock
        self.scoring = scoring
        self.conflict_policy = conflict_policy
        self.dedup_threshold = dedup_threshold
        self.importance = importance or ImportanceConfig()

    # -- indexing ---------------------------------------------------------
    def embed_many(self, texts: Sequence[str]) -> tuple[list[list[float] | None], list[str]]:
        """Embed texts; on provider failure return ``None`` vectors plus a warning."""
        if self.embedder is None or not texts:
            return [None] * len(texts), []
        try:
            vectors = check_vectors(
                self.embedder.embed(list(texts)), len(texts), provider=self.embedder.name
            )
        except HighHXPackError as exc:
            message = (
                f"Embedding with {self.embedder.name} failed ({exc.message}); memories were "
                "stored without vectors and remain searchable by keyword. Run reindex() later."
            )
            logger.info(message)  # also returned in RememberResult.warnings
            return [None] * len(texts), [message]
        return list(vectors), []

    def index_for(self, content: str, vector: list[float] | None) -> tuple[IndexData, Claim | None]:
        claim = parse_claim(content)
        return (
            IndexData(
                content_hash=content_hash(content),
                terms=term_frequencies(content),
                claim_slot=claim.slot if claim else None,
                vector=(self.embedder.name, vector)
                if vector is not None and self.embedder is not None
                else None,
            ),
            claim,
        )

    # -- writes -----------------------------------------------------------
    def find_duplicate(self, user_id: str, content: str, now: datetime) -> MemoryRecord | None:
        """Return an active, unexpired memory that duplicates ``content``."""
        for record in self.storage.find_by_hash(user_id, content_hash(content)):
            if not record.is_expired(now):
                return record
        flt = MemoryFilter(user_id=user_id, now=now)
        scores = keyword_search(
            self.storage,
            content,
            flt,
            k1=self.scoring.bm25_k1,
            b=self.scoring.bm25_b,
            limit=DEDUP_CANDIDATES,
        )
        if not scores:
            return None
        records = self.storage.get_memories(list(scores))
        match = best_duplicate(
            content,
            ((i, records[i].content) for i in scores if i in records),
            threshold=self.dedup_threshold,
        )
        return records[match.memory_id] if match else None

    def store(
        self,
        new: NewMemory,
        *,
        dedupe: bool = True,
        detect: bool = True,
        relations: Sequence[ExtractedRelation] = (),
        event_kind: EventKind = EventKind.CREATED,
    ) -> RememberResult:
        now = self.clock()
        if dedupe:
            duplicate = self.find_duplicate(new.user_id, new.content, now)
            if duplicate is not None:
                return self._reinforce(duplicate, new, relations, now)
        vectors, warnings = self.embed_many([new.content])
        index, claim = self.index_for(new.content, vectors[0])
        record = MemoryRecord(
            id=new_id(),
            user_id=new.user_id,
            content=new.content,
            memory_type=new.memory_type,
            source=new.source,
            importance=new.importance,
            confidence=new.confidence,
            created_at=now,
            updated_at=now,
            expires_at=new.expires_at,
            metadata=dict(new.metadata),
            embedding_model=index.vector[0] if index.vector else None,
        )
        with self.storage.transaction():
            if dedupe:
                # Re-check exact duplicates inside the transaction to close the race
                # with concurrent writers of identical content.
                for existing in self.storage.find_by_hash(new.user_id, index.content_hash):
                    if not existing.is_expired(now):
                        return self._reinforce(existing, new, relations, now)
            self.storage.insert_memory(record, index)
            self._event(record, event_kind, now)
            superseded, conflicts = (
                self._handle_conflicts(record, claim, now) if detect and claim else ((), ())
            )
            self._add_relations(record, relations)
        return RememberResult(
            memory=self.storage.get_memory(record.id) or record,
            action=RememberAction.CREATED,
            superseded_ids=superseded,
            conflicts=conflicts,
            warnings=tuple(warnings),
        )

    def _reinforce(
        self,
        existing: MemoryRecord,
        new: NewMemory,
        relations: Sequence[ExtractedRelation],
        now: datetime,
    ) -> RememberResult:
        updated = existing.with_changes(
            mention_count=existing.mention_count + 1,
            importance=max(reinforce(existing.importance, self.importance), new.importance),
            confidence=max(existing.confidence, new.confidence),
            updated_at=now,
        )
        with self.storage.transaction():
            self.storage.update_memory(updated)
            self._event(
                updated,
                EventKind.REINFORCED,
                now,
                content=new.content,
                reason=f"repeated mention from source {new.source!r}",
            )
            self._add_relations(updated, relations)
        return RememberResult(memory=updated, action=RememberAction.REINFORCED)

    def update(
        self, existing: MemoryRecord, changes: Mapping[str, Any], *, detect: bool = True
    ) -> MemoryRecord:
        now = self.clock()
        content_changed = "content" in changes and changes["content"] != existing.content
        updated = existing.with_changes(
            **changes,
            version=existing.version + 1,
            updated_at=now,
        )
        index: IndexData | None = None
        claim: Claim | None = None
        if content_changed:
            vectors, _ = self.embed_many([updated.content])
            index, claim = self.index_for(updated.content, vectors[0])
            updated = updated.with_changes(
                embedding_model=index.vector[0] if index.vector else None
            )
        with self.storage.transaction():
            self.storage.update_memory(updated, index)
            self._event(updated, EventKind.UPDATED, now)
            if content_changed and detect and claim and updated.is_active:
                self._handle_conflicts(updated, claim, now)
        return self.storage.get_memory(updated.id) or updated

    def set_status(
        self,
        memory: MemoryRecord,
        status: MemoryStatus,
        kind: EventKind,
        *,
        reason: str | None = None,
        clear_superseded_by: bool = False,
    ) -> MemoryRecord:
        """Change a memory's status.  ``superseded_by`` is kept (it is history)
        unless ``clear_superseded_by`` is set, which ``restore`` does."""
        now = self.clock()
        updated = memory.with_changes(
            status=status.value,
            superseded_by=None if clear_superseded_by else memory.superseded_by,
            updated_at=now,
        )
        with self.storage.transaction():
            self.storage.update_memory(updated)
            self._event(updated, kind, now, reason=reason)
        return updated

    def _event(
        self,
        memory: MemoryRecord,
        kind: EventKind,
        now: datetime,
        *,
        content: str | None = None,
        related_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.storage.add_event(
            MemoryEvent(
                memory_id=memory.id,
                user_id=memory.user_id,
                kind=kind.value,
                version=memory.version,
                content=content if content is not None else memory.content,
                created_at=now,
                related_id=related_id,
                reason=reason,
            )
        )

    def _add_relations(self, memory: MemoryRecord, relations: Sequence[ExtractedRelation]) -> None:
        for rel in relations:
            try:
                self.graph.add(
                    memory.user_id,
                    rel.source,
                    rel.relation,
                    rel.target,
                    confidence=memory.confidence,
                    memory_id=memory.id,
                    metadata=rel.metadata,
                )
            except ValidationError as exc:
                logger.debug("Skipping invalid extracted relation %r: %s", rel, exc)

    # -- conflicts --------------------------------------------------------
    def _handle_conflicts(
        self, record: MemoryRecord, claim: Claim, now: datetime
    ) -> tuple[tuple[str, ...], tuple[Conflict, ...]]:
        existing = [
            m
            for m in self.storage.find_by_slot(record.user_id, claim.slot)
            if m.id != record.id and not m.is_expired(now)
        ]
        superseded: list[str] = []
        conflicts: list[Conflict] = []
        for found in detect_conflicts(claim, existing):
            replaced, conflict = self._resolve(record, found.existing, found.kind, now)
            if replaced:
                superseded.append(found.existing.id)
            conflicts.append(conflict)
        return tuple(superseded), tuple(conflicts)

    def _resolve(
        self, newer: MemoryRecord, older: MemoryRecord, kind: ConflictKind, now: datetime
    ) -> tuple[bool, Conflict]:
        replace = should_supersede(
            self.conflict_policy,
            new_confidence=newer.confidence,
            old_confidence=older.confidence,
        )
        if replace:
            reason = (
                f"superseded by newer memory {newer.id} "
                f"(policy: {self.conflict_policy.value}, {kind.value})"
            )
            self.storage.update_memory(
                older.with_changes(
                    status=MemoryStatus.SUPERSEDED.value, superseded_by=newer.id, updated_at=now
                )
            )
            self._event(older, EventKind.SUPERSEDED, now, related_id=newer.id, reason=reason)
            conflict = Conflict(
                id=new_id(),
                user_id=newer.user_id,
                memory_id=newer.id,
                other_id=older.id,
                kind=kind.value,
                status=ConflictStatus.RESOLVED.value,
                created_at=now,
                resolution=reason,
                resolved_at=now,
            )
        else:
            why = (
                "policy keep_both"
                if self.conflict_policy == ConflictPolicy.KEEP_BOTH
                else "newer memory has lower confidence than the existing one"
            )
            self._event(
                newer,
                EventKind.CONFLICT_DETECTED,
                now,
                related_id=older.id,
                reason=f"{kind.value}; both kept active ({why})",
            )
            conflict = Conflict(
                id=new_id(),
                user_id=newer.user_id,
                memory_id=newer.id,
                other_id=older.id,
                kind=kind.value,
                status=ConflictStatus.OPEN.value,
                created_at=now,
            )
        self.storage.add_conflict(conflict)
        return replace, conflict

    def resolve_conflict(self, conflict: Conflict, keep: str) -> Conflict:
        """Make ``keep`` the current belief and supersede the other memory."""
        if keep not in (conflict.memory_id, conflict.other_id):
            raise ValidationError(
                f"Memory {keep!r} is not part of conflict {conflict.id!r}.",
                hint=f"Choose {conflict.memory_id!r} or {conflict.other_id!r}.",
            )
        drop = conflict.other_id if keep == conflict.memory_id else conflict.memory_id
        records = self.storage.get_memories([keep, drop])
        if keep not in records or drop not in records:
            raise MemoryNotFoundError(keep if keep not in records else drop)
        now = self.clock()
        reason = f"resolved manually: kept {keep}"
        with self.storage.transaction():
            kept = records[keep]
            if not kept.is_active:
                kept = kept.with_changes(
                    status=MemoryStatus.ACTIVE.value, superseded_by=None, updated_at=now
                )
                self.storage.update_memory(kept)
                self._event(kept, EventKind.RESTORED, now, reason=reason)
            dropped = records[drop]
            if dropped.status != MemoryStatus.SUPERSEDED or dropped.superseded_by != keep:
                self.storage.update_memory(
                    dropped.with_changes(
                        status=MemoryStatus.SUPERSEDED.value, superseded_by=keep, updated_at=now
                    )
                )
                self._event(dropped, EventKind.SUPERSEDED, now, related_id=keep, reason=reason)
            resolved = Conflict(
                id=conflict.id,
                user_id=conflict.user_id,
                memory_id=conflict.memory_id,
                other_id=conflict.other_id,
                kind=conflict.kind,
                status=ConflictStatus.RESOLVED.value,
                created_at=conflict.created_at,
                resolution=reason,
                resolved_at=now,
            )
            self.storage.update_conflict(resolved)
        return resolved

    # -- consolidation ----------------------------------------------------
    def iter_user_memories(
        self, user_id: str, *, statuses: tuple[str, ...], include_expired: bool, now: datetime
    ) -> Iterator[MemoryRecord]:
        flt = MemoryFilter(
            user_id=user_id, statuses=statuses, include_expired=include_expired, now=now
        )
        offset = 0
        while True:
            page = self.storage.list_memories(
                flt, limit=SCAN_PAGE_SIZE, offset=offset, order="oldest"
            )
            yield from page
            if len(page) < SCAN_PAGE_SIZE:
                return
            offset += SCAN_PAGE_SIZE

    def consolidate(
        self,
        user_id: str,
        *,
        dry_run: bool,
        summarizer: Summarizer | None,
    ) -> ConsolidationReport:
        now = self.clock()
        actions: list[str] = []
        active = list(
            self.iter_user_memories(
                user_id, statuses=(MemoryStatus.ACTIVE.value,), include_expired=True, now=now
            )
        )

        # 1. Archive expired memories (their expiry was set explicitly).
        expired = [m for m in active if m.is_expired(now)]
        for memory in expired:
            actions.append(f"archive expired memory {memory.short_id}")
            if not dry_run:
                self.set_status(memory, MemoryStatus.ARCHIVED, EventKind.ARCHIVED, reason="expired")
        live = [m for m in active if not m.is_expired(now)]

        # 2. Merge near-duplicates into one surviving memory.
        merged = self._merge_duplicates(live, now, actions, dry_run)
        merged_ids = {loser for _, loser in merged}
        live = [m for m in live if m.id not in merged_ids]

        # 3. Detect contradictions that were not caught at write time.
        superseded, conflict_count = self._consolidate_conflicts(
            user_id, live, now, actions, dry_run
        )
        superseded_ids = {old for _, old in superseded}
        live = [m for m in live if m.id not in superseded_ids]

        # 4. Optionally summarize groups of related memories (additive only).
        summaries: list[str] = []
        if summarizer is not None:
            summaries = self._summarize(user_id, live, summarizer, actions, dry_run)

        return ConsolidationReport(
            user_id=user_id,
            dry_run=dry_run,
            merged=tuple(merged),
            superseded=tuple(superseded),
            conflicts_detected=conflict_count,
            expired_archived=tuple(m.id for m in expired),
            summaries_created=tuple(summaries),
            actions=tuple(actions),
        )

    def _merge_duplicates(
        self, memories: list[MemoryRecord], now: datetime, actions: list[str], dry_run: bool
    ) -> list[tuple[str, str]]:
        by_id = {m.id: m for m in memories}
        pairs = find_duplicate_pairs(
            [(m.id, m.content) for m in memories], threshold=self.dedup_threshold
        )
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        for a, b, _ in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        groups: dict[str, list[MemoryRecord]] = {}
        for memory_id in {i for pair in pairs for i in pair[:2]}:
            groups.setdefault(find(memory_id), []).append(by_id[memory_id])
        merged: list[tuple[str, str]] = []
        for members in sorted(groups.values(), key=lambda g: min(m.id for m in g)):
            members.sort(key=lambda m: (-m.mention_count, m.created_at, m.id))
            survivor, losers = members[0], members[1:]
            for loser in losers:
                merged.append((survivor.id, loser.id))
                actions.append(f"merge {loser.short_id} into {survivor.short_id}")
            if dry_run:
                continue
            with self.storage.transaction():
                combined = survivor.with_changes(
                    mention_count=sum(m.mention_count for m in members),
                    access_count=sum(m.access_count for m in members),
                    importance=max(m.importance for m in members),
                    confidence=max(m.confidence for m in members),
                    updated_at=now,
                )
                self.storage.update_memory(combined)
                for loser in losers:
                    self.storage.update_memory(
                        loser.with_changes(
                            status=MemoryStatus.MERGED.value,
                            superseded_by=survivor.id,
                            updated_at=now,
                        )
                    )
                    self._event(
                        loser,
                        EventKind.MERGED,
                        now,
                        related_id=survivor.id,
                        reason="near-duplicate merged during consolidation",
                    )
                    self._event(
                        combined,
                        EventKind.MERGED,
                        now,
                        content=loser.content,
                        related_id=loser.id,
                        reason="absorbed near-duplicate",
                    )
        return merged

    def _consolidate_conflicts(
        self,
        user_id: str,
        memories: list[MemoryRecord],
        now: datetime,
        actions: list[str],
        dry_run: bool,
    ) -> tuple[list[tuple[str, str]], int]:
        known_pairs = {
            frozenset((c.memory_id, c.other_id)) for c in self.storage.list_conflicts(user_id)
        }
        by_slot: dict[str, list[tuple[MemoryRecord, Claim]]] = {}
        for memory in memories:
            claim = parse_claim(memory.content)
            if claim is not None:
                by_slot.setdefault(claim.slot, []).append((memory, claim))
        superseded: list[tuple[str, str]] = []
        count = 0
        for slot in sorted(by_slot):
            entries = sorted(by_slot[slot], key=lambda e: (e[0].created_at, e[0].id))
            current: list[tuple[MemoryRecord, Claim]] = []
            for newer, claim in entries:
                still_current: list[tuple[MemoryRecord, Claim]] = []
                for older, older_claim in current:
                    if (
                        older_claim.value == claim.value
                        or frozenset((newer.id, older.id)) in known_pairs
                    ):
                        still_current.append((older, older_claim))
                        continue
                    count += 1
                    kind = ConflictKind.CONTRADICTION if claim.polar else ConflictKind.CHANGED_VALUE
                    replace = should_supersede(
                        self.conflict_policy,
                        new_confidence=newer.confidence,
                        old_confidence=older.confidence,
                    )
                    actions.append(
                        f"{'supersede' if replace else 'flag conflict between'} {older.short_id} "
                        f"{'with' if replace else 'and'} {newer.short_id} ({slot})"
                    )
                    if not dry_run:
                        with self.storage.transaction():
                            self._resolve(newer, older, kind, now)
                    if replace:
                        superseded.append((newer.id, older.id))
                    else:
                        still_current.append((older, older_claim))
                current = [*still_current, (newer, claim)]
        return superseded, count

    def _summarize(
        self,
        user_id: str,
        memories: list[MemoryRecord],
        summarizer: Summarizer,
        actions: list[str],
        dry_run: bool,
    ) -> list[str]:
        candidates = [m for m in memories if m.source != SUMMARY_SOURCE]
        created: list[str] = []
        for group in group_by_entity(candidates)[:MAX_SUMMARIES_PER_PASS]:
            ids = [m.id for m in group.memories]
            key = summary_key(ids)
            existing = self.storage.count_memories(
                MemoryFilter(user_id=user_id, statuses=(), metadata={"summary_key": key})
            )
            if existing:
                continue
            actions.append(f"summarize {len(ids)} memories about {group.topic}")
            if dry_run:
                continue
            content = summarizer.summarize(group.topic, [m.content for m in group.memories])
            result = self.store(
                NewMemory(
                    user_id=user_id,
                    content=content,
                    memory_type="knowledge",
                    source=SUMMARY_SOURCE,
                    importance=max(m.importance for m in group.memories),
                    confidence=min(m.confidence for m in group.memories),
                    metadata={"summary_of": ids, "summary_key": key, "topic": group.topic},
                ),
                dedupe=False,
                detect=False,
            )
            created.append(result.memory.id)
        return created
