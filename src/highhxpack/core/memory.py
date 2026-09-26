"""The public :class:`Memory` API."""

from __future__ import annotations

import builtins
import re
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from highhxpack.consolidation.conflict import parse_policy
from highhxpack.consolidation.importance import estimate_importance
from highhxpack.consolidation.summarization import ExtractiveSummarizer, LLMSummarizer, Summarizer
from highhxpack.core.config import Config
from highhxpack.core.lifecycle import Clock, compute_expiry, system_clock
from highhxpack.core.manager import MemoryManager, NewMemory
from highhxpack.core.transfer import OnExisting, export_memories, import_memories
from highhxpack.embeddings.base import EmbeddingProvider
from highhxpack.embeddings.providers import create_embedding_provider
from highhxpack.exceptions import (
    ConfigurationError,
    ConflictNotFoundError,
    HighHXPackError,
    MemoryNotFoundError,
    ValidationError,
)
from highhxpack.extraction.extractor import (
    Extractor,
    LLMExtractor,
    RuleBasedExtractor,
    classify_memory_type,
)
from highhxpack.graph.graph import KnowledgeGraph
from highhxpack.models.event import EventKind, MemoryEvent
from highhxpack.models.memory import MemoryRecord, MemoryStatus
from highhxpack.models.relationship import Conflict, ConflictStatus
from highhxpack.models.result import (
    ConsolidationReport,
    ExportReport,
    ImportReport,
    MemoryStats,
    RecallResult,
    RememberResult,
)
from highhxpack.models.user import UserSummary
from highhxpack.providers import create_llm_provider
from highhxpack.providers.base import LLMProvider
from highhxpack.retrieval.filters import JSONScalar, build_filter
from highhxpack.retrieval.retriever import Retriever, Strategy, parse_strategy
from highhxpack.storage.base import ListOrder, StorageBackend
from highhxpack.storage.sqlite import SQLiteStorage
from highhxpack.utils.logging import get_logger
from highhxpack.utils.timestamps import to_storage
from highhxpack.utils.validation import (
    validate_content,
    validate_limit,
    validate_memory_id,
    validate_memory_type,
    validate_metadata,
    validate_offset,
    validate_path,
    validate_query,
    validate_source,
    validate_unit_interval,
    validate_user_id,
)

logger = get_logger("memory")

_PREFIX_RE = re.compile(r"^[0-9a-f]{4,32}$")
_LIST_ORDERS = ("newest", "oldest", "importance", "updated")
_MAX_INGEST_CHARS = 200_000


class Memory:
    """A local-first memory store.

    ``Memory()`` stores data in a SQLite file in the platform data directory and
    needs no API keys or network access.  Everything is configurable through
    :class:`Config`, a configuration file, environment variables, or these
    arguments (see :mod:`highhxpack.core.config` for precedence).

    Args:
        storage_path: Database file (``":memory:"`` for a temporary in-memory store).
        config: A fully resolved :class:`Config`; when omitted, configuration is
            loaded from the config file and environment.
        storage: A custom :class:`StorageBackend`; overrides ``storage_path``.
        embedder: An :class:`EmbeddingProvider`, a provider name, or ``"none"``.
        llm: An :class:`LLMProvider`, a provider name, or ``"none"``.  Created on
            first use, so a missing optional dependency only affects LLM features.
        clock: Function returning the current UTC time (useful in tests).

    ``Memory`` is thread-safe and can be used as a context manager; call
    :meth:`close` when done.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        *,
        config: Config | None = None,
        storage: StorageBackend | None = None,
        embedder: EmbeddingProvider | str | None = None,
        llm: LLMProvider | str | None = None,
        clock: Clock | None = None,
    ):
        if config is None:
            config = Config.load(
                storage_path=(
                    storage_path
                    if storage_path is None or str(storage_path) == ":memory:"
                    else validate_path(storage_path, field="storage_path")
                )
            )
        elif storage_path is not None:
            config = config.replace(storage_path=storage_path)
        self._config = config
        self._clock: Clock = clock or system_clock
        self._embedder = self._resolve_embedder(embedder, config)
        # The LLM is created on first use: a missing optional package or API key
        # must not break operations that never need an LLM.
        if llm is not None and not isinstance(llm, LLMProvider | str):
            raise ConfigurationError("llm must be an LLMProvider or a provider name.")
        self._llm_spec: LLMProvider | str | None = llm
        self._llm: LLMProvider | None = None
        self._llm_resolved = False
        self._llm_lock = threading.Lock()
        policy = parse_policy(config.conflict_policy)
        # Open storage last so that no earlier failure can leak a connection.
        self._storage = storage or SQLiteStorage(config.database_path)
        self._owns_storage = storage is None
        self._graph = KnowledgeGraph(self._storage, self._clock)
        self._retriever = Retriever(self._storage, self._embedder, config.scoring)
        self._manager = MemoryManager(
            self._storage,
            self._embedder,
            self._graph,
            clock=self._clock,
            scoring=config.scoring,
            conflict_policy=policy,
            dedup_threshold=config.dedup_threshold,
        )

    @staticmethod
    def _resolve_embedder(
        embedder: EmbeddingProvider | str | None, config: Config
    ) -> EmbeddingProvider | None:
        if isinstance(embedder, EmbeddingProvider):
            return embedder
        if embedder is not None and not isinstance(embedder, str):
            raise ConfigurationError("embedder must be an EmbeddingProvider or a provider name.")
        return create_embedding_provider(
            embedder or config.embedding_provider,
            model=config.embedding_model if embedder is None else None,
            ollama_url=config.ollama_url,
            openai_base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            timeout=config.request_timeout,
        )

    @staticmethod
    def _resolve_llm(llm: LLMProvider | str | None, config: Config) -> LLMProvider | None:
        if isinstance(llm, LLMProvider):
            return llm
        if llm is not None and not isinstance(llm, str):
            raise ConfigurationError("llm must be an LLMProvider or a provider name.")
        return create_llm_provider(
            llm or config.llm_provider,
            model=config.llm_model if llm is None else None,
            ollama_url=config.ollama_url,
            openai_base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            timeout=config.request_timeout,
        )

    # -- properties -------------------------------------------------------
    @property
    def config(self) -> Config:
        return self._config

    @property
    def storage(self) -> StorageBackend:
        return self._storage

    @property
    def embedder(self) -> EmbeddingProvider | None:
        return self._embedder

    @property
    def llm(self) -> LLMProvider | None:
        """The configured LLM, created on first access (``None`` when disabled)."""
        with self._llm_lock:
            if not self._llm_resolved:
                self._llm = self._resolve_llm(self._llm_spec, self._config)
                self._llm_resolved = True
        return self._llm

    @property
    def graph(self) -> KnowledgeGraph:
        """The optional knowledge graph (see :class:`KnowledgeGraph`)."""
        return self._graph

    @property
    def location(self) -> str:
        """Where the data is stored (a file path or ``":memory:"``)."""
        return self._storage.location

    def __repr__(self) -> str:
        embedder = self._embedder.name if self._embedder else "none"
        return f"Memory(location={self.location!r}, embedder={embedder!r})"

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        """Close the underlying storage.  Safe to call more than once."""
        if self._owns_storage:
            self._storage.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- writing ----------------------------------------------------------
    def remember(
        self,
        user_id: str,
        content: str,
        *,
        memory_type: str | None = None,
        source: str = "api",
        importance: float | None = None,
        confidence: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
        ttl: str | timedelta | None = None,
        expires_at: str | datetime | None = None,
        dedupe: bool = True,
        detect_conflicts: bool = True,
    ) -> RememberResult:
        """Store one memory.

        Content is stored as given (whitespace-trimmed).  When it duplicates an
        existing active memory, that memory is reinforced instead of storing a
        copy (``result.action == "reinforced"``).  When it contradicts an
        existing memory, the configured conflict policy decides which one is
        current; the other is kept as history (see ``result.conflicts``).

        Args:
            memory_type: Any lower-case identifier; inferred from the text when omitted.
            importance: 0-1; estimated from the type and wording when omitted.
            confidence: 0-1, how certain the information is.
            ttl: Lifetime such as ``"30d"``; alternatively pass ``expires_at``.
        """
        uid = validate_user_id(user_id)
        text = validate_content(content)
        mtype = (
            validate_memory_type(memory_type)
            if memory_type is not None
            else (classify_memory_type(text))
        )
        new = NewMemory(
            user_id=uid,
            content=text,
            memory_type=mtype,
            source=validate_source(source),
            importance=(
                validate_unit_interval(importance, field="importance")
                if importance is not None
                else estimate_importance(text, mtype)
            ),
            confidence=validate_unit_interval(confidence, field="confidence"),
            metadata=validate_metadata(metadata),
            expires_at=compute_expiry(self._clock(), ttl=ttl, expires_at=expires_at),
        )
        return self._manager.store(new, dedupe=dedupe, detect=detect_conflicts)

    def ingest(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "conversation",
        extractor: Extractor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> builtins.list[RememberResult]:
        """Extract memorable statements from free text and store them.

        Questions, greetings and small talk are ignored.  Relationships found
        by extraction ("I prefer C++" -> ``user --prefers--> C++``) are added to
        the knowledge graph.  The default extractor is rule-based and offline;
        configure ``extractor = "llm"`` (or pass one) to use an LLM instead.
        """
        uid = validate_user_id(user_id)
        if not isinstance(text, str):
            raise ValidationError(f"text must be a string, got {type(text).__name__}.")
        if len(text) > _MAX_INGEST_CHARS:
            raise ValidationError(
                f"text is too long to ingest ({len(text)} characters).",
                hint=f"Split it into chunks of at most {_MAX_INGEST_CHARS} characters.",
            )
        src = validate_source(source)
        meta = validate_metadata(metadata)
        chosen = extractor or self._default_extractor()
        results: list[RememberResult] = []
        for item in chosen.extract(text):
            importance = (
                item.importance
                if item.importance is not None
                else estimate_importance(item.content, item.memory_type)
            )
            new = NewMemory(
                user_id=uid,
                content=validate_content(item.content),
                memory_type=validate_memory_type(item.memory_type),
                source=src,
                importance=importance,
                confidence=item.confidence,
                metadata=meta,
            )
            results.append(self._manager.store(new, relations=item.relations))
        return results

    def _default_extractor(self) -> Extractor:
        if self._config.extractor == "llm":
            llm = self.llm
            if llm is None:
                raise ConfigurationError(
                    "extractor = 'llm' but no LLM provider is configured.",
                    hint="Set llm_provider, pass llm=..., or use extractor = 'rules'.",
                )
            return LLMExtractor(llm)
        return RuleBasedExtractor()

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        memory_type: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        ttl: str | timedelta | None = None,
        expires_at: str | datetime | None = None,
        clear_expiry: bool = False,
    ) -> MemoryRecord:
        """Change a memory.  The version is incremented and the change is recorded
        in :meth:`history`; the previous content remains available there."""
        existing = self.get(memory_id)
        changes: dict[str, Any] = {}
        if content is not None:
            changes["content"] = validate_content(content)
        if memory_type is not None:
            changes["memory_type"] = validate_memory_type(memory_type)
        if importance is not None:
            changes["importance"] = validate_unit_interval(importance, field="importance")
        if confidence is not None:
            changes["confidence"] = validate_unit_interval(confidence, field="confidence")
        if metadata is not None:
            changes["metadata"] = validate_metadata(metadata)
        if clear_expiry:
            if ttl is not None or expires_at is not None:
                raise ValidationError("clear_expiry cannot be combined with ttl or expires_at.")
            changes["expires_at"] = None
        elif ttl is not None or expires_at is not None:
            changes["expires_at"] = compute_expiry(self._clock(), ttl=ttl, expires_at=expires_at)
        if not changes:
            raise ValidationError("Nothing to update.", hint="Pass at least one field to change.")
        return self._manager.update(existing, changes)

    def forget(self, memory_id: str, *, reason: str | None = None) -> MemoryRecord:
        """Archive a memory: hide it from recall while keeping it in history.

        Use :meth:`restore` to undo, or :meth:`delete` to erase it permanently.
        """
        memory = self.get(memory_id)
        if memory.status == MemoryStatus.ARCHIVED:
            return memory
        return self._manager.set_status(
            memory, MemoryStatus.ARCHIVED, EventKind.ARCHIVED, reason=reason or "forgotten by user"
        )

    def restore(self, memory_id: str) -> MemoryRecord:
        """Make an archived, superseded or merged memory active again."""
        memory = self.get(memory_id)
        if memory.is_active:
            return memory
        return self._manager.set_status(
            memory,
            MemoryStatus.ACTIVE,
            EventKind.RESTORED,
            reason="restored by user",
            clear_superseded_by=True,
        )

    def delete(self, memory_id: str) -> None:
        """Permanently erase a memory, its history, vector and derived graph edges."""
        mid = validate_memory_id(memory_id)
        if not self._storage.delete_memory(mid):
            raise MemoryNotFoundError(mid)

    def clear(self, user_id: str | None = None, *, all_users: bool = False) -> int:
        """Permanently delete all data of ``user_id`` (or of everyone with ``all_users=True``).

        Returns the number of memories deleted.
        """
        if user_id is not None and all_users:
            raise ValidationError("Pass either user_id or all_users=True, not both.")
        if user_id is not None:
            return self._storage.delete_user(validate_user_id(user_id))
        if all_users:
            return self._storage.delete_all()
        raise ValidationError(
            "clear() needs a user_id.",
            hint="Pass clear(user_id) or clear(all_users=True) to erase everything.",
        )

    # -- reading ----------------------------------------------------------
    def get(self, memory_id: str) -> MemoryRecord:
        """Return a memory by id (any status).  Raises :class:`MemoryNotFoundError`."""
        mid = validate_memory_id(memory_id)
        memory = self._storage.get_memory(mid)
        if memory is None:
            raise MemoryNotFoundError(mid)
        return memory

    def recall(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 10,
        memory_type: str | Iterable[str] | None = None,
        min_score: float = 0.0,
        strategy: str | Strategy = Strategy.HYBRID,
        source: str | Iterable[str] | None = None,
        metadata: Mapping[str, JSONScalar] | None = None,
        created_after: str | datetime | None = None,
        created_before: str | datetime | None = None,
    ) -> builtins.list[RecallResult]:
        """Return a user's current memories most relevant to ``query``, best first.

        Only active, unexpired memories are considered.  Each result carries a
        :class:`ScoreBreakdown` explaining its score.  Recalled memories have
        their access statistics updated (disable with ``track_access = false``).
        """
        uid = validate_user_id(user_id)
        results = self._retrieve(
            query,
            user_id=uid,
            limit=limit,
            memory_type=memory_type,
            min_score=min_score,
            strategy=strategy,
            source=source,
            metadata=metadata,
            created_after=created_after,
            created_before=created_before,
            status=MemoryStatus.ACTIVE.value,
            include_expired=False,
        )
        if results and self._config.track_access:
            try:
                self._storage.record_access([r.id for r in results], self._clock())
            except HighHXPackError as exc:  # access stats are best-effort
                logger.warning("Could not record memory access: %s", exc)
        return results

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 10,
        strategy: str | Strategy = Strategy.HYBRID,
        memory_type: str | Iterable[str] | None = None,
        status: str | Iterable[str] | None = MemoryStatus.ACTIVE.value,
        min_score: float = 0.0,
        include_expired: bool = False,
        source: str | Iterable[str] | None = None,
        metadata: Mapping[str, JSONScalar] | None = None,
    ) -> builtins.list[RecallResult]:
        """Search memories, optionally across all users and statuses.

        Unlike :meth:`recall`, ``search`` does not update access statistics and
        can include superseded or archived memories (``status="any"``).
        """
        return self._retrieve(
            query,
            user_id=validate_user_id(user_id) if user_id is not None else None,
            limit=limit,
            memory_type=memory_type,
            min_score=min_score,
            strategy=strategy,
            source=source,
            metadata=metadata,
            status=status,
            include_expired=include_expired,
        )

    def _retrieve(
        self,
        query: str,
        *,
        user_id: str | None,
        limit: int,
        memory_type: str | Iterable[str] | None,
        min_score: float,
        strategy: str | Strategy,
        source: str | Iterable[str] | None,
        metadata: Mapping[str, JSONScalar] | None,
        status: str | Iterable[str] | None,
        include_expired: bool,
        created_after: str | datetime | None = None,
        created_before: str | datetime | None = None,
    ) -> builtins.list[RecallResult]:
        text = validate_query(query)
        now = self._clock()
        flt = build_filter(
            user_id=user_id,
            memory_type=memory_type,
            source=source,
            status=status,
            include_expired=include_expired,
            now=now,
            created_after=created_after,
            created_before=created_before,
            metadata=metadata,
        )
        return self._retriever.retrieve(
            text,
            flt,
            now=now,
            limit=validate_limit(limit),
            strategy=parse_strategy(strategy),
            min_score=validate_unit_interval(min_score, field="min_score"),
        )

    def count(
        self,
        user_id: str | None = None,
        *,
        memory_type: str | Iterable[str] | None = None,
        status: str | Iterable[str] | None = MemoryStatus.ACTIVE.value,
        include_expired: bool = False,
    ) -> int:
        """Number of memories matching the filters."""
        flt = build_filter(
            user_id=user_id,
            memory_type=memory_type,
            status=status,
            include_expired=include_expired,
            now=self._clock(),
        )
        return self._storage.count_memories(flt)

    def history(self, memory_id: str) -> builtins.list[MemoryEvent]:
        """All recorded changes of a memory, oldest first."""
        memory = self.get(memory_id)
        return self._storage.get_events(memory.id)

    def users(self) -> dict[str, int]:
        """Map of user id to number of stored memories (any status)."""
        return dict(self._storage.users())

    def inspect(self, user_id: str) -> UserSummary:
        """Overview of everything stored for one user."""
        uid = validate_user_id(user_id)
        stats = self._storage.stats(uid)
        return UserSummary(
            user_id=uid,
            total=stats.total,
            active=stats.active,
            by_type=stats.by_type,
            by_status=stats.by_status,
            open_conflicts=stats.open_conflicts,
            relationships=stats.relationships,
            first_memory_at=stats.first_memory_at,
            last_memory_at=stats.last_memory_at,
        )

    def stats(self, user_id: str | None = None) -> MemoryStats:
        """Storage statistics for everything, or for one user."""
        uid = validate_user_id(user_id) if user_id is not None else None
        stats = self._storage.stats(uid)
        missing = (
            len(self._storage.ids_missing_vector(self._embedder.name, uid)) if self._embedder else 0
        )
        return MemoryStats(
            user_id=uid,
            total=stats.total,
            active=stats.active,
            users=stats.users,
            by_type=stats.by_type,
            by_status=stats.by_status,
            relationships=stats.relationships,
            open_conflicts=stats.open_conflicts,
            vectors=stats.vectors,
            missing_vectors=missing,
            storage_path=self._storage.location,
            storage_bytes=stats.storage_bytes,
            schema_version=self._storage.schema_version,
            embedding_model=self._embedder.name if self._embedder else None,
        )

    # -- conflicts & consolidation ---------------------------------------
    def conflicts(
        self, user_id: str | None = None, *, status: str | None = ConflictStatus.OPEN.value
    ) -> builtins.list[Conflict]:
        """Recorded conflicts; ``status=None`` returns resolved ones too."""
        if status is not None and status not in {s.value for s in ConflictStatus}:
            raise ValidationError(
                f"Unknown conflict status: {status!r}", hint="Use 'open' or 'resolved'."
            )
        uid = validate_user_id(user_id) if user_id is not None else None
        return self._storage.list_conflicts(uid, status=status)

    def resolve_conflict(self, conflict_id: str, *, keep: str) -> Conflict:
        """Resolve a conflict by choosing which memory is the current belief."""
        cid = validate_memory_id(conflict_id)
        conflict = self._storage.get_conflict(cid)
        if conflict is None:
            raise ConflictNotFoundError(cid)
        return self._manager.resolve_conflict(conflict, validate_memory_id(keep))

    def consolidate(
        self,
        user_id: str,
        *,
        dry_run: bool = False,
        summarize: bool = False,
        summarizer: Summarizer | None = None,
    ) -> ConsolidationReport:
        """Tidy a user's memories.

        Archives expired memories, merges near-duplicates (the survivor keeps
        the combined mention counts; merged memories stay in history), and
        applies the conflict policy to contradictions that were not caught when
        stored.  With ``summarize=True`` it also adds ``knowledge`` summaries of
        groups of related memories; originals are never modified.  Use
        ``dry_run=True`` to preview.
        """
        uid = validate_user_id(user_id)
        chosen: Summarizer | None = summarizer
        if summarize and chosen is None:
            llm = self.llm
            chosen = LLMSummarizer(llm) if llm is not None else ExtractiveSummarizer()
        return self._manager.consolidate(uid, dry_run=dry_run, summarizer=chosen)

    def reindex(self, user_id: str | None = None) -> int:
        """Compute missing or outdated embeddings; returns how many were updated."""
        if self._embedder is None:
            raise ConfigurationError(
                "Embeddings are disabled, so there is nothing to reindex.",
                hint="Configure an embedding provider first.",
            )
        uid = validate_user_id(user_id) if user_id is not None else None
        ids = self._storage.ids_missing_vector(self._embedder.name, uid)
        updated = 0
        batch_size = 64
        for start in range(0, len(ids), batch_size):
            records = self._storage.get_memories(ids[start : start + batch_size])
            batch = [records[i] for i in ids[start : start + batch_size] if i in records]
            vectors, warnings = self._manager.embed_many([m.content for m in batch])
            if warnings:
                raise HighHXPackError(
                    "Reindexing stopped because embedding failed.", reason=warnings[0]
                )
            with self._storage.transaction():
                for memory, vector in zip(batch, vectors, strict=True):
                    if vector is not None:
                        self._storage.set_vector(memory.id, self._embedder.name, vector)
                        updated += 1
        return updated

    # -- import / export --------------------------------------------------
    def export(
        self, path: str | Path, *, user_id: str | None = None, overwrite: bool = False
    ) -> ExportReport:
        """Write memories, history, relationships and conflicts to a JSON file.

        The file is created with owner-only permissions.  Pass ``user_id`` to
        export a single user.
        """
        target = validate_path(path)
        uid = validate_user_id(user_id) if user_id is not None else None
        return export_memories(
            self._manager,
            target,
            user_id=uid,
            overwrite=overwrite,
            now_iso=to_storage(self._clock()),
        )

    def import_(
        self,
        path: str | Path,
        *,
        user_id: str | None = None,
        on_existing: OnExisting = "skip",
    ) -> ImportReport:
        """Import a file produced by :meth:`export`.

        Every record is validated; invalid ones are skipped and listed in
        ``report.errors``.  ``user_id`` re-assigns all imported data to one user.
        Memories whose id already exists are skipped unless
        ``on_existing="replace"``.
        """
        if on_existing not in ("skip", "replace"):
            raise ValidationError("on_existing must be 'skip' or 'replace'.")
        uid = validate_user_id(user_id) if user_id is not None else None
        return import_memories(
            self._manager, validate_path(path), user_id=uid, on_existing=on_existing
        )

    # -- helpers ----------------------------------------------------------
    def resolve_id(self, id_or_prefix: str) -> str:
        """Expand a unique id prefix (at least 4 hex characters) to a full memory id."""
        if not isinstance(id_or_prefix, str):
            raise ValidationError("memory id must be a string.")
        prefix = id_or_prefix.strip().lower()
        if not _PREFIX_RE.match(prefix):
            raise ValidationError(
                f"Invalid memory id: {id_or_prefix!r}",
                hint="Use the full id or at least its first 4 characters.",
            )
        matches = self._storage.resolve_prefix(prefix, limit=2)
        if not matches:
            raise MemoryNotFoundError(prefix)
        if len(matches) > 1:
            raise ValidationError(
                f"Memory id prefix {prefix!r} is ambiguous.", hint="Use more characters."
            )
        return matches[0]

    def list(
        self,
        user_id: str | None = None,
        *,
        memory_type: str | Iterable[str] | None = None,
        status: str | Iterable[str] | None = MemoryStatus.ACTIVE.value,
        limit: int = 50,
        offset: int = 0,
        order: ListOrder = "newest",
        include_expired: bool = False,
    ) -> builtins.list[MemoryRecord]:
        """List memories (active and unexpired by default), newest first."""
        if order not in _LIST_ORDERS:
            raise ValidationError(
                f"Unknown order: {order!r}", hint=f"Use one of: {', '.join(_LIST_ORDERS)}."
            )
        flt = build_filter(
            user_id=user_id,
            memory_type=memory_type,
            status=status,
            include_expired=include_expired,
            now=self._clock(),
        )
        return self._storage.list_memories(
            flt, limit=validate_limit(limit), offset=validate_offset(offset), order=order
        )


__all__ = ["Memory"]
