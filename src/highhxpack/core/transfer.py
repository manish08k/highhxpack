"""Export and import of memories as JSON.

The export format is plain JSON (never pickle), so importing a file can never
execute code.  Every imported record is validated exactly like API input;
invalid records are rejected individually and reported.  Vectors are not
exported: they are recomputed on import with the configured embedding provider.

Format (``format_version`` 1)::

    {
      "format": "highhxpack-export",
      "format_version": 1,
      "highhxpack_version": "0.1.0",
      "exported_at": "2024-05-01T12:00:00.000000Z",
      "user_id": null,
      "memories": [...], "events": [...], "relationships": [...], "conflicts": [...]
    }
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from highhxpack.__version__ import __version__
from highhxpack.core.manager import MemoryManager
from highhxpack.exceptions import ExportError, HighHXPackError, MemoryImportError
from highhxpack.graph.edges import normalize_relation
from highhxpack.graph.nodes import normalize_node
from highhxpack.models.event import EventKind, MemoryEvent
from highhxpack.models.memory import MemoryRecord, MemoryStatus
from highhxpack.models.relationship import Conflict, ConflictStatus, Relationship
from highhxpack.models.result import ExportReport, ImportReport
from highhxpack.retrieval.filters import MemoryFilter
from highhxpack.storage.base import StorageBackend
from highhxpack.utils.hashing import new_id
from highhxpack.utils.timestamps import parse_datetime
from highhxpack.utils.validation import (
    validate_content,
    validate_memory_id,
    validate_memory_type,
    validate_metadata,
    validate_source,
    validate_unit_interval,
    validate_user_id,
)

EXPORT_FORMAT = "highhxpack-export"
EXPORT_FORMAT_VERSION = 1
#: Refuse to read import files larger than this (protects against memory exhaustion).
MAX_IMPORT_BYTES = 256 * 1024 * 1024
MAX_REPORTED_ERRORS = 100
_EMBED_BATCH = 64

OnExisting = Literal["skip", "replace"]


def export_memories(
    manager: MemoryManager,
    path: Path,
    *,
    user_id: str | None,
    overwrite: bool,
    now_iso: str,
) -> ExportReport:
    storage = manager.storage
    if path.exists() and not overwrite:
        raise ExportError(f"{path} already exists.", hint="Pass overwrite=True (CLI: --force).")
    if path.exists() and path.is_dir():
        raise ExportError(
            f"{path} is a directory.", hint="Choose a file name such as memories.json."
        )
    memories: list[dict[str, Any]] = []
    flt = MemoryFilter(user_id=user_id, statuses=(), include_expired=True)
    offset = 0
    while True:
        page = storage.list_memories(flt, limit=1_000, offset=offset, order="oldest")
        memories.extend(m.to_dict() for m in page)
        if len(page) < 1_000:
            break
        offset += 1_000
    events = [e.to_dict() for e in storage.list_events(user_id)]
    relationships = [r.to_dict() for r in storage.list_relationships(user_id)]
    conflicts = [c.to_dict() for c in storage.list_conflicts(user_id)]
    document = {
        "format": EXPORT_FORMAT,
        "format_version": EXPORT_FORMAT_VERSION,
        "highhxpack_version": __version__,
        "exported_at": now_iso,
        "user_id": user_id,
        "memories": memories,
        "events": events,
        "relationships": relationships,
        "conflicts": conflicts,
    }
    write_private_json(path, document)
    return ExportReport(
        path=path,
        memories=len(memories),
        events=len(events),
        relationships=len(relationships),
        conflicts=len(conflicts),
    )


def write_private_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically write JSON to ``path`` with owner-only permissions."""
    directory = path.parent if str(path.parent) else Path(".")
    if not directory.is_dir():
        raise ExportError(f"Directory {directory} does not exist.")
    fd, tmp_name = tempfile.mkstemp(prefix=".highhxpack-export-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except (OSError, ValueError) as exc:  # ValueError: unencodable text or NaN
        _silent_unlink(tmp_name)
        raise ExportError(f"Could not write {path}.", reason=str(exc)) from exc
    except BaseException:
        _silent_unlink(tmp_name)
        raise


def _silent_unlink(name: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(name)


def read_import_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MemoryImportError(f"Import file not found: {path}")
    if not path.is_file():
        raise MemoryImportError(f"{path} is not a regular file.")
    size = path.stat().st_size
    if size > MAX_IMPORT_BYTES:
        raise MemoryImportError(
            f"{path} is too large to import ({size} bytes).",
            hint=f"The limit is {MAX_IMPORT_BYTES} bytes; split the export by user.",
        )

    def reject_constant(value: str) -> Any:
        raise ValueError(f"invalid JSON constant {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=reject_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise MemoryImportError(
            f"{path} is not valid JSON.", reason=str(exc), hint="Use a file created by export."
        ) from exc
    except OSError as exc:
        raise MemoryImportError(f"Cannot read {path}.", reason=str(exc)) from exc
    if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT:
        raise MemoryImportError(
            f"{path} is not a HighHXPack export.",
            hint="Only files produced by `highhxpack export` can be imported.",
        )
    version = data.get("format_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not 1 <= version <= EXPORT_FORMAT_VERSION
    ):
        raise MemoryImportError(
            f"Unsupported export format version: {version!r}.",
            hint="Upgrade HighHXPack to import files from newer releases.",
        )
    for key in ("memories", "events", "relationships", "conflicts"):
        if not isinstance(data.get(key, []), list):
            raise MemoryImportError(f"'{key}' must be a list in {path}.")
    return data


@dataclass
class _Tally:
    imported: int = 0
    replaced: int = 0
    skipped: int = 0
    relationships: int = 0
    events: int = 0
    errors: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        if len(self.errors) < MAX_REPORTED_ERRORS:
            self.errors.append(message)
        elif len(self.errors) == MAX_REPORTED_ERRORS:
            self.errors.append("... further errors omitted")


def _field(
    item: dict[str, Any], name: str, parse: Callable[[Any], Any], default: Any = None
) -> Any:
    value = item.get(name, default)
    return parse(value) if value is not None else None


def parse_memory(item: object, user_override: str | None) -> MemoryRecord:
    """Validate one exported memory dictionary."""
    if not isinstance(item, dict):
        raise MemoryImportError("memory entry is not an object")
    status = item.get("status", MemoryStatus.ACTIVE.value)
    if status not in {s.value for s in MemoryStatus}:
        raise MemoryImportError(f"invalid status {status!r}")
    counts = {}
    for name, minimum, default in (
        ("version", 1, 1),
        ("mention_count", 1, 1),
        ("access_count", 0, 0),
    ):
        value = item.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise MemoryImportError(f"invalid {name} {value!r}")
        counts[name] = value
    superseded_by = item.get("superseded_by")
    created_at = parse_datetime(_require_str(item, "created_at"))
    return MemoryRecord(
        id=validate_memory_id(item.get("id")),
        user_id=validate_user_id(user_override or item.get("user_id")),
        content=validate_content(item.get("content")),
        memory_type=validate_memory_type(item.get("memory_type", "fact")),
        source=validate_source(item.get("source", "import")),
        status=status,
        importance=validate_unit_interval(item.get("importance", 0.5), field="importance"),
        confidence=validate_unit_interval(item.get("confidence", 1.0), field="confidence"),
        version=counts["version"],
        mention_count=counts["mention_count"],
        access_count=counts["access_count"],
        created_at=created_at,
        updated_at=_field(item, "updated_at", _parse_ts) or created_at,
        last_accessed_at=_field(item, "last_accessed_at", _parse_ts),
        expires_at=_field(item, "expires_at", _parse_ts),
        superseded_by=validate_memory_id(superseded_by) if superseded_by is not None else None,
        metadata=validate_metadata(item.get("metadata")),
    )


def _require_str(item: dict[str, Any], name: str) -> str:
    value = item.get(name)
    if not isinstance(value, str):
        raise MemoryImportError(f"missing or invalid {name}")
    return value


def _parse_ts(value: Any) -> Any:
    if not isinstance(value, str):
        raise MemoryImportError(f"invalid timestamp {value!r}")
    return parse_datetime(value)


def import_memories(
    manager: MemoryManager,
    path: Path,
    *,
    user_id: str | None,
    on_existing: OnExisting,
) -> ImportReport:
    data = read_import_file(path)
    storage = manager.storage
    tally = _Tally()

    records: list[MemoryRecord] = []
    for position, item in enumerate(data.get("memories", [])):
        try:
            records.append(parse_memory(item, user_id))
        except HighHXPackError as exc:
            tally.error(f"memories[{position}]: {exc.message}")
            tally.skipped += 1

    existing = storage.get_memories([r.id for r in records])
    to_write: list[MemoryRecord] = []
    for record in records:
        current = existing.get(record.id)
        if current is None or (on_existing == "replace" and current.user_id == record.user_id):
            to_write.append(record)
        else:
            if current.user_id != record.user_id:
                tally.error(f"memory {record.id}: id already used by another user")
            tally.skipped += 1

    vectors: list[list[float] | None] = []
    for start in range(0, len(to_write), _EMBED_BATCH):
        batch, _warnings = manager.embed_many(
            [r.content for r in to_write[start : start + _EMBED_BATCH]]
        )
        vectors.extend(batch)

    written: set[str] = set()
    now = manager.clock()
    with storage.transaction():
        for record, vector in zip(to_write, vectors, strict=True):
            index, _ = manager.index_for(record.content, vector)
            if record.id in existing:
                storage.delete_memory(record.id)
                tally.replaced += 1
            else:
                tally.imported += 1
            # superseded_by may reference a memory that is not part of this import.
            storage.insert_memory(record.with_changes(superseded_by=None), index)
            written.add(record.id)
        for record in to_write:
            if record.superseded_by and (
                record.superseded_by in written or storage.get_memory(record.superseded_by)
            ):
                storage.update_memory(record)

        _import_related(
            storage,
            data,
            user_id=user_id,
            written=written,
            to_write=to_write,
            tally=tally,
            now=now,
            filename=path.name,
        )

    return ImportReport(
        imported=tally.imported,
        replaced=tally.replaced,
        skipped=tally.skipped,
        relationships=tally.relationships,
        events=tally.events,
        errors=tuple(tally.errors),
    )


def _import_related(
    storage: StorageBackend,
    data: dict[str, Any],
    *,
    user_id: str | None,
    written: set[str],
    to_write: list[MemoryRecord],
    tally: _Tally,
    now: datetime,
    filename: str,
) -> None:
    """Import history, relationships and conflicts that belong to written memories."""
    # Only events that pass validation count as history; a memory whose history
    # was entirely invalid still gets an "imported" event below.
    has_history: set[str] = set()
    for position, item in enumerate(data.get("events", [])):
        event = _parse_event(item, user_id, written, tally, position)
        if event is not None:
            storage.add_event(event)
            has_history.add(event.memory_id)
            tally.events += 1
    for record in to_write:
        if record.id not in has_history:
            storage.add_event(
                MemoryEvent(
                    memory_id=record.id,
                    user_id=record.user_id,
                    kind=EventKind.IMPORTED.value,
                    version=record.version,
                    content=record.content,
                    created_at=now,
                    reason=f"imported from {filename}",
                )
            )

    def owned_by(memory_id: str, owner: str) -> bool:
        record = storage.get_memory(memory_id)
        return record is not None and record.user_id == owner

    for position, item in enumerate(data.get("relationships", [])):
        rel = _parse_relationship(item, user_id, tally, position)
        if rel is None:
            continue
        if rel.memory_id is not None and not owned_by(rel.memory_id, rel.user_id):
            continue  # never link an edge to another user's memory
        # Edge ids are not referenced anywhere else, so a fresh id avoids clashes with
        # edges already in this database (e.g. re-importing under another user).
        storage.add_relationship(dataclasses.replace(rel, id=new_id()))
        tally.relationships += 1

    for position, item in enumerate(data.get("conflicts", [])):
        conflict = _parse_conflict(item, user_id, tally, position)
        if (
            conflict is not None
            and storage.get_conflict(conflict.id) is None
            and owned_by(conflict.memory_id, conflict.user_id)
            and owned_by(conflict.other_id, conflict.user_id)
        ):
            storage.add_conflict(conflict)


def _parse_event(
    item: object, user_override: str | None, written: set[str], tally: _Tally, position: int
) -> MemoryEvent | None:
    try:
        if not isinstance(item, dict):
            raise MemoryImportError("event entry is not an object")
        memory_id = validate_memory_id(item.get("memory_id"))
        if memory_id not in written:
            return None
        kind = item.get("kind")
        if kind not in {k.value for k in EventKind}:
            raise MemoryImportError(f"invalid event kind {kind!r}")
        version = item.get("version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise MemoryImportError("invalid event version")
        related = item.get("related_id")
        reason = item.get("reason")
        return MemoryEvent(
            memory_id=memory_id,
            user_id=validate_user_id(user_override or item.get("user_id")),
            kind=kind,
            version=version,
            content=validate_content(item.get("content")),
            created_at=parse_datetime(_require_str(item, "created_at")),
            related_id=validate_memory_id(related) if related is not None else None,
            reason=str(reason)[:1000] if reason is not None else None,
        )
    except HighHXPackError as exc:
        tally.error(f"events[{position}]: {exc.message}")
        return None


def _parse_relationship(
    item: object, user_override: str | None, tally: _Tally, position: int
) -> Relationship | None:
    try:
        if not isinstance(item, dict):
            raise MemoryImportError("relationship entry is not an object")
        memory_id = item.get("memory_id")
        return Relationship(
            id=validate_memory_id(item.get("id")),
            user_id=validate_user_id(user_override or item.get("user_id")),
            source=normalize_node(item.get("source")),
            relation=normalize_relation(item.get("relation")),
            target=normalize_node(item.get("target")),
            confidence=validate_unit_interval(item.get("confidence", 1.0), field="confidence"),
            created_at=parse_datetime(_require_str(item, "created_at")),
            memory_id=validate_memory_id(memory_id) if memory_id is not None else None,
            metadata=validate_metadata(item.get("metadata")),
        )
    except HighHXPackError as exc:
        tally.error(f"relationships[{position}]: {exc.message}")
        return None


def _parse_conflict(
    item: object, user_override: str | None, tally: _Tally, position: int
) -> Conflict | None:
    try:
        if not isinstance(item, dict):
            raise MemoryImportError("conflict entry is not an object")
        status = item.get("status")
        if status not in {s.value for s in ConflictStatus}:
            raise MemoryImportError(f"invalid conflict status {status!r}")
        kind = item.get("kind")
        if not isinstance(kind, str) or not kind.isidentifier():
            raise MemoryImportError("invalid conflict kind")
        resolution = item.get("resolution")
        resolved_at = item.get("resolved_at")
        return Conflict(
            id=validate_memory_id(item.get("id")),
            user_id=validate_user_id(user_override or item.get("user_id")),
            memory_id=validate_memory_id(item.get("memory_id")),
            other_id=validate_memory_id(item.get("other_id")),
            kind=kind,
            status=status,
            created_at=parse_datetime(_require_str(item, "created_at")),
            resolution=str(resolution)[:1000] if resolution is not None else None,
            resolved_at=_parse_ts(resolved_at) if resolved_at is not None else None,
        )
    except HighHXPackError as exc:
        tally.error(f"conflicts[{position}]: {exc.message}")
        return None
