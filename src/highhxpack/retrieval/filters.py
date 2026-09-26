"""Backend-agnostic memory filters.

A :class:`MemoryFilter` describes *which* memories an operation may see.
Storage backends translate it into their native query language; the
:meth:`MemoryFilter.matches` method is the reference semantics and is used by
tests to verify backends.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from highhxpack.exceptions import ValidationError
from highhxpack.models.memory import MemoryRecord, MemoryStatus
from highhxpack.utils.timestamps import parse_datetime
from highhxpack.utils.validation import (
    validate_memory_type,
    validate_source,
    validate_unit_interval,
    validate_user_id,
)

_METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_VALID_STATUSES = frozenset(s.value for s in MemoryStatus)

JSONScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class MemoryFilter:
    """Criteria a memory must satisfy.

    Empty tuples mean "no restriction".  ``statuses`` defaults to active
    memories only.  Expired memories are excluded unless ``include_expired``;
    ``now`` is the reference time for expiry and must be set by the caller when
    ``include_expired`` is false.
    """

    user_id: str | None = None
    memory_types: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    statuses: tuple[str, ...] = (MemoryStatus.ACTIVE.value,)
    include_expired: bool = False
    now: datetime | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    min_importance: float | None = None
    min_confidence: float | None = None
    metadata: Mapping[str, JSONScalar] = field(default_factory=dict)

    def matches(self, memory: MemoryRecord) -> bool:
        if self.user_id is not None and memory.user_id != self.user_id:
            return False
        if self.memory_types and memory.memory_type not in self.memory_types:
            return False
        if self.sources and memory.source not in self.sources:
            return False
        if self.statuses and memory.status not in self.statuses:
            return False
        if not self.include_expired and self.now is not None and memory.is_expired(self.now):
            return False
        if self.created_after is not None and memory.created_at < self.created_after:
            return False
        if self.created_before is not None and memory.created_at >= self.created_before:
            return False
        if self.min_importance is not None and memory.importance < self.min_importance:
            return False
        if self.min_confidence is not None and memory.confidence < self.min_confidence:
            return False
        return all(memory.metadata.get(k, _MISSING) == v for k, v in self.metadata.items())


_MISSING: Any = object()


def _as_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or hasattr(value, "value"):
        return (str(getattr(value, "value", value)),)
    return tuple(str(getattr(v, "value", v)) for v in value)


def build_filter(
    *,
    user_id: str | None = None,
    memory_type: str | Iterable[str] | None = None,
    source: str | Iterable[str] | None = None,
    status: str | Iterable[str] | None = MemoryStatus.ACTIVE.value,
    include_expired: bool = False,
    now: datetime | None = None,
    created_after: str | datetime | None = None,
    created_before: str | datetime | None = None,
    min_importance: float | None = None,
    min_confidence: float | None = None,
    metadata: Mapping[str, JSONScalar] | None = None,
) -> MemoryFilter:
    """Validate user-facing filter arguments and build a :class:`MemoryFilter`.

    ``status="any"`` (or ``None``) disables status filtering.
    """
    types = tuple(validate_memory_type(t) for t in _as_tuple(memory_type))
    sources = tuple(validate_source(s) for s in _as_tuple(source))
    statuses: tuple[str, ...] = ()
    if status is not None and status != "any":
        statuses = _as_tuple(status)
        unknown = [s for s in statuses if s not in _VALID_STATUSES]
        if unknown:
            raise ValidationError(
                f"Unknown memory status: {unknown[0]!r}",
                hint=f"Use one of: {', '.join(sorted(_VALID_STATUSES))}, or 'any'.",
            )
    meta: dict[str, JSONScalar] = {}
    for key, value in (metadata or {}).items():
        if not isinstance(key, str) or not _METADATA_KEY_RE.match(key):
            raise ValidationError(
                f"Invalid metadata filter key: {key!r}",
                hint="Metadata filter keys may contain letters, digits, '_' and '-'.",
            )
        if value is not None and not isinstance(value, str | int | float | bool):
            raise ValidationError("Metadata filter values must be str, int, float, bool or None.")
        meta[key] = value
    return MemoryFilter(
        user_id=validate_user_id(user_id) if user_id is not None else None,
        memory_types=types,
        sources=sources,
        statuses=statuses,
        include_expired=include_expired,
        now=now,
        created_after=parse_datetime(created_after) if created_after is not None else None,
        created_before=parse_datetime(created_before) if created_before is not None else None,
        min_importance=(
            validate_unit_interval(min_importance, field="min_importance")
            if min_importance is not None
            else None
        ),
        min_confidence=(
            validate_unit_interval(min_confidence, field="min_confidence")
            if min_confidence is not None
            else None
        ),
        metadata=meta,
    )
