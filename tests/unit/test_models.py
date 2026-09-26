from __future__ import annotations

import builtins
import json
from datetime import UTC, datetime, timedelta

import pytest

import highhxpack
from highhxpack import exceptions
from highhxpack.models import (
    Conflict,
    ConsolidationReport,
    MemoryEvent,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    Relationship,
    UserSummary,
)

NOW = datetime(2025, 1, 1, tzinfo=UTC)


def make_record(**changes: object) -> MemoryRecord:
    base = MemoryRecord(
        id="a" * 32,
        user_id="u",
        content="I use Python",
        memory_type=MemoryType.FACT.value,
        source="api",
        importance=0.5,
        confidence=1.0,
        created_at=NOW,
        updated_at=NOW,
        metadata={"k": [1, 2]},
    )
    return base.with_changes(**changes)


class TestMemoryRecord:
    def test_roundtrip_through_json(self) -> None:
        record = make_record(expires_at=NOW + timedelta(days=1), last_accessed_at=NOW)
        data = json.loads(json.dumps(record.to_dict()))
        assert MemoryRecord.from_dict(data) == record

    def test_properties(self) -> None:
        record = make_record(expires_at=NOW)
        assert record.is_active
        assert record.short_id == "a" * 8
        assert record.is_expired(NOW)
        assert not make_record().is_expired(NOW)
        assert not make_record(status=MemoryStatus.ARCHIVED.value).is_active

    def test_immutable(self) -> None:
        record = make_record()
        with pytest.raises(AttributeError):
            record.content = "x"  # type: ignore[misc]


def test_other_models_serialize() -> None:
    event = MemoryEvent("a" * 32, "u", "created", 1, "text", NOW)
    rel = Relationship("b" * 32, "u", "user", "uses", "Python", 1.0, NOW)
    conflict = Conflict("c" * 32, "u", "a" * 32, "d" * 32, "changed_value", "open", NOW)
    summary = UserSummary("u", 1, 1, first_memory_at=NOW)
    for obj in (event, rel, conflict, summary):
        json.dumps(obj.to_dict())
    assert ConsolidationReport("u", dry_run=True).changed is False
    assert ConsolidationReport("u", dry_run=False, conflicts_detected=1).changed is True


class TestExceptions:
    def test_message_reason_hint(self) -> None:
        err = exceptions.StorageError("broken", reason="disk", hint="free space")
        assert str(err) == "broken\nReason: disk\nHint: free space"
        assert isinstance(err, highhxpack.HighHXPackError)

    def test_validation_error_is_value_error(self) -> None:
        assert issubclass(exceptions.ValidationError, ValueError)
        assert issubclass(exceptions.MemoryNotFoundError, LookupError)

    def test_import_error_does_not_shadow_builtin(self) -> None:
        assert not issubclass(exceptions.MemoryImportError, builtins.ImportError)
        assert "ImportError" not in exceptions.__all__

    def test_not_found_carries_id(self) -> None:
        err = exceptions.MemoryNotFoundError("abc")
        assert err.memory_id == "abc"
        assert "abc" in str(err)


def test_public_api_is_importable() -> None:
    for name in highhxpack.__all__:
        assert hasattr(highhxpack, name), name
    assert highhxpack.__version__.count(".") == 2
