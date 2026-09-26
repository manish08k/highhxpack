from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from highhxpack import ExportError, Memory, MemoryImportError, ValidationError
from highhxpack.core import transfer
from support import FakeClock


@pytest.fixture
def populated(memory: Memory) -> Memory:
    memory.ingest("alice", "I prefer Python for AI. My sister Anna lives in Berlin.")
    memory.remember("alice", "I live in Paris")
    memory.remember("alice", "I live in Rome")  # supersedes Paris -> conflict + history
    memory.remember("bob", "I use Rust", metadata={"team": "core"})
    return memory


def test_roundtrip_preserves_everything(
    populated: Memory, tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "export.json"
    report = populated.export(path)
    assert (report.memories, report.relationships, report.conflicts) == (5, 2, 1)
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    data = json.loads(path.read_text())
    assert data["format"] == "highhxpack-export"
    assert data["format_version"] == 1

    with Memory(tmp_path / "restored.db", clock=clock) as restored:
        result = restored.import_(path)
        assert (result.imported, result.skipped, result.errors) == (5, 0, ())
        assert result.relationships == 2
        for original in populated.list(status="any", limit=100):
            copy = restored.get(original.id)
            assert copy.to_dict() | {"embedding_model": None} == original.to_dict() | {
                "embedding_model": None
            }
        assert [
            e.kind for e in restored.history(populated.list("alice", status="superseded")[0].id)
        ] == [
            "created",
            "superseded",
        ]
        assert len(restored.conflicts("alice", status=None)) == 1
        assert restored.stats().missing_vectors == 0
        assert restored.recall("alice", "python")

        again = restored.import_(path)
        assert (again.imported, again.skipped) == (0, 5)
        replaced = restored.import_(path, on_existing="replace")
        assert replaced.replaced == 5


def test_export_single_user_and_import_as_other_user(
    populated: Memory, tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "bob.json"
    assert populated.export(path, user_id="bob").memories == 1
    with Memory(tmp_path / "other.db", clock=clock) as other:
        other.import_(path, user_id="carol")
        assert other.users() == {"carol": 1}
    # the same ids now belong to another user in `populated` -> they are skipped
    report = populated.import_(path, user_id="carol")
    assert report.skipped == 1
    assert "another user" in report.errors[0]


def test_export_refuses_overwrite(populated: Memory, tmp_path: Path) -> None:
    path = tmp_path / "e.json"
    populated.export(path)
    with pytest.raises(ExportError, match="exists"):
        populated.export(path)
    populated.export(path, overwrite=True)
    with pytest.raises(ExportError, match="directory"):
        populated.export(tmp_path, overwrite=True)
    with pytest.raises(ExportError, match="does not exist"):
        populated.export(tmp_path / "missing" / "e.json")


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def base_doc(memories: list[Any]) -> dict[str, Any]:
    return {"format": "highhxpack-export", "format_version": 1, "memories": memories}


GOOD = {
    "id": "a" * 32,
    "user_id": "u",
    "content": "I use Python",
    "created_at": "2024-01-01T00:00:00Z",
}


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("not json at all", "not valid JSON"),
        ('{"format": "other"}', "not a HighHXPack export"),
        ('{"format": "highhxpack-export", "format_version": 99}', "Unsupported"),
        ('{"format": "highhxpack-export", "format_version": 1, "memories": {}}', "must be a list"),
        (
            '{"format": "highhxpack-export", "format_version": 1, "memories": [NaN]}',
            "not valid JSON",
        ),
        ("[1, 2]", "not a HighHXPack export"),
    ],
)
def test_malformed_files(memory: Memory, tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(MemoryImportError, match=match):
        memory.import_(path)


def test_invalid_records_are_reported_not_imported(memory: Memory, tmp_path: Path) -> None:
    records = [
        GOOD,
        {**GOOD, "id": "not-an-id"},
        {**GOOD, "id": "b" * 32, "content": ""},
        {**GOOD, "id": "c" * 32, "importance": 7},
        {**GOOD, "id": "d" * 32, "status": "deleted"},
        {**GOOD, "id": "e" * 32, "created_at": 5},
        {**GOOD, "id": "f" * 32, "version": 0},
        {**GOOD, "id": "1" * 32, "metadata": "nope"},
        {**GOOD, "id": "2" * 32, "memory_type": "__import__('os')"},
        "string entry",
    ]
    doc = base_doc(records)
    doc["events"] = [{"memory_id": "a" * 32, "kind": "exploded"}, 7]
    doc["relationships"] = [{"id": "x"}, "bad"]
    doc["conflicts"] = [{"id": "3" * 32, "status": "weird"}, None]
    report = memory.import_(write_json(tmp_path / "mixed.json", doc))
    assert report.imported == 1
    assert report.skipped == 9
    assert len(report.errors) == 9 + 2 + 2 + 2
    assert memory.get("a" * 32).content == "I use Python"
    # an IMPORTED history event is synthesized when no valid history exists
    assert [e.kind for e in memory.history("a" * 32)] == ["imported"]


def test_imported_content_is_sanitized(memory: Memory, tmp_path: Path) -> None:
    doc = base_doc([{**GOOD, "content": "evil \x1b[2J text"}])
    memory.import_(write_json(tmp_path / "x.json", doc))
    assert memory.get("a" * 32).content == "evil [2J text"


def test_file_checks(memory: Memory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(MemoryImportError, match="not found"):
        memory.import_(tmp_path / "missing.json")
    with pytest.raises(MemoryImportError, match="regular file"):
        memory.import_(tmp_path)
    path = write_json(tmp_path / "big.json", base_doc([GOOD]))
    monkeypatch.setattr(transfer, "MAX_IMPORT_BYTES", 10)
    with pytest.raises(MemoryImportError, match="too large"):
        memory.import_(path)
    with pytest.raises(ValidationError):
        memory.import_(path, on_existing="merge")  # type: ignore[arg-type]


def test_error_list_is_capped(memory: Memory, tmp_path: Path) -> None:
    doc = base_doc(["bad"] * 250)
    report = memory.import_(write_json(tmp_path / "many.json", doc))
    assert len(report.errors) == transfer.MAX_REPORTED_ERRORS + 1
    assert report.errors[-1].startswith("...")


def test_superseded_by_outside_import_is_dropped(memory: Memory, tmp_path: Path) -> None:
    doc = base_doc([{**GOOD, "status": "superseded", "superseded_by": "9" * 32}])
    memory.import_(write_json(tmp_path / "s.json", doc))
    assert memory.get("a" * 32).superseded_by is None


def test_reimport_under_another_user_into_same_database(memory: Memory, tmp_path: Path) -> None:
    # Regression: edge id collisions used to crash with TypeError.
    memory.ingest("alice", "I live in Berlin")
    memory.remember("alice", "I prefer Java")
    memory.remember("alice", "I prefer Python")  # creates a conflict record
    path = tmp_path / "a.json"
    memory.export(path)
    report = memory.import_(path, user_id="copy")
    assert report.imported == 0  # ids are owned by alice, so memories are skipped
    assert report.skipped == 3
    # ...and nothing may be linked from "copy" to alice's memories.
    assert memory.graph.edges("copy") == []
    assert memory.conflicts("copy", status=None) == []
    assert len(memory.graph.edges("alice")) == 1


@pytest.mark.parametrize("version", [0, -1, 2, "1", True])
def test_format_version_must_be_supported(memory: Memory, tmp_path: Path, version: object) -> None:
    doc = {"format": "highhxpack-export", "format_version": version, "memories": []}
    with pytest.raises(MemoryImportError, match="Unsupported"):
        memory.import_(write_json(tmp_path / "v.json", doc))
