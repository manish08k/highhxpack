"""Guards against accidental changes to the public API and against crashes on odd input."""

from __future__ import annotations

import contextlib
import inspect
import random

import pytest

import highhxpack
from highhxpack import Memory, ValidationError

#: The public API of 0.1.0.  Removing or renaming a name here is a breaking change:
#: update CHANGELOG.md and follow semantic versioning before editing this set.
PUBLIC_API = {
    "CallableEmbedding", "CallableProvider", "Config", "ConfigurationError", "Conflict",
    "ConflictNotFoundError", "ConflictPolicy", "ConflictStatus", "ConsolidationReport",
    "EmbeddingError", "EmbeddingProvider", "EventKind", "ExportError", "ExportReport",
    "ExtractionError", "HashingEmbedding", "HighHXPackError", "ImportReport", "LLMProvider",
    "Memory", "MemoryEvent", "MemoryImportError", "MemoryNotFoundError", "MemoryRecord",
    "MemoryStats", "MemoryStatus", "MemoryType", "MigrationError", "ProviderError",
    "ProviderNotAvailableError", "RecallResult", "Relationship", "RememberAction",
    "RememberResult", "ScoreBreakdown", "ScoringConfig", "StorageBackend", "StorageError",
    "Strategy", "UserSummary", "ValidationError", "__version__",
}  # fmt: skip

MEMORY_METHODS = {
    "remember", "ingest", "recall", "search", "get", "update", "forget", "restore", "delete",
    "list", "count", "clear", "stats", "inspect", "users", "history", "conflicts",
    "resolve_conflict", "consolidate", "reindex", "export", "import_", "resolve_id", "close",
}  # fmt: skip


def test_public_names_are_stable() -> None:
    assert set(highhxpack.__all__) == PUBLIC_API


def test_memory_methods_are_stable() -> None:
    public = {n for n, v in inspect.getmembers(Memory) if callable(v) and not n.startswith("_")}
    assert public >= MEMORY_METHODS


def test_spec_call_style_is_supported() -> None:
    params = inspect.signature(Memory.remember).parameters
    assert list(params)[1:3] == ["user_id", "content"]
    params = inspect.signature(Memory.recall).parameters
    assert list(params)[1:3] == ["user_id", "query"]


def _random_text(rng: random.Random) -> str:
    exotic = "".join(map(chr, (0x00, 0x1B, 0x200B, 0x2019, 0xE9, 0x4E2D, 0x1F642, 0xD7FF)))
    punctuation = "abc XYZ 123 .,;:!?'\"-_+#/ " + chr(92) + chr(10) + chr(9)
    alphabet = punctuation + exotic + "I prefer like hate live in my name is"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))


def test_random_input_never_crashes() -> None:
    rng = random.Random(1234)
    with Memory(":memory:") as memory:
        for _ in range(400):
            text = _random_text(rng)
            for call in (
                lambda t=text: memory.remember("u", t),
                lambda t=text: memory.recall("u", t),
                lambda t=text: memory.search(t),
                lambda t=text: memory.ingest("u", t),
            ):
                with contextlib.suppress(ValidationError):  # the only acceptable failure
                    call()
        assert memory.count("u") >= 0
        memory.consolidate("u", summarize=True)


@pytest.mark.parametrize("user_id", ["", "   ", "a\nb", "x" * 300])
def test_invalid_user_ids_are_validation_errors(user_id: str) -> None:
    with Memory(":memory:") as memory, pytest.raises(ValidationError):
        memory.remember(user_id, "I use Python")


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.remember("u", "bad " + chr(0xD800) + " text"),
        lambda m: m.remember("u" + chr(0xDC80), "some text"),
        lambda m: m.recall("u", "query " + chr(0xD800)),
        lambda m: m.remember("u", "some text", metadata={"k": chr(0xD800)}),
        lambda m: m.graph.add("u", "user", "uses", chr(0xD800)),
    ],
)
def test_lone_surrogates_are_validation_errors(call: object) -> None:
    # Regression: these used to escape as UnicodeEncodeError (or break export later).
    with Memory(":memory:") as memory, pytest.raises(ValidationError):
        call(memory)  # type: ignore[operator]
