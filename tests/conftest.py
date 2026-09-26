from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from highhxpack import Memory
from support import FakeClock


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the real data directory or pick up the developer's settings."""
    for key in list(os.environ):
        if key.startswith("HIGHHXPACK_") or key in {"OPENAI_API_KEY", "NO_COLOR"}:
            monkeypatch.delenv(key, raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HIGHHXPACK_HOME", str(home))
    return home


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "memory.db"


@pytest.fixture
def memory(db_path: Path, clock: FakeClock) -> Iterator[Memory]:
    with Memory(db_path, clock=clock) as mem:
        yield mem


@pytest.fixture
def mem_memory(clock: FakeClock) -> Iterator[Memory]:
    """An in-memory store for fast tests."""
    with Memory(":memory:", clock=clock) as mem:
        yield mem
