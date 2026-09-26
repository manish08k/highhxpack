"""End-to-end tests: installed entry point, multiple processes, larger datasets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from highhxpack import Memory
from highhxpack.integrations import ChatMemory, format_memories

pytestmark = pytest.mark.integration


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv built by the test
        [sys.executable, "-m", "highhxpack", *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=60,
        check=False,
    )


def test_module_entry_point(tmp_path: Path) -> None:
    db = str(tmp_path / "p.db")
    assert run_cli("--version").stdout.startswith("highhxpack ")
    assert run_cli("remember", "I use Python", "--db", db).returncode == 0
    listed = run_cli("memories", "--db", db, "--json")
    assert json.loads(listed.stdout)[0]["content"] == "I use Python"
    failed = run_cli("recall", "", "--db", db)
    assert failed.returncode == 2
    assert "Traceback" not in failed.stderr


def _console_script() -> str | None:
    """The installed script next to the running interpreter (or on PATH)."""
    for name in ("highhxpack", "highhxpack.exe"):
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
    return shutil.which("highhxpack")


@pytest.mark.skipif(_console_script() is None, reason="console script not installed")
def test_console_script() -> None:
    script = _console_script()
    assert script is not None
    result = subprocess.run(  # noqa: S603
        [script, "--help"], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0
    assert "remember" in result.stdout


def test_concurrent_processes_share_one_database(tmp_path: Path) -> None:
    db = tmp_path / "shared.db"
    code = (
        "import sys\n"
        "from highhxpack import Memory\n"
        "with Memory(sys.argv[1]) as m:\n"
        "    for i in range(40):\n"
        "        m.remember(sys.argv[2], f'process {sys.argv[2]} fact number {i}')\n"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", code, str(db), f"p{n}"])  # noqa: S603
        for n in range(4)
    ]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0, 0]
    with Memory(db) as memory:
        assert memory.users() == {f"p{n}": 40 for n in range(4)}


def test_threads_share_one_memory(tmp_path: Path) -> None:
    errors: list[BaseException] = []
    with Memory(tmp_path / "t.db") as memory:

        def work(n: int) -> None:
            try:
                for i in range(25):
                    memory.remember(f"t{n}", f"thread fact {i} from {n}")
                    memory.recall(f"t{n}", "thread fact")
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert memory.count() == 150


@pytest.mark.slow
def test_large_dataset(tmp_path: Path) -> None:
    topics = ["python", "rust", "flutter", "postgres", "kubernetes", "react", "go", "java"]
    verbs = ["uses", "prefers", "is learning", "deploys with", "writes docs for", "debugs"]
    with Memory(tmp_path / "big.db") as memory:
        start = time.perf_counter()
        for i in range(3_000):
            memory.remember(
                "u",
                f"Project {i} {verbs[i % len(verbs)]} {topics[i % len(topics)]} for team {i % 37}",
                dedupe=False,
                detect_conflicts=False,
            )
        insert_seconds = time.perf_counter() - start
        start = time.perf_counter()
        results = memory.recall("u", "which project prefers rust", limit=10)
        recall_seconds = time.perf_counter() - start
        assert len(results) == 10
        assert all("rust" in r.content for r in results[:5])
        assert memory.count("u") == 3_000
        # Generous bounds: this guards against pathological regressions, not speed.
        assert insert_seconds < 120
        assert recall_seconds < 10


def test_chatbot_scenario(tmp_path: Path) -> None:
    with Memory(tmp_path / "chat.db") as memory:
        chat = ChatMemory(memory, "alice")
        assert chat.context("anything") == ""
        chat.observe("Hi! I'm a data engineer and I prefer Python for pipelines.")
        chat.observe("My manager is Bob. What's the weather like?")
        chat.observe("   ")
        context = chat.context("Which language should I use for my pipeline?")
        assert context.startswith("Relevant memories about the user:")
        assert "I prefer Python for pipelines" in context
        assert chat.context("   ") == ""
        assert isinstance(chat.context("pipeline " * 2_000), str)  # long input is truncated
        # Preferences change: the old one is kept as history, not shown as current.
        chat.observe("Actually, I prefer Rust for pipelines now.")
        context = chat.context("pipeline language")
        assert "Rust" in context
        assert "I prefer Python for pipelines" not in context


def test_format_memories_budget(tmp_path: Path) -> None:
    with Memory(tmp_path / "f.db") as memory:
        for i in range(20):
            memory.remember("u", f"I like topic{i} a lot")
        results = memory.recall("u", "like lot", limit=20)
        text = format_memories(results, max_chars=200)
        assert len(text) <= 200
        assert text.count("\n") >= 1
        assert format_memories(results, max_chars=10) == ""
        assert format_memories([]) == ""


def test_optional_framework_integrations(tmp_path: Path) -> None:
    from highhxpack.exceptions import ProviderNotAvailableError
    from highhxpack.integrations import langchain, llamaindex

    with Memory(tmp_path / "i.db") as memory:
        memory.remember("u", "I prefer Python")
        for module, package in ((langchain, "langchain_core"), (llamaindex, "llama_index.core")):
            try:
                __import__(package)
            except ImportError:
                with pytest.raises(ProviderNotAvailableError, match="pip install"):
                    module.create_retriever(memory, "u")
                continue
            retriever = module.create_retriever(memory, "u", k=3)  # pragma: no cover
            docs = (  # pragma: no cover
                retriever.invoke("python") if module is langchain else retriever.retrieve("python")
            )
            assert docs  # pragma: no cover
