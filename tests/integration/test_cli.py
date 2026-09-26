from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from highhxpack import __version__
from highhxpack.cli.display import Console, format_bytes, truncate
from highhxpack.cli.main import main

pytestmark = pytest.mark.integration


class CLI:
    def __init__(self, capsys: pytest.CaptureFixture[str], db: Path):
        self.capsys = capsys
        self.db = db

    def __call__(self, *args: str) -> tuple[int, str, str]:
        code = main([*args, "--db", str(self.db), "--no-color"])
        out, err = self.capsys.readouterr()
        return code, out, err

    def json(self, *args: str) -> object:
        code, out, err = self(*args, "--json")
        assert code == 0, err
        return json.loads(out)


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> CLI:
    return CLI(capsys, tmp_path / "cli.db")


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    assert "remember" in capsys.readouterr().out
    assert main([]) == 2
    with pytest.raises(SystemExit) as info:
        main(["no-such-command"])
    assert info.value.code == 2


def test_global_options_before_or_after_command(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = str(tmp_path / "g.db")
    assert main(["--db", db, "--user", "ann", "remember", "I use Python"]) == 0
    assert main(["recall", "python", "--db", db, "-u", "ann"]) == 0
    capsys.readouterr()
    assert main(["--db", db, "--json", "memories", "--user", "ann"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["user_id"] == "ann"


def test_phase_one_flow(cli: CLI, isolated_env: Path) -> None:
    code, out, _ = cli("init")
    assert code == 0
    assert "ready" in out
    assert (isolated_env / "config.toml").exists()
    code, out, _ = cli("init")
    assert "existing, kept" in out

    code, out, _ = cli("remember", "I use Python")
    assert code == 0
    assert "Remembered" in out
    code, out, _ = cli("remember", "i use python!")
    assert "reinforced" in out
    code, out, _ = cli("memories")
    assert code == 0
    assert "I use Python" in out
    assert out.count("I use Python") == 1


def test_recall_search_json(cli: CLI) -> None:
    cli("remember", "I prefer Python for AI development")
    cli("remember", "My favorite editor is Neovim", "--user", "bob")
    results = cli.json("recall", "python")
    assert isinstance(results, list)
    assert results[0]["memory"]["content"] == "I prefer Python for AI development"
    assert set(results[0]["breakdown"]) >= {"keyword", "semantic", "final"}
    assert cli.json("recall", "neovim") == []  # other user's memory
    everyone = cli.json("search", "neovim OR python")
    assert isinstance(everyone, list)
    assert {r["memory"]["user_id"] for r in everyone} == {"default", "bob"}
    code, out, _ = cli("recall", "python", "--explain")
    assert "relevance=" in out
    code, out, _ = cli("recall", "zzzz")
    assert code == 0
    assert "No matching" in out


def test_remember_options_and_stdin(cli: CLI, monkeypatch: pytest.MonkeyPatch) -> None:
    data = cli.json(
        "remember",
        "Deploys",
        "on",
        "Friday",
        "--type",
        "project_note",
        "--importance",
        "0.9",
        "--meta",
        "team=ops",
        "--meta",
        "count=3",
        "--ttl",
        "30d",
    )
    assert isinstance(data, dict)
    memory = data["memory"]
    assert memory["content"] == "Deploys on Friday"
    assert memory["metadata"] == {"team": "ops", "count": 3}
    assert memory["expires_at"] is not None
    monkeypatch.setattr("sys.stdin", io.StringIO("I live in Oslo\n"))
    code, out, _ = cli("remember", "-")
    assert code == 0
    assert "I live in Oslo" in out
    code, _, err = cli("remember", "x", "--meta", "novalue")
    assert code == 2
    assert "KEY=VALUE" in err


def test_extract_mode(cli: CLI) -> None:
    code, out, _ = cli(
        "remember",
        "--extract",
        "I've been using Python for ML, but I prefer C++ for competitive programming.",
    )
    assert code == 0
    assert out.count("Remembered") == 2
    code, out, _ = cli("remember", "--extract", "Hello there!")
    assert "Nothing worth remembering" in out
    code, out, _ = cli("inspect")
    assert "user ─prefers→ C++" in out


def test_forget_restore_history(cli: CLI) -> None:
    created = cli.json("remember", "I use Vim")
    assert isinstance(created, dict)
    mid = created["memory"]["id"]
    code, out, _ = cli("forget", mid[:8])
    assert code == 0
    assert "archived" in out
    assert cli.json("memories") == []
    code, out, _ = cli("restore", mid[:8])
    assert "Restored" in out
    code, out, _ = cli("history", mid[:8])
    assert "archived" in out
    assert "restored" in out
    code, out, _ = cli("forget", mid, "--permanent", "--yes")
    assert "Erased" in out
    code, _, err = cli("history", mid)
    assert code == 3
    assert "not found" in err


def test_errors_are_human_readable(cli: CLI) -> None:
    code, out, err = cli("recall", "")
    assert code == 2
    assert err.startswith("error: query must not be empty")
    assert "Traceback" not in out + err
    code, _, err = cli("forget", "zz")
    assert code == 2
    code, _, err = cli("forget", "abcdef")
    assert code == 3
    assert "hint:" in err
    code, _, err = cli("remember", "x", "--importance", "5")
    assert code == 2
    code, _, err = cli("import", "/definitely/missing.json")
    assert code == 1
    assert "not found" in err


def test_bad_config_is_reported(cli: CLI, tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("openai_api_key = 'sk-secret'\n")
    code, out, err = cli("stats", "--config", str(bad))
    assert code == 1
    assert "not allowed" in err
    assert "sk-secret" not in out + err


def test_stats_inspect_config(cli: CLI) -> None:
    cli("remember", "I use Python")
    stats = cli.json("stats")
    assert isinstance(stats, dict)
    assert stats["total"] == 1
    assert "hashing-v1-384" in cli("stats")[1]
    inspect = cli.json("inspect", "default")
    assert isinstance(inspect, dict)
    assert inspect["active"] == 1
    config = cli.json("config")
    assert isinstance(config, dict)
    assert config["openai_api_key"] is None
    assert "base_weight" in cli("config")[1]


def test_conflicts_and_consolidate(cli: CLI) -> None:
    cli("remember", "I live in Paris")
    code, _, err = cli("remember", "I live in Rome", "--confidence", "0.5")
    assert "conflicts with" in err
    conflicts = cli.json("conflicts")
    assert isinstance(conflicts, list)
    assert len(conflicts) == 1
    code, out, _ = cli("conflicts")
    assert "I live in Rome" in out
    rome = conflicts[0]["memory_id"]
    code, out, _ = cli("resolve", conflicts[0]["id"], "--keep", rome[:8])
    assert code == 0
    assert "Resolved" in out
    code, out, _ = cli("conflicts")
    assert "No open conflicts" in out
    cli("remember", "I love tea", "--no-dedupe")
    cli("remember", "I like tea", "--no-dedupe")
    code, out, _ = cli("consolidate", "--dry-run")
    assert "Would merge" in out
    code, out, _ = cli("consolidate")
    assert "1 merged" in out
    code, out, _ = cli("consolidate")
    assert "Nothing to consolidate" in out
    code, _, _ = cli("resolve", "f" * 32, "--keep", rome[:8])
    assert code == 3


def test_export_import(cli: CLI, tmp_path: Path) -> None:
    cli("remember", "I use Python")
    target = tmp_path / "out.json"
    code, out, _ = cli("export", str(target))
    assert code == 0
    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == 0o600
    code, _, err = cli("export", str(target))
    assert code == 1
    assert "--force" in err
    assert cli("export", str(target), "--force")[0] == 0
    other = CLI(cli.capsys, tmp_path / "other.db")
    report = other.json("import", str(target), "--user", "copy")
    assert isinstance(report, dict)
    assert report["imported"] == 1
    code, out, _ = other("memories", "--all-users")
    assert "copy" in out
    code, out, _ = other("import", str(target), "--user", "copy")
    assert "1 skipped" in out


def test_memories_listing_options(cli: CLI) -> None:
    for i in range(3):
        cli("remember", f"note number {i}", "--type", "knowledge")
    out = cli("memories", "--limit", "2")[1]
    assert "Showing 2 of 3" in out
    out = cli("list", "--status", "any", "--order", "oldest", "--all-users")[1]
    assert "STATUS" in out
    assert "USER" in out
    assert "No memories" in cli("memories", "--user", "nobody")[1]


def test_display_helpers() -> None:
    out = io.StringIO()
    console = Console(out=out, err=out, color=True)
    assert console.style("x", "bold") == "\x1b[1mx\x1b[0m"
    console.table(["A", "B"], [["1", "a long cell " * 20]], flex=1)
    assert "…" in out.getvalue()
    assert truncate("a  b", 10) == "a b"
    assert format_bytes(10) == "10 B"
    assert format_bytes(2048) == "2.0 KiB"
    assert Console(out=io.StringIO(), color=None).color is False  # not a TTY
    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n")
    plain = Console(out=legacy, color=False)
    assert plain.sym("arrow") == "->"
    plain.table(["A"], [["x" * 500]], flex=0)
    legacy.flush()
    assert legacy.buffer.getvalue().endswith(b"...\n")  # type: ignore[attr-defined]


def test_permanent_delete_requires_confirmation_when_not_interactive(
    cli: CLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = cli.json("remember", "I use Vim")
    assert isinstance(created, dict)
    mid = created["memory"]["id"]
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # not a TTY
    code, _, err = cli("forget", mid, "--permanent")
    assert code == 2
    assert "--yes" in err
    assert cli.json("memories")  # still there
    assert cli("forget", mid, "--permanent", "--yes")[0] == 0


def test_interactive_permanent_delete_prompts(cli: CLI, monkeypatch: pytest.MonkeyPatch) -> None:
    created = cli.json("remember", "I use Emacs")
    assert isinstance(created, dict)
    mid = created["memory"]["id"]
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    code, out, _ = cli("forget", mid, "--permanent")
    assert code == 1
    assert "Cancelled" in out
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert cli("forget", mid, "--permanent")[0] == 0


def test_init_creates_requested_config_file(cli: CLI, tmp_path: Path) -> None:
    target = tmp_path / "custom" / "settings.toml"
    code, out, err = cli("init", "--config", str(target))
    assert code == 0, err
    assert target.exists()
    assert "existing" not in out


def test_unexpected_errors_are_reported_without_traceback(
    cli: CLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("highhxpack.core.memory.Memory.stats", boom)
    code, _, err = cli("stats")
    assert code == 1
    assert "Unexpected internal error: RuntimeError: kaboom" in err
    assert "Traceback" not in err
    code, _, err = cli("stats", "-v")
    assert "Traceback" in err


def test_invalid_user_option_is_usage_error(cli: CLI) -> None:
    code, _, err = cli("memories", "--user", "")
    assert code == 2
    assert "user_id" in err


def test_undecodable_stdin_is_usage_error(cli: CLI, monkeypatch: pytest.MonkeyPatch) -> None:
    stdin = io.TextIOWrapper(io.BytesIO(b"\xff\xfe bad bytes"), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", stdin)
    code, _, err = cli("remember", "-")
    assert code == 2
    assert "not valid text" in err
