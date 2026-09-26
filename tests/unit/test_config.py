from __future__ import annotations

import os
from pathlib import Path

import pytest

from highhxpack import Config, ConfigurationError, Memory, ScoringConfig
from highhxpack.core.config import CONFIG_TEMPLATE, default_home, write_config_template


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults(isolated_env: Path) -> None:
    config = Config.load()
    assert config.home == isolated_env
    assert config.storage_path == isolated_env / "memory.db"
    assert config.embedding_provider == "hashing"
    assert config.llm_provider == "none"
    assert config.config_file is None


def test_precedence_file_env_explicit(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(
        isolated_env / "config.toml",
        'default_user = "from-file"\nconflict_policy = "keep_both"\nextractor = "rules"\n',
    )
    config = Config.load()
    assert config.default_user == "from-file"
    assert config.conflict_policy == "keep_both"
    assert config.config_file == isolated_env / "config.toml"

    monkeypatch.setenv("HIGHHXPACK_USER", "from-env")
    assert Config.load().default_user == "from-env"
    assert Config.load(default_user="explicit").default_user == "explicit"
    # None means "not given" and must not mask lower-precedence sources.
    assert Config.load(default_user=None).default_user == "from-env"


def test_relative_storage_path_is_relative_to_file(tmp_path: Path) -> None:
    cfg = write(tmp_path / "conf" / "settings.toml", 'storage_path = "db/memory.db"\n')
    config = Config.load(config_file=cfg)
    assert config.storage_path == tmp_path / "conf" / "db" / "memory.db"


def test_scoring_table(tmp_path: Path) -> None:
    cfg = write(tmp_path / "c.toml", "[scoring]\nrecency_half_life_days = 7\n")
    assert Config.load(config_file=cfg).scoring.recency_half_life_days == 7


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('openai_api_key = "sk-123"\n', "not allowed"),
        ("bogus = 1\n", "Unknown option"),
        ("not = [valid\n", "not valid TOML"),
        ('embedding_provider = "magic"\n', "embedding_provider"),
        ('home = "/tmp"\n', "home"),
        ("[scoring]\nnope = 1\n", "Unknown scoring option"),
        ("[scoring]\nbase_weight = -1\n", "non-negative"),
        ('track_access = "yes"\n', "track_access"),
        ("default_user = 5\n", "default_user"),
        ('extractor = "llm"\n', "requires an LLM"),
        ('ollama_url = "file:///etc/passwd"\n', "Invalid Ollama base URL"),
        ("request_timeout = 0\n", "request_timeout"),
        ("dedup_threshold = 1.5\n", "dedup_threshold"),
    ],
)
def test_invalid_config_files(tmp_path: Path, text: str, match: str) -> None:
    cfg = write(tmp_path / "c.toml", text)
    with pytest.raises(ConfigurationError, match=match):
        Config.load(config_file=cfg)


def test_missing_explicit_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        Config.load(config_file=tmp_path / "missing.toml")
    monkeypatch.setenv("HIGHHXPACK_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(ConfigurationError, match="not found"):
        Config.load()


def test_env_coercion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HIGHHXPACK_TRACK_ACCESS", "off")
    monkeypatch.setenv("HIGHHXPACK_REQUEST_TIMEOUT", "5")
    monkeypatch.setenv("HIGHHXPACK_STORAGE_PATH", str(tmp_path / "x.db"))
    config = Config.load()
    assert config.track_access is False
    assert config.request_timeout == 5.0
    assert config.storage_path == tmp_path / "x.db"
    monkeypatch.setenv("HIGHHXPACK_TRACK_ACCESS", "maybe")
    with pytest.raises(ConfigurationError):
        Config.load()
    monkeypatch.setenv("HIGHHXPACK_TRACK_ACCESS", "1")
    monkeypatch.setenv("HIGHHXPACK_DEDUP_THRESHOLD", "high")
    with pytest.raises(ConfigurationError, match="number"):
        Config.load()


def test_unknown_override() -> None:
    with pytest.raises(ConfigurationError, match="Unknown configuration option"):
        Config.load(colour="blue")


def test_secrets_are_redacted() -> None:
    config = Config(openai_api_key="sk-secret-value")
    assert "sk-secret" not in repr(config)
    assert config.to_dict()["openai_api_key"] == "***"
    assert config.to_dict(redact=False)["openai_api_key"] == "sk-secret-value"


def test_template_is_valid_and_private(tmp_path: Path) -> None:
    path = tmp_path / "home" / "config.toml"
    write_config_template(path)
    assert path.read_text() == CONFIG_TEMPLATE
    assert Config.load(config_file=path).default_user == "default"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ConfigurationError, match="exists"):
        write_config_template(path)
    write_config_template(path, overwrite=True)


def test_default_home_env_override(tmp_path: Path) -> None:
    assert default_home({"HIGHHXPACK_HOME": str(tmp_path)}) == tmp_path
    assert default_home({}).name == "highhxpack"


def test_memory_uses_config(tmp_path: Path) -> None:
    config = Config(storage_path=tmp_path / "c.db", embedding_provider="none")
    with Memory(config=config) as memory:
        assert memory.embedder is None
        assert memory.location == str(tmp_path / "c.db")
    with Memory(tmp_path / "override.db", config=config) as memory:
        assert memory.location == str(tmp_path / "override.db")


def test_scoring_config_validation() -> None:
    with pytest.raises(ConfigurationError):
        ScoringConfig(
            base_weight=0,
            importance_weight=0,
            recency_weight=0,
            confidence_weight=0,
            frequency_weight=0,
        )
    with pytest.raises(ConfigurationError):
        ScoringConfig(keyword_weight=0, semantic_weight=0)
    with pytest.raises(ConfigurationError):
        ScoringConfig(recency_half_life_days=0)
    with pytest.raises(ConfigurationError):
        ScoringConfig(bm25_b=2)
    with pytest.raises(ConfigurationError):
        ScoringConfig(base_weight=True)  # type: ignore[arg-type]
