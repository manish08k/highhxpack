"""Configuration loading.

Precedence, from lowest to highest (later sources override earlier ones):

1. built-in defaults,
2. the TOML configuration file,
3. ``HIGHHXPACK_*`` environment variables,
4. explicit arguments (Python keyword arguments or CLI options).

The configuration file is ``$HIGHHXPACK_CONFIG`` if set, otherwise
``<home>/config.toml`` when it exists.  ``<home>`` is ``$HIGHHXPACK_HOME`` or the
platform data directory (see :func:`default_home`).

API keys are never read from the configuration file, so the file can be shared
or committed safely; use environment variables (``OPENAI_API_KEY``) instead.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from highhxpack.consolidation.conflict import ConflictPolicy
from highhxpack.consolidation.deduplication import DEFAULT_THRESHOLD
from highhxpack.embeddings.providers import EMBEDDING_PROVIDERS
from highhxpack.exceptions import ConfigurationError, HighHXPackError
from highhxpack.providers import LLM_PROVIDERS
from highhxpack.providers.base import DEFAULT_TIMEOUT_SECONDS, validate_base_url
from highhxpack.providers.ollama import DEFAULT_OLLAMA_URL
from highhxpack.retrieval.ranking import ScoringConfig
from highhxpack.utils.validation import validate_path, validate_user_id

APP_NAME = "highhxpack"
CONFIG_FILENAME = "config.toml"
DATABASE_FILENAME = "memory.db"
EXTRACTORS = ("rules", "llm")
MAX_CONFIG_BYTES = 1024 * 1024

#: Environment variable -> configuration field.
ENV_VARS: dict[str, str] = {
    "HIGHHXPACK_STORAGE_PATH": "storage_path",
    "HIGHHXPACK_USER": "default_user",
    "HIGHHXPACK_EMBEDDING_PROVIDER": "embedding_provider",
    "HIGHHXPACK_EMBEDDING_MODEL": "embedding_model",
    "HIGHHXPACK_LLM_PROVIDER": "llm_provider",
    "HIGHHXPACK_LLM_MODEL": "llm_model",
    "HIGHHXPACK_OLLAMA_URL": "ollama_url",
    "HIGHHXPACK_OPENAI_BASE_URL": "openai_base_url",
    "HIGHHXPACK_REQUEST_TIMEOUT": "request_timeout",
    "HIGHHXPACK_DEDUP_THRESHOLD": "dedup_threshold",
    "HIGHHXPACK_CONFLICT_POLICY": "conflict_policy",
    "HIGHHXPACK_EXTRACTOR": "extractor",
    "HIGHHXPACK_TRACK_ACCESS": "track_access",
}
_FILE_FORBIDDEN = {"openai_api_key", "api_key"}
_SECRET_FIELDS = {"openai_api_key"}


def default_home(env: Mapping[str, str] | None = None) -> Path:
    """Directory for HighHXPack data when ``HIGHHXPACK_HOME`` is not set.

    * Linux/BSD: ``$XDG_DATA_HOME/highhxpack`` or ``~/.local/share/highhxpack``
    * macOS: ``~/Library/Application Support/highhxpack``
    * Windows: ``%LOCALAPPDATA%\\highhxpack``
    """
    env = os.environ if env is None else env
    if env.get("HIGHHXPACK_HOME"):
        return validate_path(env["HIGHHXPACK_HOME"], field="HIGHHXPACK_HOME")
    platform = sys.platform  # a variable, so type checkers do not prune branches
    if platform == "win32":  # pragma: no cover - platform specific
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if platform == "darwin":  # pragma: no cover - platform specific
        return Path.home() / "Library" / "Application Support" / APP_NAME
    xdg = env.get("XDG_DATA_HOME")  # pragma: no cover - platform specific
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / APP_NAME  # pragma: no cover


@dataclass(frozen=True, slots=True)
class Config:
    """Resolved configuration.  Build it with :meth:`Config.load` or directly."""

    home: Path = field(default_factory=default_home)
    storage_path: Path | None = None
    default_user: str = "default"
    embedding_provider: str = "hashing"
    embedding_model: str | None = None
    llm_provider: str = "none"
    llm_model: str | None = None
    ollama_url: str = DEFAULT_OLLAMA_URL
    openai_base_url: str | None = None
    openai_api_key: str | None = field(default=None, repr=False)
    request_timeout: float = DEFAULT_TIMEOUT_SECONDS
    dedup_threshold: float = DEFAULT_THRESHOLD
    conflict_policy: str = ConflictPolicy.SUPERSEDE.value
    extractor: str = "rules"
    track_access: bool = True
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    #: The configuration file that was loaded, if any.
    config_file: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "home", validate_path(self.home, field="home"))
        if self.storage_path is None:
            object.__setattr__(self, "storage_path", self.home / DATABASE_FILENAME)
        elif str(self.storage_path) != ":memory:":
            object.__setattr__(
                self, "storage_path", validate_path(self.storage_path, field="storage_path")
            )
        try:
            validate_user_id(self.default_user)
        except HighHXPackError as exc:
            raise ConfigurationError(f"Invalid default_user: {exc.message}") from exc
        for name in ("embedding_model", "llm_model", "openai_base_url", "openai_api_key"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ConfigurationError(f"{name} must be a string.")
        if not isinstance(self.track_access, bool):
            raise ConfigurationError("track_access must be true or false.")
        _one_of("embedding_provider", self.embedding_provider, EMBEDDING_PROVIDERS)
        _one_of("llm_provider", self.llm_provider, LLM_PROVIDERS)
        _one_of("conflict_policy", self.conflict_policy, tuple(p.value for p in ConflictPolicy))
        _one_of("extractor", self.extractor, EXTRACTORS)
        validate_base_url(self.ollama_url, provider="Ollama")
        if self.openai_base_url:
            validate_base_url(self.openai_base_url, provider="OpenAI")
        if not isinstance(self.request_timeout, int | float) or not (
            0 < self.request_timeout <= 3600
        ):
            raise ConfigurationError("request_timeout must be between 0 and 3600 seconds.")
        value = self.dedup_threshold
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or math.isnan(value)
            or not 0 < value <= 1
        ):
            raise ConfigurationError(f"dedup_threshold must be in (0, 1], got {value!r}.")
        if self.extractor == "llm" and self.llm_provider == "none":
            raise ConfigurationError(
                "extractor = 'llm' requires an LLM provider.",
                hint="Set llm_provider to 'ollama' or 'openai', or use extractor = 'rules'.",
            )

    @property
    def database_path(self) -> Path:
        """The database location (``storage_path`` is always set after init)."""
        return self.storage_path if self.storage_path is not None else self.home / DATABASE_FILENAME

    def replace(self, **changes: Any) -> Config:
        return dataclasses.replace(self, **changes)

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        """Serializable view; secrets are masked unless ``redact=False``."""
        result: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name in _SECRET_FIELDS:
                value = ("***" if value else None) if redact else value
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, ScoringConfig):
                value = value.to_dict()
            result[f.name] = value
        return result

    @classmethod
    def load(
        cls,
        *,
        config_file: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> Config:
        """Resolve configuration from all sources (see the module docstring).

        ``overrides`` whose value is ``None`` are ignored, so CLI options that
        were not given do not mask lower-precedence sources.
        """
        env = os.environ if env is None else env
        known = {f.name for f in dataclasses.fields(cls)} - {"config_file"}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ConfigurationError(f"Unknown configuration option: {unknown[0]!r}")
        explicit = {k: v for k, v in overrides.items() if v is not None}

        home = validate_path(explicit["home"]) if "home" in explicit else default_home(env)
        path = _config_file_path(config_file, env, home)
        values: dict[str, Any] = {}
        if path is not None:
            values.update(_read_config_file(path, known))
        for var, name in ENV_VARS.items():
            raw = env.get(var)
            if raw is not None and raw.strip():
                values[name] = _coerce(name, raw.strip(), source=var)
        values.update(explicit)

        scoring = values.pop("scoring", None)
        if isinstance(scoring, Mapping):
            scoring = ScoringConfig.from_mapping(scoring)
        elif scoring is not None and not isinstance(scoring, ScoringConfig):
            raise ConfigurationError("scoring must be a table of scoring options.")
        values["home"] = home
        try:
            return cls(**values, scoring=scoring or ScoringConfig(), config_file=path)
        except ConfigurationError:
            raise
        except HighHXPackError as exc:
            where = f" (loaded from {path})" if path else ""
            raise ConfigurationError(f"Invalid configuration{where}: {exc.message}") from exc


def _one_of(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigurationError(
            f"Invalid {name}: {value!r}", hint=f"Use one of: {', '.join(allowed)}."
        )


def _config_file_path(
    explicit: str | Path | None, env: Mapping[str, str], home: Path
) -> Path | None:
    if explicit is not None or env.get("HIGHHXPACK_CONFIG"):
        path = validate_path(
            explicit if explicit is not None else env["HIGHHXPACK_CONFIG"], field="config_file"
        )
        if not path.is_file():
            raise ConfigurationError(
                f"Configuration file not found: {path}",
                hint="Create one with `highhxpack init` or fix the path.",
            )
        return path
    candidate = home / CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def _read_config_file(path: Path, known: set[str]) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            raise ConfigurationError(f"Configuration file {path} is larger than 1 MiB.")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(
            f"Configuration file {path} is not valid TOML.",
            reason=str(exc),
            hint="Fix the syntax or regenerate it with `highhxpack init --force`.",
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigurationError(
            f"Cannot read configuration file {path}.", reason=str(exc)
        ) from exc
    forbidden = sorted(set(data) & _FILE_FORBIDDEN)
    if forbidden:
        raise ConfigurationError(
            f"{forbidden[0]!r} is not allowed in the configuration file.",
            reason="API keys in files are easily leaked.",
            hint="Remove it and set the OPENAI_API_KEY environment variable instead.",
        )
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigurationError(
            f"Unknown option {unknown[0]!r} in {path}.",
            hint=f"Valid options: {', '.join(sorted(known - _SECRET_FIELDS))}.",
        )
    values = dict(data)
    storage = values.get("storage_path")
    if isinstance(storage, str) and storage != ":memory:":
        candidate = Path(storage).expanduser()
        # Relative paths in the file are relative to the file, not the cwd.
        values["storage_path"] = candidate if candidate.is_absolute() else path.parent / candidate
    if "home" in values:
        raise ConfigurationError(
            "'home' cannot be set in the configuration file (the file lives in home).",
            hint="Use the HIGHHXPACK_HOME environment variable instead.",
        )
    return values


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(name: str, raw: str, *, source: str) -> Any:
    if name == "track_access":
        lowered = raw.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ConfigurationError(f"{source} must be true or false, got {raw!r}.")
    if name in {"request_timeout", "dedup_threshold"}:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{source} must be a number, got {raw!r}.") from exc
    if name == "storage_path":
        return raw if raw == ":memory:" else Path(raw).expanduser()
    return raw


CONFIG_TEMPLATE = """\
# HighHXPack configuration.
# Precedence: defaults < this file < HIGHHXPACK_* environment variables < CLI options.
# API keys are never read from this file; use environment variables (OPENAI_API_KEY).

# Database file. Relative paths are resolved against this file's directory.
# storage_path = "memory.db"

# User id used by the CLI when --user is not given.
default_user = "default"

# Embeddings: "hashing" (local, default), "none", "sentence-transformers", "ollama", "openai".
embedding_provider = "hashing"
# embedding_model = "nomic-embed-text"

# LLM (only needed for extractor = "llm" and LLM summaries): "none", "ollama", "openai".
llm_provider = "none"
# llm_model = "llama3.2"
# ollama_url = "http://localhost:11434"

# Automatic extraction used by `ingest`: "rules" (offline) or "llm".
extractor = "rules"

# What happens when new information contradicts old: "supersede" or "keep_both".
conflict_policy = "supersede"

# [scoring]
# base_weight = 0.60
# importance_weight = 0.15
# recency_weight = 0.10
# confidence_weight = 0.10
# frequency_weight = 0.05
# recency_half_life_days = 30.0
"""


def write_config_template(path: Path, *, overwrite: bool = False) -> None:
    """Write :data:`CONFIG_TEMPLATE` to ``path`` with owner-only permissions."""
    if path.exists() and not overwrite:
        raise ConfigurationError(f"{path} already exists.", hint="Pass --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(CONFIG_TEMPLATE)
