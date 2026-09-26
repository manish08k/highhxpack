"""Input validation for everything that crosses the public API boundary."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from highhxpack.exceptions import ValidationError

MAX_USER_ID_LENGTH = 256
MAX_CONTENT_LENGTH = 32_000
MAX_METADATA_BYTES = 64 * 1024
MAX_QUERY_LENGTH = 4_000
MAX_LIMIT = 1_000

_MEMORY_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SOURCE_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
_MEMORY_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _require_encodable(value: str, field: str) -> None:
    """Reject strings that cannot be stored as UTF-8 (e.g. lone surrogates)."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValidationError(
            f"{field} contains characters that are not valid Unicode text.",
            reason=str(exc),
            hint="Decode input with errors='replace' before storing it.",
        ) from exc


def validate_user_id(user_id: object) -> str:
    """Return a validated user identifier."""
    if not isinstance(user_id, str):
        raise ValidationError(f"user_id must be a string, got {type(user_id).__name__}.")
    value = user_id.strip()
    if not value:
        raise ValidationError("user_id must not be empty.")
    _require_encodable(value, "user_id")
    if len(value) > MAX_USER_ID_LENGTH:
        raise ValidationError(f"user_id must be at most {MAX_USER_ID_LENGTH} characters.")
    if _CONTROL_CHARS_RE.search(value) or "\n" in value or "\r" in value or "\t" in value:
        raise ValidationError("user_id must not contain control characters or whitespace breaks.")
    return value


def validate_content(content: object, *, field: str = "content") -> str:
    """Return validated, whitespace-trimmed memory content."""
    if not isinstance(content, str):
        raise ValidationError(f"{field} must be a string, got {type(content).__name__}.")
    value = _CONTROL_CHARS_RE.sub("", content).strip()
    if not value:
        raise ValidationError(f"{field} must not be empty.")
    _require_encodable(value, field)
    if len(value) > MAX_CONTENT_LENGTH:
        raise ValidationError(
            f"{field} is too long ({len(value)} characters, maximum {MAX_CONTENT_LENGTH}).",
            hint="Split long documents into smaller memories or use `ingest()`.",
        )
    return value


def validate_query(query: object) -> str:
    """Return a validated search query."""
    if not isinstance(query, str):
        raise ValidationError(f"query must be a string, got {type(query).__name__}.")
    value = query.strip()
    if not value:
        raise ValidationError("query must not be empty.")
    _require_encodable(value, "query")
    if len(value) > MAX_QUERY_LENGTH:
        raise ValidationError(f"query must be at most {MAX_QUERY_LENGTH} characters.")
    return value


def validate_memory_type(memory_type: object) -> str:
    """Return a validated memory type name.

    Memory types are open-ended: any lower-case identifier such as ``"fact"`` or
    ``"project_note"`` is accepted, so new types need no code changes.
    """
    value = str(memory_type.value) if hasattr(memory_type, "value") else memory_type
    if not isinstance(value, str) or not _MEMORY_TYPE_RE.match(value):
        raise ValidationError(
            f"Invalid memory_type: {memory_type!r}",
            hint="Use a lower-case identifier (max 32 characters), e.g. 'fact' or 'project_note'.",
        )
    return value


def validate_source(source: object) -> str:
    """Return a validated source label."""
    if not isinstance(source, str) or not _SOURCE_RE.match(source):
        raise ValidationError(
            f"Invalid source: {source!r}",
            hint="Use up to 64 letters, digits, '_', '.', ':' or '-'.",
        )
    return source


def validate_unit_interval(value: object, *, field: str) -> float:
    """Return ``value`` as a float in the closed interval [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field} must be a number between 0 and 1.")
    number = float(value)
    if math.isnan(number) or not 0.0 <= number <= 1.0:
        raise ValidationError(f"{field} must be between 0 and 1, got {value!r}.")
    return number


def validate_limit(limit: object, *, field: str = "limit") -> int:
    """Return a validated positive result limit."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValidationError(f"{field} must be an integer.")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValidationError(f"{field} must be between 1 and {MAX_LIMIT}, got {limit}.")
    return limit


def validate_offset(offset: object) -> int:
    """Return a validated non-negative offset."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValidationError("offset must be a non-negative integer.")
    return offset


def validate_memory_id(memory_id: object) -> str:
    """Return a validated full memory identifier."""
    if not isinstance(memory_id, str) or not _MEMORY_ID_RE.match(memory_id.strip().lower()):
        raise ValidationError(
            f"Invalid memory id: {memory_id!r}",
            hint="Memory ids are 32 hexadecimal characters.",
        )
    return memory_id.strip().lower()


def validate_metadata(metadata: object) -> dict[str, Any]:
    """Return a validated, JSON-serializable copy of ``metadata``."""
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValidationError(f"metadata must be a mapping, got {type(metadata).__name__}.")
    for key in metadata:
        if not isinstance(key, str):
            raise ValidationError("metadata keys must be strings.")
    try:
        encoded = json.dumps(dict(metadata), allow_nan=False, sort_keys=True, ensure_ascii=False)
        encoded.encode("utf-8")  # rejects lone surrogates
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "metadata must be JSON-serializable.",
            reason=str(exc),
            hint="Use only str, int, float, bool, None, lists and dicts.",
        ) from exc
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValidationError(f"metadata is larger than {MAX_METADATA_BYTES} bytes when encoded.")
    result: dict[str, Any] = json.loads(encoded)
    return result


def validate_path(path: object, *, field: str = "path") -> Path:
    """Return ``path`` as an expanded :class:`Path` after basic sanity checks."""
    if isinstance(path, Path):
        candidate = path
    elif isinstance(path, str):
        if not path.strip():
            raise ValidationError(f"{field} must not be empty.")
        if "\x00" in path:
            raise ValidationError(f"{field} must not contain NUL bytes.")
        candidate = Path(path)
    else:
        raise ValidationError(f"{field} must be a string or pathlib.Path.")
    return candidate.expanduser()
