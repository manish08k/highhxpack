"""Deterministic hashing helpers."""

from __future__ import annotations

import hashlib
import uuid

from highhxpack.utils.text import normalize_for_hash


def content_hash(text: str) -> str:
    """Return a stable hash of ``text`` that ignores case, punctuation and spacing.

    Two memories with the same content hash are treated as exact duplicates.
    """
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def stable_hash64(value: str) -> int:
    """Return a 64-bit hash that is identical across processes and platforms.

    Python's built-in ``hash`` is randomized per process and therefore unusable
    for persisted feature hashing.
    """
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def new_id() -> str:
    """Return a new random identifier (32 lowercase hex characters)."""
    return uuid.uuid4().hex
