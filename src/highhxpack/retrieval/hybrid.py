"""Fusion of keyword and semantic candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    memory_id: str
    keyword: float
    semantic: float


def merge_candidates(
    keyword: Mapping[str, float], semantic: Mapping[str, float]
) -> list[Candidate]:
    """Union both channels; a memory missing from one channel scores 0 there.

    The result is ordered by memory id so downstream processing is deterministic.
    """
    ids = sorted(set(keyword) | set(semantic))
    return [Candidate(i, keyword.get(i, 0.0), semantic.get(i, 0.0)) for i in ids]
