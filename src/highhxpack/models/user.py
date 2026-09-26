"""Per-user summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from highhxpack.utils.timestamps import to_storage


@dataclass(frozen=True, slots=True)
class UserSummary:
    """An overview of everything stored for one user."""

    user_id: str
    total: int
    active: int
    by_type: dict[str, int] = field(default_factory=dict)
    by_status: dict[str, int] = field(default_factory=dict)
    open_conflicts: int = 0
    relationships: int = 0
    first_memory_at: datetime | None = None
    last_memory_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total": self.total,
            "active": self.active,
            "by_type": dict(self.by_type),
            "by_status": dict(self.by_status),
            "open_conflicts": self.open_conflicts,
            "relationships": self.relationships,
            "first_memory_at": to_storage(self.first_memory_at) if self.first_memory_at else None,
            "last_memory_at": to_storage(self.last_memory_at) if self.last_memory_at else None,
        }
