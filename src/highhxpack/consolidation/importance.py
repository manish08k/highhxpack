"""Initial importance estimation and reinforcement.

Importance is a ``[0, 1]`` prior on how useful a memory is long-term.  It is
estimated once when a memory is stored (unless the caller supplies it) and
raised each time the same information is mentioned again.  All constants are in
:class:`ImportanceConfig`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from highhxpack.utils.text import words

_DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "goal": 0.7,
    "decision": 0.7,
    "preference": 0.6,
    "relationship": 0.6,
    "fact": 0.5,
    "knowledge": 0.5,
    "event": 0.4,
    "conversation": 0.3,
}
_EMPHASIS_WORDS = frozenset(
    "important crucial critical essential remember always never must allergic allergy "
    "deadline urgent password emergency medication".split()
)
_HEDGE_WORDS = frozenset("maybe perhaps probably possibly might guess think unsure".split())


@dataclass(frozen=True, slots=True)
class ImportanceConfig:
    #: Base importance per memory type.
    type_weights: Mapping[str, float] = field(default_factory=lambda: dict(_DEFAULT_TYPE_WEIGHTS))
    #: Base importance for memory types not listed in ``type_weights``.
    default_weight: float = 0.5
    #: Added when the text contains emphasis words ("important", "always", ...).
    emphasis_boost: float = 0.2
    #: Subtracted when the text is hedged ("maybe", "probably", ...).
    hedge_penalty: float = 0.1
    #: Fraction of the remaining headroom gained on each repeated mention.
    reinforcement_rate: float = 0.1


def estimate_importance(
    content: str, memory_type: str, config: ImportanceConfig | None = None
) -> float:
    """Heuristic importance for new content."""
    cfg = config or ImportanceConfig()
    value = cfg.type_weights.get(memory_type, cfg.default_weight)
    tokens = set(words(content))
    if tokens & _EMPHASIS_WORDS:
        value += cfg.emphasis_boost
    if tokens & _HEDGE_WORDS:
        value -= cfg.hedge_penalty
    return round(min(max(value, 0.0), 1.0), 4)


def reinforce(importance: float, config: ImportanceConfig | None = None) -> float:
    """Importance after the same information was mentioned once more."""
    cfg = config or ImportanceConfig()
    return round(min(importance + (1.0 - importance) * cfg.reinforcement_rate, 1.0), 4)
