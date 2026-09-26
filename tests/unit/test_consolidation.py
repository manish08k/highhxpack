from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from highhxpack import CallableProvider, ConflictPolicy, ProviderError
from highhxpack.consolidation import (
    ConflictKind,
    ExtractiveSummarizer,
    LLMSummarizer,
    canonical_tokens,
    detect_conflicts,
    estimate_importance,
    find_duplicate_pairs,
    group_by_entity,
    is_near_duplicate,
    parse_claim,
    reinforce,
)
from highhxpack.consolidation.conflict import parse_policy, should_supersede
from highhxpack.consolidation.deduplication import best_duplicate, similarity
from highhxpack.consolidation.importance import ImportanceConfig
from highhxpack.exceptions import ValidationError
from highhxpack.models.memory import MemoryRecord
from highhxpack.utils.hashing import new_id
from highhxpack.utils.text import jaccard

NOW = datetime(2025, 1, 1, tzinfo=UTC)


def rec(content: str, **changes: object) -> MemoryRecord:
    return MemoryRecord(
        id=new_id(),
        user_id="u",
        content=content,
        memory_type="fact",
        source="api",
        importance=0.5,
        confidence=1.0,
        created_at=NOW,
        updated_at=NOW,
    ).with_changes(**changes)


class TestDeduplication:
    def test_spec_example_all_duplicates(self) -> None:
        a, b, c = "User likes Python.", "User likes Python.", "User prefers Python."
        assert is_near_duplicate(a, b)
        assert is_near_duplicate(a, c)
        assert is_near_duplicate("I love Python", c)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("I like Python", "I don't like Python"),
            ("I prefer Python", "I prefer Java"),
            ("I use Python and Rust", "I use Python and Go"),
            ("I have 2 cats", "I have 3 cats"),
            ("I prefer Python", "I prefer Python for web development"),
        ],
    )
    def test_not_duplicates(self, a: str, b: str) -> None:
        assert not is_near_duplicate(a, b)

    def test_canonical_tokens(self) -> None:
        assert canonical_tokens("The user really loves Python!") == frozenset({"prefer", "python"})

    def test_best_duplicate(self) -> None:
        match = best_duplicate(
            "User prefers Python", [("1", "I prefer Rust"), ("2", "I like Python")]
        )
        assert match is not None
        assert match.memory_id == "2"
        assert best_duplicate("zzz", [("1", "I prefer Rust")]) is None

    def test_prefix_filtering_matches_brute_force(self) -> None:
        rng = random.Random(7)
        vocab = [f"w{i}" for i in range(12)]
        items = [(str(i), " ".join(rng.sample(vocab, rng.randint(2, 5)))) for i in range(150)]
        items += [("neg", "not w1 w2"), ("pos", "w1 w2")]
        for threshold in (0.5, 0.85, 1.0):
            fast = {(a, b) for a, b, _ in find_duplicate_pairs(items, threshold=threshold)}
            brute = {
                (items[i][0], items[j][0])
                for i in range(len(items))
                for j in range(i + 1, len(items))
                if similarity(items[i][1], items[j][1]) >= threshold
            }
            assert fast == brute


class TestConflicts:
    @pytest.mark.parametrize(
        ("text", "slot", "value"),
        [
            ("I prefer Java", "prefers:*", "java"),
            ("User prefers Python.", "prefers:*", "python"),
            ("I prefer Python for AI", "prefers:ai", "python"),
            ("My favorite color is blue", "favorite:color", "blu"),
            ("I like Python", "sentiment:python", "positive"),
            ("I can't stand Python", "sentiment:python", "negative"),
            ("I no longer use Vim", "uses:vim", "negative"),
            ("I live in Paris", "lives_in", "paris"),
            ("I work for Acme", "works_at", "acm"),
            ("Call me Ada", "name", "ada"),
            ("I'm 30 years old", "age", "30"),
            ("My manager is Bob", "attr:manager", "bob"),
            # Regression: temporal fillers must not change the slot or value.
            ("I prefer Rust for pipelines now", "prefers:pipelin", "rust"),
            ("I now prefer Go", "prefers:*", "go"),
            ("I actually prefer tea these days", "prefers:*", "tea"),
            ("I live in Rome now", "lives_in", "rom"),
        ],
    )
    def test_parse_claim(self, text: str, slot: str, value: str) -> None:
        claim = parse_claim(text)
        assert claim is not None
        assert (claim.slot, claim.value) == (slot, value)

    @pytest.mark.parametrize(
        "text", ["The sky is blue", "My sister is a doctor", "I went home", "Hello"]
    )
    def test_no_claim(self, text: str) -> None:
        assert parse_claim(text) is None

    def test_detect(self) -> None:
        claim = parse_claim("I prefer Python")
        assert claim is not None
        java, python, rust_web = (
            rec("I prefer Java"),
            rec("I prefer python"),
            rec("I prefer Rust for web"),
        )
        found = detect_conflicts(claim, [java, python, rust_web])
        assert [(f.existing.id, f.kind) for f in found] == [(java.id, ConflictKind.CHANGED_VALUE)]
        polar = parse_claim("I hate Python")
        assert polar is not None
        found = detect_conflicts(polar, [rec("I love Python")])
        assert found[0].kind == ConflictKind.CONTRADICTION

    def test_policy(self) -> None:
        assert parse_policy("keep_both") == ConflictPolicy.KEEP_BOTH
        with pytest.raises(ValidationError):
            parse_policy("newest_wins")
        sup = ConflictPolicy.SUPERSEDE
        assert should_supersede(sup, new_confidence=1.0, old_confidence=1.0)
        assert not should_supersede(sup, new_confidence=0.5, old_confidence=0.9)
        assert not should_supersede(
            ConflictPolicy.KEEP_BOTH, new_confidence=1.0, old_confidence=0.1
        )


class TestImportance:
    def test_type_and_wording(self) -> None:
        assert estimate_importance("I want to ship v1", "goal") == 0.7
        assert estimate_importance("Remember: I am allergic to nuts", "fact") == 0.7
        assert estimate_importance("Maybe I like jazz", "preference") == 0.5
        assert estimate_importance("Something", "custom_type") == 0.5
        cfg = ImportanceConfig(default_weight=0.9)
        assert estimate_importance("Something", "custom_type", cfg) == 0.9

    def test_reinforce_is_monotonic_and_bounded(self) -> None:
        value = 0.5
        for _ in range(100):
            new = reinforce(value)
            assert value <= new <= 1.0
            value = new
        assert value == pytest.approx(1.0, abs=1e-3)


class TestSummarization:
    def test_extractive_is_verbatim_and_bounded(self) -> None:
        summary = ExtractiveSummarizer().summarize(
            "Python", ["I use Python.", "i use python", "I prefer Python for AI"]
        )
        assert summary == "About Python: I use Python; I prefer Python for AI."
        short = ExtractiveSummarizer(max_chars=20).summarize("X", ["a" * 50])
        assert len(short) == 20
        assert short.endswith("...")

    def test_llm_summarizer(self) -> None:
        prompts: list[str] = []

        def llm(prompt: str) -> str:
            prompts.append(prompt)
            return "  User uses Python.  "

        assert LLMSummarizer(CallableProvider(llm)).summarize("Python", ["a", "b"]) == (
            "User uses Python."
        )
        assert "- a\n- b" in prompts[0]
        with pytest.raises(ProviderError, match="empty"):
            LLMSummarizer(CallableProvider(lambda p: " ")).summarize("x", ["a"])

    def test_group_by_entity(self) -> None:
        memories = [
            rec("I use Python daily", importance=0.2),
            rec("Python is my first language", importance=0.9),
            rec("I teach Python and Rust"),
            rec("I like Rust"),
        ]
        groups = group_by_entity(memories, min_size=2)
        assert [(g.topic, len(g.memories)) for g in groups] == [("Python", 3), ("Rust", 2)]
        assert groups[0].memories[0].importance == 0.9


def test_jaccard_symmetry() -> None:
    a, b = canonical_tokens("I love Python"), canonical_tokens("Python love")
    assert jaccard(a, b) == jaccard(b, a)
