from __future__ import annotations

import json

import pytest

from highhxpack import CallableProvider, ExtractionError
from highhxpack.extraction import (
    LLMExtractor,
    RuleBasedExtractor,
    classify_memory_type,
    extract_entities,
    split_clauses,
)
from highhxpack.extraction.extractor import MAX_EXTRACTED_PER_CALL, parse_llm_output
from highhxpack.extraction.patterns import expand_contractions, object_phrase

extract = RuleBasedExtractor().extract


def test_spec_example() -> None:
    text = (
        "I've been using Python for machine learning for three years, "
        "but I prefer C++ for competitive programming."
    )
    memories = extract(text)
    assert [(m.memory_type, m.content) for m in memories] == [
        ("fact", "I've been using Python for machine learning for three years"),
        ("preference", "I prefer C++ for competitive programming"),
    ]
    edges = [(r.source, r.relation, r.target) for m in memories for r in m.relations]
    assert edges == [("user", "uses", "Python"), ("user", "prefers", "C++")]
    assert memories[1].relations[0].metadata == {"context": "competitive programming"}


@pytest.mark.parametrize(
    "text",
    [
        "Hi!",
        "Thanks, that sounds good.",
        "What language should I learn?",
        "Can you help me with this bug",
        "ok",
        "The weather is nice.",
        "",
    ],
)
def test_not_everything_is_stored(text: str) -> None:
    assert extract(text) == []


@pytest.mark.parametrize(
    ("text", "memory_type", "relation", "target"),
    [
        ("I prefer tabs over spaces", "preference", "prefers", "tabs"),
        ("I really love Rust", "preference", "likes", "Rust"),
        ("I don't like Java", "preference", "dislikes", "Java"),
        ("My favorite editor is Neovim", "preference", "favorite_editor", "Neovim"),
        ("I'm interested in AI safety", "preference", "interested_in", "AI"),
        ("I work at Acme Corp", "fact", "works_at", "Acme Corp"),
        ("I live in Paris", "fact", "lives_in", "Paris"),
        ("I'm working on HighHXPack", "fact", "works_on", "HighHXPack"),
        ("My name is Ada", "fact", "named", "Ada"),
        ("I am allergic to peanuts", "fact", "allergic_to", "peanuts"),
        ("I am a data scientist", "fact", "is_a", "data scientist"),
        ("My timezone is UTC+2", "fact", "timezone", "UTC+2"),
        ("I want to learn Japanese", "goal", "wants_to", "learn Japanese"),
        ("My goal is to run a marathon", "goal", "wants_to", "run a marathon"),
        ("We decided to use Postgres for the backend", "decision", "decided", "Postgres"),
        ("We'll go with Flutter", "decision", "decided", "Flutter"),
        ("My sister Anna is a doctor", "relationship", "sister", "Anna"),
        ("My manager's name is Bob", "relationship", "manager", "Bob"),
    ],
)
def test_classification(text: str, memory_type: str, relation: str, target: str) -> None:
    [memory] = extract(text)
    assert memory.memory_type == memory_type
    assert [(r.relation, r.target) for r in memory.relations] == [(relation, target)]


@pytest.mark.parametrize(
    "text",
    ["I went to PyCon last week", "I have a meeting tomorrow", "I started a new job in 2023"],
)
def test_events(text: str) -> None:
    [memory] = extract(text)
    assert memory.memory_type == "event"


def test_duplicates_within_text_collapsed_and_capitalized() -> None:
    memories = extract("i use python. I use Python.")
    assert [m.content for m in memories] == ["I use python"]


def test_output_is_bounded() -> None:
    text = " ".join(f"I use tool{i}." for i in range(MAX_EXTRACTED_PER_CALL + 10))
    assert len(extract(text)) == MAX_EXTRACTED_PER_CALL


def test_split_clauses() -> None:
    assert split_clauses("I like tea, but I prefer coffee; my dog is Rex.\nWhat?") == [
        "I like tea",
        "I prefer coffee",
        "my dog is Rex",
        "What?",
    ]


def test_classify_memory_type_default() -> None:
    assert classify_memory_type("I prefer dark mode") == "preference"
    assert classify_memory_type("Deploy script lives in /ops") == "fact"
    assert classify_memory_type("random note", default="knowledge") == "knowledge"


def test_contractions_and_object_phrase() -> None:
    assert expand_contractions("I\u2019ve got it, we're done, don't") == (
        "I have got it, we are done, do not"
    )
    phrase = object_phrase("the new laptop for work.")
    assert phrase is not None
    assert (phrase.target, phrase.context) == ("new laptop", "work")
    assert object_phrase(" .") is None


def test_entities() -> None:
    entities = extract_entities("Ada uses C++ and Node.js at Google. Then she moved to New York.")
    assert [(e.name, e.kind) for e in entities] == [
        ("Ada", "name"),
        ("C++", "technology"),
        ("Node.js", "technology"),
        ("Google", "name"),
        ("New York", "name"),
    ]
    assert extract_entities("I think so") == []


class TestLLMExtraction:
    def test_valid_output(self) -> None:
        payload = {
            "memories": [
                {
                    "content": "User prefers Python",
                    "memory_type": "preference",
                    "confidence": 0.9,
                    "importance": 0.7,
                    "relations": [{"source": "user", "relation": "prefers", "target": "Python"}],
                },
                {"content": "", "memory_type": "fact"},  # invalid: skipped
                {"content": "x", "memory_type": "Bad Type"},  # invalid: skipped
                {"content": "y", "confidence": 5},  # invalid: skipped
                "not an object",
            ]
        }
        seen: list[str] = []

        def llm(prompt: str) -> str:
            seen.append(prompt)
            return "```json\n" + json.dumps(payload) + "\n```"

        results = LLMExtractor(CallableProvider(llm)).extract("I like Python")
        assert len(results) == 1
        assert results[0].importance == 0.7
        assert results[0].relations[0].target == "Python"
        assert "I like Python" in seen[0]

    def test_bare_list_and_bad_relations(self) -> None:
        raw = json.dumps([{"content": "A fact", "relations": [{"source": "user"}, 3]}])
        [memory] = parse_llm_output(raw)
        assert memory.memory_type == "fact"
        assert memory.relations == ()

    @pytest.mark.parametrize("raw", ["not json", '{"other": []}', "42"])
    def test_invalid_output_raises(self, raw: str) -> None:
        with pytest.raises(ExtractionError):
            parse_llm_output(raw)

    def test_provider_failure_and_long_input(self) -> None:
        def broken(prompt: str) -> str:
            raise TimeoutError("slow")

        with pytest.raises(ExtractionError, match="failed"):
            LLMExtractor(CallableProvider(broken)).extract("text")
        with pytest.raises(ExtractionError, match="too long"):
            LLMExtractor(CallableProvider(lambda p: "{}"), max_input_chars=5).extract("123456")
