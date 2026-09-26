from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from highhxpack.exceptions import ValidationError
from highhxpack.utils.hashing import content_hash, new_id, stable_hash64
from highhxpack.utils.logging import configure_cli_logging, get_logger, redact
from highhxpack.utils.text import jaccard, normalize_for_hash, stem, term_frequencies, tokenize
from highhxpack.utils.timestamps import (
    age_in_days,
    ensure_utc,
    from_storage,
    parse_datetime,
    parse_duration,
    to_storage,
)
from highhxpack.utils.validation import (
    MAX_CONTENT_LENGTH,
    MAX_LIMIT,
    validate_content,
    validate_limit,
    validate_memory_id,
    validate_memory_type,
    validate_metadata,
    validate_offset,
    validate_path,
    validate_query,
    validate_source,
    validate_unit_interval,
    validate_user_id,
)


class TestText:
    def test_tokenize_keeps_technical_tokens(self) -> None:
        assert tokenize("I love C++, C# and node.js with GPT-4") == [
            "lov",
            "c++",
            "c#",
            "node.js",
            "gpt-4",
        ]

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("prefers", "prefer"),
            ("preferred", "prefer"),
            ("programming", "program"),
            ("libraries", "library"),
            ("boxes", "box"),
            ("class", "class"),
            ("python", "python"),
            ("c++", "c++"),
            ("user's", "user"),
        ],
    )
    def test_stem(self, word: str, expected: str) -> None:
        assert stem(word) == expected

    def test_stop_words_removed(self) -> None:
        assert tokenize("What is the language that I prefer?") == ["languag", "prefer"]

    def test_normalize_for_hash_ignores_case_punctuation_space(self) -> None:
        assert normalize_for_hash("I use  Python.") == normalize_for_hash("i USE python")
        assert normalize_for_hash("C++") != normalize_for_hash("C")

    def test_curly_apostrophes_are_unified(self) -> None:
        assert tokenize("I\u2019m") == tokenize("I'm")

    def test_term_frequencies(self) -> None:
        assert term_frequencies("python python rust") == {"python": 2, "rust": 1}

    def test_jaccard(self) -> None:
        assert jaccard(set(), set()) == 0.0
        assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


class TestHashing:
    def test_content_hash_normalizes(self) -> None:
        assert content_hash("I use Python.") == content_hash("i use python")

    def test_stable_hash_is_deterministic(self) -> None:
        assert stable_hash64("abc") == stable_hash64("abc")
        assert stable_hash64("abc") != stable_hash64("abd")

    def test_new_id_format(self) -> None:
        ids = {new_id() for _ in range(100)}
        assert len(ids) == 100
        assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)


class TestTimestamps:
    def test_roundtrip(self) -> None:
        value = datetime(2024, 5, 1, 13, 45, 1, 123456, tzinfo=UTC)
        assert from_storage(to_storage(value)) == value

    def test_storage_format_sorts_chronologically(self) -> None:
        a = to_storage(datetime(2024, 1, 2, tzinfo=UTC))
        b = to_storage(datetime(2024, 1, 10, tzinfo=UTC))
        assert a < b

    def test_naive_is_utc_and_offsets_convert(self) -> None:
        assert ensure_utc(datetime(2024, 1, 1)).tzinfo is UTC
        plus_two = datetime(2024, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
        assert ensure_utc(plus_two).hour == 10

    def test_parse_datetime(self) -> None:
        assert parse_datetime("2024-05-01T10:00:00Z") == datetime(2024, 5, 1, 10, tzinfo=UTC)
        assert parse_datetime("2024-05-01") == datetime(2024, 5, 1, tzinfo=UTC)
        with pytest.raises(ValidationError, match="Invalid date"):
            parse_datetime("yesterday-ish")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("30d", timedelta(days=30)), ("12h", timedelta(hours=12)), ("1.5w", timedelta(weeks=1.5))],
    )
    def test_parse_duration(self, text: str, expected: timedelta) -> None:
        assert parse_duration(text) == expected

    @pytest.mark.parametrize("text", ["", "10", "5y", "-3d", "0d"])
    def test_parse_duration_invalid(self, text: str) -> None:
        with pytest.raises(ValidationError):
            parse_duration(text)

    def test_age_never_negative(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=UTC)
        assert age_in_days(now + timedelta(days=1), now) == 0.0
        assert age_in_days(now - timedelta(days=2), now) == pytest.approx(2.0)


class TestValidation:
    def test_user_id(self) -> None:
        assert validate_user_id("  alice ") == "alice"
        for bad in ["", "   ", "a\nb", "x" * 257, 3, None]:
            with pytest.raises(ValidationError):
                validate_user_id(bad)

    def test_content_strips_control_characters(self) -> None:
        assert validate_content("  hi\x1b[31m there\x00 ") == "hi[31m there"

    def test_content_limits(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            validate_content(" \x00 ")
        with pytest.raises(ValidationError, match="too long"):
            validate_content("x" * (MAX_CONTENT_LENGTH + 1))
        with pytest.raises(ValidationError, match="string"):
            validate_content(b"bytes")

    def test_query(self) -> None:
        assert validate_query(" python ") == "python"
        with pytest.raises(ValidationError):
            validate_query("")

    def test_memory_type(self) -> None:
        assert validate_memory_type("project_note") == "project_note"
        for bad in ["Fact", "1fact", "a-b", "", "x" * 40]:
            with pytest.raises(ValidationError):
                validate_memory_type(bad)

    def test_source(self) -> None:
        assert validate_source("slack:channel-1") == "slack:channel-1"
        with pytest.raises(ValidationError):
            validate_source("has space")

    def test_unit_interval(self) -> None:
        assert validate_unit_interval(1, field="x") == 1.0
        for bad in [-0.1, 1.01, float("nan"), True, "0.5"]:
            with pytest.raises(ValidationError):
                validate_unit_interval(bad, field="x")

    def test_limit_and_offset(self) -> None:
        assert validate_limit(5) == 5
        for bad in [0, MAX_LIMIT + 1, True, 2.0]:
            with pytest.raises(ValidationError):
                validate_limit(bad)
        with pytest.raises(ValidationError):
            validate_offset(-1)

    def test_memory_id(self) -> None:
        assert validate_memory_id(" " + "A" * 32 + " ") == "a" * 32
        with pytest.raises(ValidationError):
            validate_memory_id("abc")

    def test_metadata(self) -> None:
        assert validate_metadata(None) == {}
        assert validate_metadata({"a": [1, 2], "b": {"c": None}}) == {"a": [1, 2], "b": {"c": None}}
        with pytest.raises(ValidationError, match="JSON"):
            validate_metadata({"a": object()})
        with pytest.raises(ValidationError, match="JSON"):
            validate_metadata({"a": float("nan")})
        with pytest.raises(ValidationError, match="keys"):
            validate_metadata({1: "x"})
        with pytest.raises(ValidationError, match="larger"):
            validate_metadata({"a": "x" * 70_000})
        with pytest.raises(ValidationError, match="mapping"):
            validate_metadata(["a"])

    def test_path(self, tmp_path: Path) -> None:
        assert validate_path(str(tmp_path)) == tmp_path
        assert validate_path("~/x").is_absolute()
        for bad in ["", "a\x00b", 5]:
            with pytest.raises(ValidationError):
                validate_path(bad)


class TestLogging:
    def test_redact(self) -> None:
        text = "key sk-abcdefghijklmnop api_key=supersecret Authorization: Bearer abc.def"
        redacted = redact(text)
        assert "abcdefghijklmnop" not in redacted
        assert "supersecret" not in redacted
        assert "abc.def" not in redacted

    def test_cli_logging_levels(self) -> None:
        logger = get_logger()
        configure_cli_logging(0)
        assert logger.level == logging.WARNING
        configure_cli_logging(2)
        assert logger.level == logging.DEBUG
        handlers = [h for h in logger.handlers if getattr(h, "_highhxpack_cli", False)]
        assert len(handlers) == 1
        for handler in handlers:
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
