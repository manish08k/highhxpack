"""``highhxpack remember``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context
from highhxpack.exceptions import ValidationError
from highhxpack.models.result import RememberResult


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "remember",
        parents=[common],
        help="store a memory",
        description='Store a memory for the current user. Use "-" to read the text from stdin.',
    )
    parser.add_argument("text", nargs="+", help="the text to remember")
    parser.add_argument(
        "-t", "--type", dest="memory_type", help="memory type (inferred if omitted)"
    )
    parser.add_argument("--importance", type=float, help="importance between 0 and 1")
    parser.add_argument("--confidence", type=float, default=1.0, help="confidence between 0 and 1")
    parser.add_argument("--source", default="cli", help="where the memory came from (default: cli)")
    parser.add_argument("--ttl", help="forget after this long, e.g. 30d, 12h")
    parser.add_argument(
        "--meta", action="append", default=[], metavar="KEY=VALUE", help="metadata (repeatable)"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extract individual facts/preferences from the text instead of storing it verbatim",
    )
    parser.add_argument("--no-dedupe", action="store_true", help="store even if it is a duplicate")
    parser.set_defaults(handler=run)


def parse_meta(items: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValidationError(f"Invalid --meta value {item!r}.", hint="Use KEY=VALUE.")
        try:
            value: Any = json.loads(raw)
            if isinstance(value, dict | list):
                value = raw
        except ValueError:
            value = raw
        metadata[key.strip()] = value
    return metadata


def read_text(parts: list[str]) -> str:
    if parts == ["-"]:
        try:
            return sys.stdin.read()
        except UnicodeDecodeError as exc:
            raise ValidationError(
                "Standard input is not valid text in the terminal's encoding.",
                reason=str(exc),
                hint="Convert the input to UTF-8 (or set PYTHONIOENCODING) and retry.",
            ) from exc
    return " ".join(parts)


def run(args: argparse.Namespace, ctx: Context) -> int:
    text = read_text(args.text)
    memory = ctx.memory
    if args.extract:
        results = memory.ingest(
            ctx.user, text, source=args.source, metadata=parse_meta(args.meta) or None
        )
        if ctx.console.json_mode:
            ctx.console.json([r.to_dict() for r in results])
        elif not results:
            ctx.console.info("Nothing worth remembering was found in that text.")
        for result in results:
            _report(ctx, result)
        return EXIT_OK
    result = memory.remember(
        ctx.user,
        text,
        memory_type=args.memory_type,
        importance=args.importance,
        confidence=args.confidence,
        source=args.source,
        metadata=parse_meta(args.meta),
        ttl=args.ttl,
        dedupe=not args.no_dedupe,
    )
    if ctx.console.json_mode:
        ctx.console.json(result.to_dict())
    else:
        _report(ctx, result)
    return EXIT_OK


def _report(ctx: Context, result: RememberResult) -> None:
    console = ctx.console
    m = result.memory
    tag = console.style(f"[{m.memory_type}]", "cyan")
    if result.created:
        console.success(f"Remembered {console.style(m.short_id, 'bold')} {tag} {m.content}")
    else:
        console.info(
            f"Already known {console.sym('em')} reinforced {console.style(m.short_id, 'bold')} "
            f"(mentioned {m.mention_count} times): {m.content}"
        )
    for old in result.superseded_ids:
        console.info(f"  supersedes {old[:8]} (kept in history)")
    for conflict in result.conflicts:
        if conflict.status == "open":
            console.warn(
                f"conflicts with {conflict.other_id[:8]}; both kept (see `highhxpack conflicts`)"
            )
    for warning in result.warnings:
        console.warn(warning)
