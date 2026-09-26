"""``highhxpack recall`` and ``highhxpack search``."""

from __future__ import annotations

import argparse
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context
from highhxpack.models.result import RecallResult


def _add_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", nargs="+", help="what to look for")
    parser.add_argument("-n", "--limit", type=int, default=10, help="maximum results (default: 10)")
    parser.add_argument("-t", "--type", dest="memory_type", action="append", help="only this type")
    parser.add_argument("--min-score", type=float, default=0.0, help="minimum score (0-1)")
    parser.add_argument(
        "--strategy",
        choices=("hybrid", "keyword", "semantic"),
        default="hybrid",
        help="retrieval strategy (default: hybrid)",
    )
    parser.add_argument("--explain", action="store_true", help="show the score breakdown")


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    recall = subparsers.add_parser(
        "recall",
        parents=[common],
        help="retrieve the current user's most relevant memories",
        description="Rank the current user's active memories by relevance, importance, "
        "recency, confidence and frequency.",
    )
    _add_query_options(recall)
    recall.set_defaults(handler=run_recall)

    search = subparsers.add_parser(
        "search",
        parents=[common],
        help="search memories (all users unless --user is given)",
        description="Search memories without updating access statistics. Searches every "
        "user unless --user is given.",
    )
    _add_query_options(search)
    search.add_argument(
        "--status",
        default="active",
        help="active, superseded, merged, archived or any (default: active)",
    )
    search.add_argument("--include-expired", action="store_true", help="include expired memories")
    search.set_defaults(handler=run_search)


def run_recall(args: argparse.Namespace, ctx: Context) -> int:
    results = ctx.memory.recall(
        ctx.user,
        " ".join(args.query),
        limit=args.limit,
        memory_type=args.memory_type,
        min_score=args.min_score,
        strategy=args.strategy,
    )
    _show(ctx, results, explain=args.explain, show_user=False)
    return EXIT_OK


def run_search(args: argparse.Namespace, ctx: Context) -> int:
    results = ctx.memory.search(
        " ".join(args.query),
        user_id=ctx.explicit_user,
        limit=args.limit,
        memory_type=args.memory_type,
        min_score=args.min_score,
        strategy=args.strategy,
        status=args.status,
        include_expired=args.include_expired,
    )
    _show(ctx, results, explain=args.explain, show_user=ctx.explicit_user is None)
    return EXIT_OK


def _show(ctx: Context, results: list[RecallResult], *, explain: bool, show_user: bool) -> None:
    console = ctx.console
    if console.json_mode:
        console.json([r.to_dict() for r in results])
        return
    if not results:
        console.info("No matching memories.")
        return
    headers = ["SCORE", "ID", "TYPE", *(["USER"] if show_user else []), "CONTENT"]
    rows = []
    for r in results:
        flag = f" {console.sym('warn')}" if r.has_conflict else ""
        rows.append(
            [
                f"{r.score:.3f}",
                r.memory.short_id,
                r.memory.memory_type,
                *([r.memory.user_id] if show_user else []),
                r.content + flag,
            ]
        )
    console.table(headers, rows, flex=len(headers) - 1)
    if explain:
        console.print()
        for r in results:
            b = r.breakdown
            console.print(
                console.style(r.memory.short_id, "bold")
                + f"  relevance={b.relevance:.3f} (keyword={b.keyword:.3f} "
                f"semantic={b.semantic:.3f})  importance={b.importance:.2f} "
                f"recency={b.recency:.2f} confidence={b.confidence:.2f} "
                f"frequency={b.frequency:.2f}"
            )
    if any(r.has_conflict for r in results):
        console.info(
            console.style(
                f"{console.sym('warn')} has an unresolved conflict; see `highhxpack conflicts`",
                "yellow",
            )
        )
