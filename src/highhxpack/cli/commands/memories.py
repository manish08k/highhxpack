"""``highhxpack memories``."""

from __future__ import annotations

import argparse
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context
from highhxpack.cli.display import format_time


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "memories",
        aliases=["list"],
        parents=[common],
        help="list stored memories",
        description="List the current user's memories (newest first).",
    )
    parser.add_argument("-t", "--type", dest="memory_type", action="append", help="only this type")
    parser.add_argument(
        "--status",
        default="active",
        help="active, superseded, merged, archived or any (default: active)",
    )
    parser.add_argument("-n", "--limit", type=int, default=50, help="maximum rows (default: 50)")
    parser.add_argument("--offset", type=int, default=0, help="skip this many rows")
    parser.add_argument(
        "--order",
        choices=("newest", "oldest", "importance", "updated"),
        default="newest",
        help="sort order (default: newest)",
    )
    parser.add_argument("--all-users", action="store_true", help="list memories of every user")
    parser.add_argument("--include-expired", action="store_true", help="include expired memories")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, ctx: Context) -> int:
    memory = ctx.memory
    user = None if args.all_users else ctx.user
    records = memory.list(
        user,
        memory_type=args.memory_type,
        status=args.status,
        limit=args.limit,
        offset=args.offset,
        order=args.order,
        include_expired=args.include_expired,
    )
    console = ctx.console
    if console.json_mode:
        console.json([r.to_dict() for r in records])
        return EXIT_OK
    if not records:
        scope = "any user" if user is None else f"user {user!r}"
        console.info(f'No memories for {scope}. Add one with: highhxpack remember "..."')
        return EXIT_OK
    show_user = user is None
    show_status = args.status != "active"
    headers = [
        "ID",
        *(["USER"] if show_user else []),
        "TYPE",
        *(["STATUS"] if show_status else []),
        "IMP",
        "CREATED",
        "CONTENT",
    ]
    rows = [
        [
            r.short_id,
            *([r.user_id] if show_user else []),
            r.memory_type,
            *([r.status] if show_status else []),
            f"{r.importance:.2f}",
            format_time(r.created_at),
            r.content,
        ]
        for r in records
    ]
    console.table(headers, rows, flex=len(headers) - 1)
    total = memory.count(
        user, memory_type=args.memory_type, status=args.status, include_expired=args.include_expired
    )
    if total > args.offset + len(records):
        console.info(
            console.style(
                f"Showing {len(records)} of {total}. Use --offset/--limit to see more.", "dim"
            )
        )
    return EXIT_OK
