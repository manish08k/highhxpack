"""``highhxpack consolidate``, ``highhxpack conflicts`` and ``highhxpack resolve``."""

from __future__ import annotations

import argparse
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context
from highhxpack.cli.display import format_time


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    consolidate = subparsers.add_parser(
        "consolidate",
        parents=[common],
        help="merge duplicates, archive expired memories and apply the conflict policy",
    )
    consolidate.add_argument("--dry-run", action="store_true", help="show what would change")
    consolidate.add_argument(
        "--summarize", action="store_true", help="also add summaries of related memories"
    )
    consolidate.set_defaults(handler=run_consolidate)

    conflicts = subparsers.add_parser(
        "conflicts", parents=[common], help="list contradictory memories"
    )
    conflicts.add_argument("--all", action="store_true", help="include resolved conflicts")
    conflicts.set_defaults(handler=run_conflicts)

    resolve = subparsers.add_parser(
        "resolve", parents=[common], help="resolve a conflict by choosing the memory to keep"
    )
    resolve.add_argument("conflict_id", help="conflict id (from `highhxpack conflicts`)")
    resolve.add_argument("--keep", required=True, help="id or prefix of the memory to keep")
    resolve.set_defaults(handler=run_resolve)


def run_consolidate(args: argparse.Namespace, ctx: Context) -> int:
    report = ctx.memory.consolidate(ctx.user, dry_run=args.dry_run, summarize=args.summarize)
    console = ctx.console
    if console.json_mode:
        console.json(report.to_dict())
        return EXIT_OK
    if not report.actions:
        console.info("Nothing to consolidate.")
        return EXIT_OK
    prefix = "Would " if report.dry_run else ""
    for action in report.actions:
        console.info(f"  {prefix}{action}" if prefix else f"  {action}")
    console.success(
        f"{'Planned' if report.dry_run else 'Done'}: {len(report.merged)} merged, "
        f"{len(report.superseded)} superseded, {report.conflicts_detected} conflicts, "
        f"{len(report.expired_archived)} expired archived, "
        f"{len(report.summaries_created)} summaries."
    )
    return EXIT_OK


def run_conflicts(args: argparse.Namespace, ctx: Context) -> int:
    memory = ctx.memory
    conflicts = memory.conflicts(ctx.explicit_user, status=None if args.all else "open")
    console = ctx.console
    if console.json_mode:
        console.json([c.to_dict() for c in conflicts])
        return EXIT_OK
    if not conflicts:
        console.info("No conflicts." if args.all else "No open conflicts.")
        return EXIT_OK
    records = memory.storage.get_memories([i for c in conflicts for i in (c.memory_id, c.other_id)])
    for c in conflicts:
        newer, older = records.get(c.memory_id), records.get(c.other_id)
        console.print(
            console.style(f"{c.id}", "bold") + f"  {c.kind}  {c.status}  "
            f"{format_time(c.created_at)}  user={c.user_id}"
        )
        if newer:
            console.print(f"  newer {newer.short_id} [{newer.status}]: {newer.content}")
        if older:
            console.print(f"  older {older.short_id} [{older.status}]: {older.content}")
        if c.resolution:
            console.print(console.style(f"  {c.resolution}", "dim"))
    return EXIT_OK


def run_resolve(args: argparse.Namespace, ctx: Context) -> int:
    memory = ctx.memory
    conflict = memory.resolve_conflict(args.conflict_id, keep=memory.resolve_id(args.keep))
    if ctx.console.json_mode:
        ctx.console.json(conflict.to_dict())
    else:
        ctx.console.success(f"Resolved: {conflict.resolution}")
    return EXIT_OK
