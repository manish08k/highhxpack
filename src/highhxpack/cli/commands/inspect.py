"""``highhxpack inspect`` (a user) and ``highhxpack history`` (a memory)."""

from __future__ import annotations

import argparse
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context
from highhxpack.cli.display import format_time


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    inspect = subparsers.add_parser(
        "inspect",
        parents=[common],
        help="summarize everything stored for a user",
    )
    inspect.add_argument("user_id", nargs="?", help="user to inspect (default: current user)")
    inspect.set_defaults(handler=run_inspect)

    history = subparsers.add_parser(
        "history",
        aliases=["show"],
        parents=[common],
        help="show a memory and its change history",
    )
    history.add_argument("memory_id", help="memory id or unique prefix")
    history.set_defaults(handler=run_history)


def run_inspect(args: argparse.Namespace, ctx: Context) -> int:
    memory = ctx.memory
    user = args.user_id or ctx.user
    summary = memory.inspect(user)
    edges = memory.graph.edges(user)
    console = ctx.console
    if console.json_mode:
        console.json({**summary.to_dict(), "graph": [e.to_dict() for e in edges]})
        return EXIT_OK
    console.print(console.style(f"User {summary.user_id}", "bold"))
    console.print(f"  memories:   {summary.active} active / {summary.total} total")
    if summary.by_type:
        types = ", ".join(f"{k} {v}" for k, v in summary.by_type.items())
        console.print(f"  by type:    {types}")
    if summary.by_status:
        statuses = ", ".join(f"{k} {v}" for k, v in summary.by_status.items())
        console.print(f"  by status:  {statuses}")
    console.print(f"  conflicts:  {summary.open_conflicts} open")
    console.print(
        f"  first/last: {format_time(summary.first_memory_at)} / "
        f"{format_time(summary.last_memory_at)}"
    )
    if edges:
        console.print(console.style("Knowledge graph", "bold"))
        for edge in edges[:50]:
            arrow = f"{console.sym('dash')}{edge.relation}{console.sym('arrow')}"
            console.print(f"  {edge.source} {arrow} {edge.target}")
        if len(edges) > 50:
            more = f"  {console.sym('more')} {len(edges) - 50} more (use --json)"
            console.print(console.style(more, "dim"))
    return EXIT_OK


def run_history(args: argparse.Namespace, ctx: Context) -> int:
    memory = ctx.memory
    record = memory.get(memory.resolve_id(args.memory_id))
    events = memory.history(record.id)
    console = ctx.console
    if console.json_mode:
        console.json({"memory": record.to_dict(), "history": [e.to_dict() for e in events]})
        return EXIT_OK
    console.print(console.style(record.id, "bold"))
    console.print(f"  content:     {record.content}")
    console.print(f"  user:        {record.user_id}")
    console.print(f"  type/status: {record.memory_type} / {record.status}")
    console.print(f"  importance:  {record.importance:.2f}   confidence: {record.confidence:.2f}")
    console.print(
        f"  version:     {record.version}   mentions: {record.mention_count}   "
        f"recalls: {record.access_count}"
    )
    console.print(
        f"  created:     {format_time(record.created_at)}   "
        f"updated: {format_time(record.updated_at)}"
    )
    if record.expires_at:
        console.print(f"  expires:     {format_time(record.expires_at)}")
    if record.superseded_by:
        console.print(f"  replaced by: {record.superseded_by}")
    if record.metadata:
        console.print(f"  metadata:    {record.metadata}")
    console.print(console.style("History", "bold"))
    for event in events:
        extra = f" {console.sym('arrow')} {event.related_id[:8]}" if event.related_id else ""
        reason = f" ({event.reason})" if event.reason else ""
        console.print(
            f"  {format_time(event.created_at)}  v{event.version}  {event.kind}{extra}{reason}"
        )
        console.print(console.style(f"      {event.content}", "dim"))
    return EXIT_OK
