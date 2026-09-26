"""``highhxpack stats``."""

from __future__ import annotations

import argparse
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context
from highhxpack.cli.display import format_bytes


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "stats",
        parents=[common],
        help="show storage statistics (all users unless --user is given)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, ctx: Context) -> int:
    stats = ctx.memory.stats(ctx.explicit_user)
    console = ctx.console
    if console.json_mode:
        console.json(stats.to_dict())
        return EXIT_OK
    scope = f"user {stats.user_id}" if stats.user_id else "all users"
    console.print(console.style(f"HighHXPack statistics ({scope})", "bold"))
    console.print(f"  memories:      {stats.active} active / {stats.total} total")
    if stats.user_id is None:
        console.print(f"  users:         {stats.users}")
    if stats.by_type:
        console.print("  by type:       " + ", ".join(f"{k} {v}" for k, v in stats.by_type.items()))
    if stats.by_status:
        console.print(
            "  by status:     " + ", ".join(f"{k} {v}" for k, v in stats.by_status.items())
        )
    console.print(f"  relationships: {stats.relationships}")
    console.print(f"  conflicts:     {stats.open_conflicts} open")
    console.print(
        f"  embeddings:    {stats.embedding_model or 'disabled'} "
        f"({stats.vectors} vectors, {stats.missing_vectors} missing)"
    )
    console.print(
        f"  storage:       {stats.storage_path} ({format_bytes(stats.storage_bytes)}, "
        f"schema v{stats.schema_version})"
    )
    return EXIT_OK
