"""``highhxpack forget`` and ``highhxpack restore``."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from highhxpack.cli.context import EXIT_ERROR, EXIT_OK, Context
from highhxpack.exceptions import ValidationError


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    forget = subparsers.add_parser(
        "forget",
        parents=[common],
        help="archive (or permanently delete) a memory",
        description="Archive a memory so it is no longer recalled; it stays in history and "
        "can be restored. With --permanent the memory and its history are erased.",
    )
    forget.add_argument("memory_id", help="memory id or unique prefix (at least 4 characters)")
    forget.add_argument("--permanent", action="store_true", help="erase instead of archiving")
    forget.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    forget.set_defaults(handler=run_forget)

    restore = subparsers.add_parser(
        "restore",
        parents=[common],
        help="make an archived or superseded memory active again",
    )
    restore.add_argument("memory_id", help="memory id or unique prefix")
    restore.set_defaults(handler=run_restore)


def run_forget(args: argparse.Namespace, ctx: Context) -> int:
    memory = ctx.memory
    memory_id = memory.resolve_id(args.memory_id)
    record = memory.get(memory_id)
    if args.permanent and not args.yes:
        if not sys.stdin.isatty() or ctx.console.json_mode:
            # Never erase data without an explicit confirmation.
            raise ValidationError(
                "Refusing to permanently erase a memory without confirmation.",
                hint="Pass --yes to confirm when running non-interactively.",
            )
        answer = input(f"Permanently erase {record.short_id} ({record.content!r})? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            ctx.console.info("Cancelled.")
            return EXIT_ERROR
    if args.permanent:
        memory.delete(memory_id)
        if ctx.console.json_mode:
            ctx.console.json({"id": memory_id, "deleted": True})
        else:
            ctx.console.success(f"Erased {record.short_id}.")
        return EXIT_OK
    archived = memory.forget(memory_id)
    if ctx.console.json_mode:
        ctx.console.json(archived.to_dict())
    else:
        ctx.console.success(
            f"Forgot {archived.short_id}. It is archived; undo with "
            f"`highhxpack restore {archived.short_id}`."
        )
    return EXIT_OK


def run_restore(args: argparse.Namespace, ctx: Context) -> int:
    memory = ctx.memory
    restored = memory.restore(memory.resolve_id(args.memory_id))
    if ctx.console.json_mode:
        ctx.console.json(restored.to_dict())
    else:
        ctx.console.success(f"Restored {restored.short_id}: {restored.content}")
    return EXIT_OK
