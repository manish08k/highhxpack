"""``highhxpack export``."""

from __future__ import annotations

import argparse
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "export",
        parents=[common],
        help="export memories to a JSON file (all users unless --user is given)",
        description="Write memories, history, relationships and conflicts to a JSON file "
        "readable only by you.",
    )
    parser.add_argument("file", help="destination file, e.g. memories.json")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite an existing file")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, ctx: Context) -> int:
    report = ctx.memory.export(args.file, user_id=ctx.explicit_user, overwrite=args.force)
    if ctx.console.json_mode:
        ctx.console.json(report.to_dict())
    else:
        ctx.console.success(
            f"Exported {report.memories} memories, {report.events} history events, "
            f"{report.relationships} relationships and {report.conflicts} conflicts "
            f"to {report.path}"
        )
    return EXIT_OK
