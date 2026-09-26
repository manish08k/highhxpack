"""``highhxpack import``."""

from __future__ import annotations

import argparse
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "import",
        parents=[common],
        help="import memories from a JSON export",
        description="Import a file created by `highhxpack export`. Every record is validated; "
        "invalid records are skipped and reported. With --user, all imported data is assigned "
        "to that user.",
    )
    parser.add_argument("file", help="file created by `highhxpack export`")
    parser.add_argument(
        "--replace", action="store_true", help="replace memories whose id already exists"
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, ctx: Context) -> int:
    report = ctx.memory.import_(
        args.file,
        user_id=ctx.explicit_user,
        on_existing="replace" if args.replace else "skip",
    )
    console = ctx.console
    if console.json_mode:
        console.json(report.to_dict())
        return EXIT_OK
    console.success(
        f"Imported {report.imported} memories ({report.replaced} replaced, "
        f"{report.skipped} skipped), {report.events} history events and "
        f"{report.relationships} relationships."
    )
    for error in report.errors[:20]:
        console.warn(error)
    if len(report.errors) > 20:
        console.warn(f"{console.sym('more')} {len(report.errors) - 20} more problems (use --json)")
    return EXIT_OK
