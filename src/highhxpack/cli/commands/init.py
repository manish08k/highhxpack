"""``highhxpack init`` and ``highhxpack config``."""

from __future__ import annotations

import argparse
import os
from typing import Any

from highhxpack.cli.context import EXIT_OK, Context
from highhxpack.core.config import CONFIG_FILENAME, write_config_template
from highhxpack.utils.validation import validate_path


def register(subparsers: Any, common: argparse.ArgumentParser) -> None:
    init = subparsers.add_parser(
        "init",
        parents=[common],
        help="create the data directory, configuration file and database",
        description="Create the data directory, a commented configuration file and the "
        "database. Safe to run more than once.",
    )
    init.add_argument("--force", action="store_true", help="overwrite an existing config file")
    init.set_defaults(handler=run_init)

    config = subparsers.add_parser(
        "config",
        parents=[common],
        help="show the resolved configuration",
        description="Show the effective configuration after applying the config file, "
        "environment variables and command-line options. Secrets are masked.",
    )
    config.set_defaults(handler=run_config)


def run_init(args: argparse.Namespace, ctx: Context) -> int:
    created_config = False
    # A config file requested with --config / $HIGHHXPACK_CONFIG may not exist yet:
    # creating it is exactly what `init` is for.
    requested = getattr(args, "config", None) or os.environ.get("HIGHHXPACK_CONFIG")
    if requested:
        target = validate_path(requested, field="config")
        if not target.exists() or args.force:
            write_config_template(target, overwrite=args.force)
            created_config = True
    config = ctx.config
    config_path = config.config_file or config.home / CONFIG_FILENAME
    if not requested and (not config_path.exists() or args.force):
        write_config_template(config_path, overwrite=args.force)
        created_config = True
    memory = ctx.memory  # creates the database and applies migrations
    data = {
        "home": str(config.home),
        "config_file": str(config_path),
        "config_created": created_config,
        "storage_path": memory.location,
        "schema_version": memory.storage.schema_version,
        "default_user": config.default_user,
    }
    if ctx.console.json_mode:
        ctx.console.json(data)
        return EXIT_OK
    console = ctx.console
    console.success("HighHXPack is ready.")
    console.print(f"  config:   {config_path}" + ("" if created_config else " (existing, kept)"))
    console.print(f"  database: {memory.location}")
    console.print(f"  user:     {config.default_user}")
    console.print()
    console.print('Try:  highhxpack remember "I prefer Python for AI development"')
    console.print('      highhxpack recall "programming language"')
    return EXIT_OK


def run_config(args: argparse.Namespace, ctx: Context) -> int:
    data = ctx.config.to_dict(redact=True)
    if ctx.console.json_mode:
        ctx.console.json(data)
        return EXIT_OK
    scoring = data.pop("scoring")
    for key, value in data.items():
        ctx.console.print(f"{key:<18} {value if value is not None else '-'}")
    ctx.console.print(ctx.console.style("scoring", "bold"))
    for key, value in scoring.items():
        ctx.console.print(f"  {key:<24} {value}")
    return EXIT_OK
