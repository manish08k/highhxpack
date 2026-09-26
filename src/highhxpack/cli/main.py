"""The ``highhxpack`` command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from collections.abc import Sequence

from highhxpack.__version__ import __version__
from highhxpack.cli.commands import (
    consolidate,
    export,
    forget,
    import_,
    init,
    inspect,
    memories,
    recall,
    remember,
    stats,
)
from highhxpack.cli.context import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_USAGE,
    Context,
    Handler,
)
from highhxpack.cli.display import Console
from highhxpack.exceptions import (
    ConflictNotFoundError,
    HighHXPackError,
    MemoryNotFoundError,
    ValidationError,
)
from highhxpack.utils.logging import configure_cli_logging

_EPILOG = """\
Configuration precedence: defaults < config file < HIGHHXPACK_* environment
variables < command-line options. Run `highhxpack config` to see the result.

Exit codes: 0 success, 1 error, 2 invalid usage or input, 3 not found,
130 interrupted.

Data stays on this machine unless you configure a remote provider.
"""

_MODULES = (init, remember, recall, memories, inspect, forget, stats, export, import_, consolidate)


#: Defaults of the global options.  The options themselves use SUPPRESS defaults
#: so they can be given before *or* after the command without one position
#: overwriting the other.
_GLOBAL_DEFAULTS: dict[str, object] = {
    "db": None,
    "config": None,
    "user": None,
    "json": False,
    "no_color": False,
    "verbose": 0,
}


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    group = common.add_argument_group("global options")
    group.add_argument("--db", metavar="PATH", help="database file (default: <home>/memory.db)")
    group.add_argument("--config", metavar="PATH", help="configuration file")
    group.add_argument("-u", "--user", metavar="USER_ID", help="user id (default: from config)")
    group.add_argument("--json", action="store_true", help="machine-readable JSON output")
    group.add_argument("--no-color", action="store_true", help="disable colored output")
    group.add_argument("-v", "--verbose", action="count", help="log more details (repeatable)")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="highhxpack",
        description="HighHXPack: local-first memory for AI applications and developer tools.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    parser.add_argument("-V", "--version", action="version", version=f"highhxpack {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for module in _MODULES:
        module.register(subparsers, common)
    return parser


def _tolerate_unencodable_output() -> None:
    """Never crash on characters the terminal encoding cannot represent."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _tolerate_unencodable_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, default in _GLOBAL_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE
    console = Console(json_mode=args.json, color=False if args.no_color else None)
    configure_cli_logging(args.verbose)
    ctx = Context(args, console)
    try:
        return handler(args, ctx)
    except (MemoryNotFoundError, ConflictNotFoundError) as exc:
        console.error(exc)
        return EXIT_NOT_FOUND
    except ValidationError as exc:
        console.error(exc)
        return EXIT_USAGE
    except HighHXPackError as exc:
        console.error(exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        console.error("interrupted")
        return EXIT_INTERRUPTED
    except BrokenPipeError:  # e.g. `highhxpack memories | head`
        sys.stderr.close()
        return EXIT_OK
    except Exception as exc:  # a bug: explain briefly instead of dumping a traceback
        console.error(
            HighHXPackError(
                f"Unexpected internal error: {type(exc).__name__}: {exc}",
                hint="Please report this at the project's issue tracker; run with -v to "
                "include the full traceback.",
            )
        )
        if args.verbose:
            traceback.print_exc()
        return EXIT_ERROR
    finally:
        try:
            ctx.close()
        except HighHXPackError as exc:  # pragma: no cover - close is best effort
            console.error(exc)


def entrypoint() -> None:  # pragma: no cover - thin wrapper used by `python -m`
    sys.exit(main())
