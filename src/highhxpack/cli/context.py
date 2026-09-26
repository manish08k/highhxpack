"""Shared state for CLI commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

from highhxpack.cli.display import Console
from highhxpack.core.config import Config
from highhxpack.utils.validation import validate_user_id

if TYPE_CHECKING:
    from highhxpack.core.memory import Memory

#: Exit codes (documented in ``highhxpack --help``).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_INTERRUPTED = 130

Handler = Callable[[argparse.Namespace, "Context"], int]


class Context:
    """Lazily opens the memory store so commands like ``--help`` stay fast."""

    def __init__(self, args: argparse.Namespace, console: Console):
        self.args = args
        self.console = console
        self._config: Config | None = None
        self._memory: Memory | None = None

    @property
    def config(self) -> Config:
        if self._config is None:
            user = getattr(self.args, "user", None)
            if user is not None:
                validate_user_id(user)  # an invalid --user is an input error (exit 2)
            self._config = Config.load(
                config_file=getattr(self.args, "config", None),
                storage_path=getattr(self.args, "db", None),
                default_user=getattr(self.args, "user", None),
            )
        return self._config

    @property
    def memory(self) -> Memory:
        if self._memory is None:
            from highhxpack.core.memory import Memory

            self._memory = Memory(config=self.config)
        return self._memory

    @property
    def user(self) -> str:
        """The user to act on: ``--user``, else the configured default user."""
        return self.config.default_user

    @property
    def explicit_user(self) -> str | None:
        """``--user`` if it was given on the command line, otherwise ``None``."""
        user = getattr(self.args, "user", None)
        return str(user) if user is not None else None

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()
            self._memory = None
