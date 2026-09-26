"""Terminal output helpers: colors, tables, JSON and error rendering.

Colors are used only when writing to a terminal and neither ``NO_COLOR`` nor
``--no-color`` is set.  All user data printed here has already had control
characters removed by input validation, so it cannot inject terminal escapes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any, TextIO

from highhxpack.exceptions import HighHXPackError
from highhxpack.utils.timestamps import ensure_utc

#: Symbols used in human output, with ASCII fallbacks for limited encodings.
_SYMBOLS = {
    "check": ("✓", "OK"),
    "arrow": ("→", "->"),
    "dash": ("─", "-"),
    "more": ("…", "..."),
    "warn": ("⚠", "!"),
    "em": ("—", "-"),
}

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}


class Console:
    """Writes human or JSON output to stdout and diagnostics to stderr."""

    def __init__(
        self,
        *,
        json_mode: bool = False,
        color: bool | None = None,
        out: TextIO | None = None,
        err: TextIO | None = None,
    ):
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.json_mode = json_mode
        if color is None:
            color = (
                self.out.isatty()
                and not os.environ.get("NO_COLOR")
                and os.environ.get("TERM") != "dumb"
            )
        self.color = color
        # Streams without an encoding (e.g. StringIO) accept any text.
        encoding = getattr(self.out, "encoding", None) or "utf-8"
        try:
            "".join(s for s, _ in _SYMBOLS.values()).encode(encoding)
            self.unicode = True
        except (UnicodeEncodeError, LookupError):
            self.unicode = False

    def sym(self, name: str) -> str:
        """A display symbol, or its ASCII fallback when the output cannot encode it."""
        fancy, plain = _SYMBOLS[name]
        return fancy if self.unicode else plain

    def style(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        codes = ";".join(_CODES[s] for s in styles)
        return f"\x1b[{codes}m{text}\x1b[0m"

    def print(self, text: str = "") -> None:
        print(text, file=self.out)

    def info(self, text: str) -> None:
        """A status line; suppressed in JSON mode so stdout stays machine-readable."""
        if not self.json_mode:
            print(text, file=self.out)

    def success(self, text: str) -> None:
        self.info(self.style(self.sym("check") + " ", "green") + text if self.color else text)

    def warn(self, text: str) -> None:
        print(self.style("warning: ", "yellow", "bold") + text, file=self.err)

    def json(self, data: Any) -> None:
        # ASCII-only JSON is valid on every terminal/pipe encoding (e.g. cp1252).
        print(json.dumps(data, indent=2, ensure_ascii=True), file=self.out)

    def error(self, exc: HighHXPackError | str) -> None:
        if isinstance(exc, HighHXPackError):
            print(self.style("error: ", "red", "bold") + exc.message, file=self.err)
            if exc.reason:
                print(self.style("  reason: ", "dim") + exc.reason, file=self.err)
            if exc.hint:
                print(self.style("  hint: ", "cyan") + exc.hint, file=self.err)
        else:
            print(self.style("error: ", "red", "bold") + exc, file=self.err)

    @property
    def width(self) -> int:
        return max(shutil.get_terminal_size((100, 24)).columns, 40)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]], *, flex: int) -> None:
        """Print an aligned table; column ``flex`` absorbs the remaining width."""
        if not rows:
            return
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        fixed = sum(w for i, w in enumerate(widths) if i != flex) + 2 * (len(headers) - 1)
        widths[flex] = max(min(widths[flex], self.width - fixed), 10)
        header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
        self.print(self.style(header, "bold"))
        for row in rows:
            cells = [
                truncate(cell, widths[i], self.sym("more")).ljust(widths[i])
                for i, cell in enumerate(row)
            ]
            self.print("  ".join(cells).rstrip())


def truncate(text: str, width: int, marker: str = "…") -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: max(width - len(marker), 0)] + marker


def format_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    return ensure_utc(value).strftime("%Y-%m-%d %H:%M")


def format_bytes(size: int) -> str:
    amount = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{size} B"  # pragma: no cover
