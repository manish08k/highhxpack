"""Measure write-path throughput: plain inserts, deduplicated inserts, ingest,
consolidation and export/import.

Usage::

    python benchmarks/benchmark_memory.py --size 2000

Results depend on hardware; run the script on your own machine.
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from highhxpack import Memory

SUBJECTS = ["Python", "Rust", "Go", "Kotlin", "Postgres", "Redis", "React", "Flutter"]
PHRASES = [
    "I use {x} at work",
    "I prefer {x} for side projects",
    "I am learning {x} this year",
    "My team migrated to {x}",
    "I dislike debugging {x} in production",
]


def timed(label: str, count: int, action: Callable[[], object]) -> None:
    start = time.perf_counter()
    action()
    seconds = time.perf_counter() - start
    rate = f"{count / seconds:,.0f}/s" if seconds > 0 else "n/a"
    print(f"{label:<38} {seconds * 1000:>10.1f} ms  {rate:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    texts = [
        rng.choice(PHRASES).format(x=rng.choice(SUBJECTS)) + f" (item {i})"
        for i in range(args.size)
    ]
    repeated = [rng.choice(PHRASES).format(x=rng.choice(SUBJECTS)) for _ in range(args.size)]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with Memory(root / "plain.db") as memory:
            timed(
                f"remember x{args.size} (no dedupe)",
                args.size,
                lambda: [
                    memory.remember("u", t, dedupe=False, detect_conflicts=False) for t in texts
                ],
            )
        with Memory(root / "dedupe.db") as memory:
            timed(
                f"remember x{args.size} (dedupe+conflicts)",
                args.size,
                lambda: [memory.remember("u", t) for t in repeated],
            )
            print(f"{'  -> distinct memories kept':<38} {memory.count('u'):>10}")
        with Memory(root / "ingest.db") as memory:
            chunk = ". ".join(repeated[:200])
            timed("ingest 200 sentences", 200, lambda: memory.ingest("u", chunk))
        with Memory(root / "consolidate.db") as memory:
            for t in repeated:
                memory.remember("u", t, dedupe=False, detect_conflicts=False)
            timed(f"consolidate {args.size} memories", args.size, lambda: memory.consolidate("u"))
            export_path = root / "export.json"
            timed("export", args.size, lambda: memory.export(export_path))
        with Memory(root / "import.db") as memory:
            timed("import", args.size, lambda: memory.import_(export_path))


if __name__ == "__main__":
    main()
