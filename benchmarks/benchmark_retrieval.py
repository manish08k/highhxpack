"""Measure retrieval latency for keyword, semantic and hybrid strategies.

Usage::

    python benchmarks/benchmark_retrieval.py --sizes 1000 5000 10000 --queries 50

Results depend on hardware; run the script on your own machine rather than
relying on published numbers.  A temporary database is used and deleted.
"""

from __future__ import annotations

import argparse
import random
import statistics
import tempfile
import time
from pathlib import Path

from highhxpack import Memory

TOPICS = [
    "python", "rust", "flutter", "postgres", "kubernetes", "react", "golang", "java",
    "docker", "terraform", "pytorch", "pandas", "redis", "kafka", "graphql", "swift",
]  # fmt: skip
TEMPLATES = [
    "I prefer {a} over {b} for {c} projects",
    "Our team deploys {a} services with {b}",
    "I am learning {a} to replace {b} in {c}",
    "The {a} migration broke the {b} pipeline last week",
    "My favorite {a} library for {c} is built on {b}",
    "We decided to use {a} and {b} for the {c} backend",
]
CONTEXTS = ["web", "data", "mobile", "infra", "ml", "analytics", "billing", "search"]


def make_text(rng: random.Random, i: int) -> str:
    template = rng.choice(TEMPLATES)
    a, b = rng.sample(TOPICS, 2)
    return template.format(a=a, b=b, c=rng.choice(CONTEXTS)) + f" (note {i})"


def populate(memory: Memory, size: int, rng: random.Random) -> float:
    start = time.perf_counter()
    for i in range(size):
        memory.remember("bench", make_text(rng, i), dedupe=False, detect_conflicts=False)
    return time.perf_counter() - start


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[index]


def bench_queries(memory: Memory, strategy: str, queries: list[str]) -> list[float]:
    timings = []
    for query in queries:
        start = time.perf_counter()
        memory.search(query, user_id="bench", strategy=strategy, limit=10)
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 5_000])
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"{'memories':>9} {'strategy':>9} {'p50 ms':>8} {'p95 ms':>8} {'mean ms':>8}")
    for size in args.sizes:
        with tempfile.TemporaryDirectory() as tmp, Memory(Path(tmp) / "bench.db") as memory:
            insert_seconds = populate(memory, size, rng)
            queries = [
                f"{rng.choice(TOPICS)} {rng.choice(CONTEXTS)} {rng.choice(TOPICS)}"
                for _ in range(args.queries)
            ]
            for strategy in ("keyword", "semantic", "hybrid"):
                timings = bench_queries(memory, strategy, queries)
                print(
                    f"{size:>9} {strategy:>9} {statistics.median(timings):>8.2f} "
                    f"{percentile(timings, 95):>8.2f} {statistics.fmean(timings):>8.2f}"
                )
            print(f"{'':>9} insert: {size / insert_seconds:,.0f} memories/s\n")


if __name__ == "__main__":
    main()
