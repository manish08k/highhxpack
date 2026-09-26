"""Lightweight, dependency-free entity recognition.

Recognizes two kinds of entities:

* ``technology`` - a curated list of languages, frameworks and tools, plus
  tokens that look technical (``c++``, ``f#``, ``node.js``, ``gpt-4``);
* ``name`` - capitalized word sequences that are not sentence-initial filler.

This is intentionally heuristic: it is used to label graph nodes and group
related memories, never to decide what is true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TECH_CANONICAL: dict[str, str] = {
    name.lower(): name
    for name in """
    Python Java JavaScript TypeScript Go Golang Rust C C++ C# Ruby PHP Swift Kotlin Scala
    Haskell Elixir Erlang Clojure Dart Flutter Lua Perl R Julia MATLAB SQL Bash Zig OCaml
    F# Objective-C Fortran COBOL Solidity HTML CSS React Vue Angular Svelte Next.js Node.js
    Deno Django Flask FastAPI Rails Spring Laravel Express PyTorch TensorFlow JAX Keras
    NumPy pandas scikit-learn Spark Hadoop Kafka Docker Kubernetes Terraform Ansible AWS
    GCP Azure Linux macOS Windows Ubuntu Postgres PostgreSQL MySQL SQLite MongoDB Redis
    Elasticsearch GraphQL Git GitHub GitLab Vim Neovim Emacs VSCode LangChain LlamaIndex
    Ollama OpenAI Anthropic Claude GPT-4 LLM LLMs AI ML
    """.split()
}
_TECHNICAL_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[+#]+|\.(?:js|ts|net)|-\d+(?:\.\d+)?)$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-']*")
_NOT_NAMES = frozenset(
    """
    i i'm i've i'd i'll my me we our you your he she they it this that the a an and but or so
    yes no ok okay hi hello hey thanks thank please also then when what which who why how
    monday tuesday wednesday thursday friday saturday sunday today yesterday tomorrow user
    january february march april may june july august september october november december
    """.split()
)


@dataclass(frozen=True, slots=True)
class Entity:
    name: str
    kind: str  # "technology" | "name"

    @property
    def key(self) -> str:
        return self.name.casefold()


def canonical_technology(token: str) -> str | None:
    """Return the canonical spelling of a known or technical-looking token."""
    stripped = token.strip(".,;:!?'\"()[]")
    known = _TECH_CANONICAL.get(stripped.lower())
    if known:
        return known
    if _TECHNICAL_TOKEN_RE.match(stripped):
        return stripped
    return None


def extract_entities(text: str) -> list[Entity]:
    """Return entities in order of first appearance, without duplicates."""
    found: list[Entity] = []
    seen: set[str] = set()

    def add(name: str, kind: str) -> None:
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            found.append(Entity(name, kind))

    tokens = _WORD_RE.findall(text)
    run: list[str] = []
    sentence_start = True

    def flush() -> None:
        if run:
            add(" ".join(run), "name")
            run.clear()

    for raw in tokens:
        token = raw.rstrip(".'")
        tech = canonical_technology(token)
        if tech is not None and (len(tech) > 1 or token.isupper()):
            flush()
            add(tech, "technology")
        elif (
            token[:1].isupper()
            and token.lower() not in _NOT_NAMES
            and not (sentence_start and token.lower() in _NOT_NAMES)
        ):
            run.append(token)
        else:
            flush()
        sentence_start = raw.endswith(".")
    flush()
    return found
