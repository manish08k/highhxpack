"""Use a local LLM (Ollama) for extraction and summaries.

Requires a running Ollama server with a model pulled::

    ollama pull llama3.2
    python examples/local_llm.py

If Ollama is not reachable, the example falls back to the built-in rule-based
extractor so it still runs.  Data is sent only to the local Ollama server.
"""

import tempfile
from pathlib import Path

from highhxpack import Config, HighHXPackError, Memory
from highhxpack.extraction import LLMExtractor, RuleBasedExtractor
from highhxpack.providers import OllamaProvider

TEXT = (
    "I've been using Python for machine learning for three years, but I prefer C++ "
    "for competitive programming. I'm allergic to peanuts. What time is it?"
)

with tempfile.TemporaryDirectory() as tmp:
    llm = OllamaProvider("llama3.2", timeout=30)
    config = Config(storage_path=Path(tmp) / "llm.db", llm_provider="ollama")
    with Memory(config=config, llm=llm) as memory:
        try:
            results = memory.ingest("user_1", TEXT, extractor=LLMExtractor(llm))
            print(f"Extracted with {llm.name}:")
        except HighHXPackError as exc:
            print(f"Ollama unavailable: {exc.reason or exc.message}")
            print("Falling back to the rule-based extractor.\n")
            results = memory.ingest("user_1", TEXT, extractor=RuleBasedExtractor())
        for result in results:
            m = result.memory
            print(f"  [{m.memory_type}] {m.content} (confidence {m.confidence:.2f})")

        for extra in ("I use Python at work", "Python is my favourite language"):
            memory.remember("user_1", extra)
        try:
            report = memory.consolidate("user_1", summarize=True)
            for summary_id in report.summaries_created:
                print("\nSummary:", memory.get(summary_id).content)
        except HighHXPackError as exc:
            print(f"\nSummarization skipped: {exc.message}")
