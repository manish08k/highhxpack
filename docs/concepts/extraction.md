# Extraction

`memory.ingest(user_id, text)` extracts memorable statements from free text and
stores each one through the normal write path (deduplication, conflicts, history).

## Rule-based extractor (default)

Offline and deterministic. The text is split into sentences and independent clauses
("…, but I prefer C++…" becomes two clauses). Each clause is kept only if it matches a
pattern:

| Type | Examples |
|---|---|
| `preference` | I prefer / like / love / dislike X · My favorite X is Y · I'm interested in X |
| `fact` | I use X · I work at X · I live in X · I'm a X · My name is X · I'm allergic to X · My X is Y |
| `goal` | I want to … · I plan to … · My goal is to … |
| `decision` | We decided to … · We'll go with X |
| `relationship` | My sister Anna … · My manager's name is Bob |
| `event` | I went / started / moved … · I have a meeting tomorrow · anything first-person with a date |

Questions, greetings, fillers and fragments shorter than three words are dropped.
Recognized relationships become graph edges (`user --prefers--> C++`, with
`{"context": "competitive programming"}` metadata).

The patterns are English-only heuristics: they favor precision over recall.

## LLM extractor

Set `extractor = "llm"` and an `llm_provider`, or pass `extractor=LLMExtractor(llm)`.
The model is asked for JSON; every item is validated (content, type, confidence,
importance, relations) and invalid items are dropped. Output that is not valid JSON
raises `ExtractionError` — nothing is invented.

## Entities

`highhxpack.extraction.extract_entities(text)` finds technologies (a curated list plus
technical-looking tokens such as `c++`, `node.js`, `gpt-4`) and capitalized names.
Entities label graph targets and group memories for summaries.
