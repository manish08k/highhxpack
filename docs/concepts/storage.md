# Storage

## Location

| Platform | Default database |
|---|---|
| Linux / BSD | `$XDG_DATA_HOME/highhxpack/memory.db` or `~/.local/share/highhxpack/memory.db` |
| macOS | `~/Library/Application Support/highhxpack/memory.db` |
| Windows | `%LOCALAPPDATA%\highhxpack\memory.db` |

`HIGHHXPACK_HOME` moves the whole data directory; `storage_path` (argument, config
file, `HIGHHXPACK_STORAGE_PATH` or `--db`) sets the database file directly.
`":memory:"` creates a temporary in-memory database.

## Requirements

SQLite 3.32 or newer with JSON functions (JSON1). Python 3.11+ builds from python.org,
Homebrew, conda and current Linux distributions meet this; other builds are refused at
open time with an explanatory `StorageError`.

## Guarantees

- **Transactions**: every multi-step change runs in one `BEGIN IMMEDIATE`
  transaction; failures roll back completely.
- **WAL mode** lets readers proceed while a writer commits; `busy_timeout` (30 s)
  makes concurrent writers wait instead of failing. One `Memory` instance may be
  shared across threads; several processes may open the same file.
- **Migrations** are versioned with `PRAGMA user_version`, applied in transactions
  and re-checked under the write lock, so concurrent first starts are safe. A database
  written by a newer release is refused with a clear error instead of being modified.
- **Clean shutdown**: `close()` checkpoints the WAL into the main file.
- **Durability**: `synchronous = NORMAL` in WAL mode. A crash or power loss cannot
  corrupt the database, but the most recent transactions may be lost on power loss.
- **Processes**: create `Memory` *after* forking; a SQLite connection must not be
  shared between a parent and a forked child (e.g. `multiprocessing` with `fork`).
- **Permissions** (POSIX): directory `0700`, database `0600`, created that way from
  the start.
- **Corruption**: a file that is not a SQLite database is reported as such;
  `SQLiteStorage.check_integrity()` runs SQLite's integrity check.

## Schema (version 1)

| Table | Contents |
|---|---|
| `memories` | memory records, content hash, claim slot, BM25 document length |
| `memory_terms` | keyword postings (term, memory, term frequency) |
| `memory_vectors` | one float32 vector per memory, with model name and dimension |
| `memory_events` | append-only history |
| `relationships` | graph edges (unique per user/source/relation/target, case-insensitive) |
| `conflicts` | detected contradictions and their resolution |
| `meta` | database metadata |

Derived rows (terms, vectors, events, edges, conflicts) are removed with their memory
via foreign-key cascades.

## Custom backends

Implement `highhxpack.StorageBackend` and pass it: `Memory(storage=MyBackend())`.
The engine computes all derived data (`IndexData`); a backend only persists and
queries it. `SQLiteStorage` is the reference implementation, and
`MemoryFilter.matches()` defines the filter semantics a backend must reproduce.
