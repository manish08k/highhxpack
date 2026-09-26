# Command-line interface

```text
highhxpack [-h] [-V] COMMAND [options]
```

Options accepted by every command: `--db PATH`, `--config PATH`, `-u/--user USER_ID`,
`--json`, `--no-color`, `-v/--verbose` (repeatable). The user defaults to
`default_user` from the configuration (`"default"`). Memory ids may be abbreviated to
a unique prefix of at least four characters.

| Command | Purpose |
|---|---|
| `init [--force]` | create data directory, config template and database (at `--config PATH` if given) |
| `config` | show the effective configuration (secrets masked) |
| `remember TEXT... [-t TYPE] [--importance F] [--confidence F] [--source S] [--ttl 30d] [--meta K=V]... [--extract] [--no-dedupe]` | store a memory; `TEXT` = `-` reads stdin; `--extract` stores only extracted statements |
| `recall QUERY... [-n N] [-t TYPE] [--min-score F] [--strategy S] [--explain]` | ranked memories of the current user |
| `search QUERY... [...recall options] [--status S] [--include-expired]` | search all users (or `--user`) without changing access stats |
| `memories` / `list [-t TYPE] [--status S] [-n N] [--offset N] [--order O] [--all-users] [--include-expired]` | list memories |
| `inspect [USER_ID]` | user summary and knowledge graph |
| `history` / `show ID` | a memory and its change history |
| `forget ID [--permanent] [-y]` | archive a memory, or erase it with `--permanent` (asks for confirmation; non-interactive use requires `--yes`) |
| `restore ID` | re-activate an archived/superseded memory |
| `conflicts [--all]` | list open (or all) conflicts |
| `resolve CONFLICT_ID --keep ID` | choose which memory is current |
| `consolidate [--dry-run] [--summarize]` | merge duplicates, archive expired, apply conflict policy |
| `stats` | statistics (all users unless `--user`) |
| `export FILE [-f]` | JSON export (all users unless `--user`), mode 0600 |
| `import FILE [--replace]` | import an export (`--user` re-assigns ownership) |

## Output and exit codes

`--json` prints machine-readable JSON on stdout (ASCII-escaped, valid in any
encoding); diagnostics go to stderr. Colors are disabled when stdout is not a terminal
or `NO_COLOR` is set.

| Code | Meaning |
|---|---|
| 0 | success (including "no results") |
| 1 | error (storage, provider, configuration, cancelled confirmation) |
| 2 | invalid usage or input |
| 3 | memory or conflict not found |
| 130 | interrupted |

Errors are printed as `error: …`, `reason: …`, `hint: …` without tracebacks. An
unexpected internal error is reported the same way (exit code 1); add `-v` to include
the traceback in a bug report.
