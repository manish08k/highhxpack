# Contributing to HighHXPack

Thanks for your interest! Bug reports, documentation fixes and pull requests are all
welcome.

## Development setup

```bash
git clone https://github.com/highhxpack/highhxpack
cd highhxpack
uv sync                 # or: python -m venv .venv && pip install -e . --group dev
uv run pre-commit install
```

## Checks

Run these before opening a pull request; CI runs the same commands.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov                 # all tests; fails below 90% coverage
uv run pytest -m "not slow"         # quicker loop
uv run python -m build && uv run twine check dist/*   # packaging (optional)
```

## Guidelines

- **Tests first.** Every feature needs tests; every bug fix needs a regression test
  that fails without the fix.
- **No new required dependencies.** The core must keep working with the standard
  library only. Optional integrations go behind an extra and a lazy import that raises
  `ProviderNotAvailableError` with install instructions.
- **Public API** is what `highhxpack/__init__.py` exports. Changing it needs a
  CHANGELOG entry; breaking changes need a deprecation path where possible.
- **Errors** raise a `HighHXPackError` subclass with a message, and ideally a reason
  and a hint.
- **No magic numbers** in ranking or policies: put tunables in a config dataclass.
- **Schema changes** are new migrations in `storage/migrations.py`; never edit a
  released migration.
- **Privacy**: never log memory content or secrets at INFO level or above, and never
  send data to a network service unless the user configured it.
- Keep documentation in `docs/` and `README.md` in sync with the code.

## Releasing (maintainers)

1. Update `src/highhxpack/__version__.py` and move `Unreleased` entries in
   `CHANGELOG.md` under the new version.
2. Merge to `main`, tag `vX.Y.Z` and publish a GitHub release. The `publish` workflow
   builds, checks and uploads to PyPI through trusted publishing.

By contributing you agree that your contributions are licensed under the MIT License
and that you will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
