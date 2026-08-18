# Contributing

HEXIF welcomes focused changes to the cell-phenotyping pipeline and Spatial
Cytometry Workbench.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,web,dev]"
```

Before opening a pull request, run:

```bash
python scripts/check_publication_hygiene.py
ruff check .
ruff format --check .
pytest -q
python -m build
```

## Scientific and privacy requirements

- Never commit patient data, source images, derived cell tables, checkpoints,
  logs, credentials, or internal filesystem paths.
- Do not add invented metrics, synthetic scientific examples, or placeholder
  results. Deterministic in-memory fixtures are permitted only in unit tests.
- Every reported result must identify an immutable data split, configuration,
  code commit, environment, and artifact hashes as described in
  [`docs/reproducibility.md`](docs/reproducibility.md).
- Keep a single supported implementation for each operation. Remove obsolete
  paths in the same change that replaces them.
- Comments should explain constraints or intent, not narrate editing history.

## Pull requests

Use a narrow branch, include tests for behavioral changes, and explain any
change to scientific semantics. CI must pass. By contributing, you agree that
your contribution is licensed under Apache-2.0.

Report security or privacy concerns privately to the maintainer rather than in
a public issue.
