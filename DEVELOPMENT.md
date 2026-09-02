# Lab development and publication workflow

HEXIF is maintained for authorized Gentles Lab research, internal
collaboration, and preparation of software and artifacts supporting journal
publications. It is not operated as an open community contribution project.
Repository access does not grant access to research data or permission to use,
share, or publish those data.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,web,dev]"
```

Before merging an internal change, run:

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

## Internal review and publication

Use a narrow branch, include tests for behavioral changes, and state any change
to scientific semantics. A lab-authorized reviewer must approve scientific,
data-handling, or result-bearing changes, and CI must pass before merge.

Manuscript figures, tables, metrics, checkpoints, or demonstration bundles may
be added only after the independent release review in
[`docs/reproducibility.md`](docs/reproducibility.md). Authorship, data-use, and
release decisions follow lab, institutional, protocol, and journal requirements;
they are not determined by repository activity.

Report security, privacy, or data-governance concerns through approved internal
lab or Stanford channels, never through a public issue containing sensitive
information.
