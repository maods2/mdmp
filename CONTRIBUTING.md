# Contributing to MDMP

Thanks for contributing. This document covers **development** install, tests,
linting, and releases. Library users should start with [`README.md`](README.md).

## Development install

Prerequisites: Python 3.8+, Git. Optional: [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/maods2/mdmp.git
cd mdmp
```

**Using `uv` (recommended):**

```bash
uv venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
uv sync
```

**Using `venv`:**

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows
pip install -e ".[dev]"
```

Or install editable with only runtime deps:

```bash
pip install -e .
pip install -e ".[dev]"     # tests, ruff, black, bump-my-version, …
```

Optional user extras still apply: `pip install -e ".[graphviz]"`, `".[umap]"`.

## Running tests

```bash
pytest tests/ -v
pytest tests/test_scoring.py -v
pytest tests/ -v --cov=mdmp --cov-report=term-missing
```

If coverage options fail because `pytest-cov` is missing:

```bash
pip install pytest-cov
# or
pytest tests/ -v -o addopts=
```

Main test modules: `test_dlm.py`, `test_mdm.py`, `test_scoring.py`,
`test_structure.py`, `test_utils.py`, `test_parallel.py`, `test_progress.py`,
`test_plotting.py`, plus group-analysis and anomaly tests under `tests/`.

All tests should pass before opening a pull request.

## Code quality

**Ruff** (lint) and **Black** (format) are configured in `pyproject.toml`.

```bash
ruff check .
ruff check --fix .
black mdmp/ tests/
black --check mdmp/ tests/
```

Ruff selects `E`, `W`, `F`, `I`, `B`, `C4`, `UP`; ignores `E501` and `B008`
(line length is handled by Black).

### Pre-commit (optional)

```bash
pre-commit install
pre-commit run --all-files
```

### Typical pre-commit workflow

```bash
ruff check .
ruff check --fix .
black mdmp/ tests/
pytest tests/ -v
```

## Changelog

Human-facing release notes live in [`CHANGELOG.md`](CHANGELOG.md). During
development, add bullets under `## [Unreleased]` (Added, Changed, Fixed, …).
On release, rename that section to `## [x.y.z] - YYYY-MM-DD` and refresh the
compare links at the bottom of the file. See also [`AGENTS.md`](AGENTS.md).

## Versioning and releases

The installed version is `mdmp.__version__` ([`mdmp/_version.py`](mdmp/_version.py);
`pyproject.toml` reads it via setuptools dynamic metadata). README copies (git
tag install line and the mdmp BibTeX block) stay aligned via **bump-my-version**.

```bash
# preview
uvx bump-my-version bump patch --dry-run --verbose --allow-dirty

# apply (use minor / major when appropriate)
uvx bump-my-version bump patch --allow-dirty
```

This updates `mdmp/_version.py`, `[tool.bumpversion].current_version` in
`pyproject.toml`, and the README patterns under `[tool.bumpversion]`. If you
use a lockfile locally, run `uv lock` afterward.

### Release checklist

1. Merge to `main` and move `[Unreleased]` items in `CHANGELOG.md` into a dated
   `## [x.y.z]` section.
2. Run `bump-my-version bump`, then `uv lock` if applicable.
3. Commit, create annotated tag `vx.y.z`, push the tag, and publish to PyPI.
