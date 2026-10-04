# CLAUDE.md

Guidance for AI coding agents (and human contributors) working in this repository.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). `uv sync` creates the environment
(`uv sync --group docs` adds the Sphinx toolchain).

## Checks to run before considering a change done

```sh
uv run ruff format .        # formatting
uv run ruff check .         # lint (ruff's default rule set)
uv run ty check             # type checking
uv run vulture              # dead code (min_confidence = 80, see pyproject.toml)
uv run pytest               # tests (runs in parallel with xdist)
uv run sphinx-build -W docs docs/_build   # docs, if docstrings or docs/ changed
```

CI runs the same checks. `pre-commit install` runs ruff on each commit.

### Tests must stay fast

The whole suite should finish in seconds. Slow tests are a bug, not a cost of doing
business: use small problem sizes and a capped `maxiter`, and assert the same
properties. `conftest.py` pins BLAS to one thread because the test matrices are tiny and
multithreaded BLAS makes them dramatically slower. Warnings are errors
(`filterwarnings = ["error", ...]`); fix the cause, or add a narrowly-scoped,
commented ignore for third-party warnings.

### Dead code -- vulture

Anything vulture reports at 80%+ confidence should be removed. Public methods and
functions (the library API) intentionally have no in-repo callers and are not
reported below that threshold; do not delete them just because they are unused here.

## Conventions

- Docstrings are NumPy style (they feed Sphinx autodoc).
- Missing data is represented by `NaN`; solvers must ignore it, not impute it.
- Public API lives in `tensorpack/__init__.py` (`__all__`).
