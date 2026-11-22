# Contributing conventions

## Commits
- Use conventional-commit subjects: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Write the subject in the imperative and describe the change only.
- Do not reference tooling, task ids, plan steps or personal context.
- Do not add co-author or tool attribution trailers.

## Pull requests
- Use a conventional-commit title. PRs are squash-merged, so the title and body become the commit on `main`.
- Write a concise English body with Summary, Design decisions (when relevant) and Testing. Tables are fine where they help.
- Do not paste tool output, logs, review reports or evidence dumps.

## Branches
- Use short descriptive names such as `feat/rate-scenarios` or `docs/model-comparison`.

## Data
- Use only the public ECB data loaded by `data.py`, or clearly labelled synthetic data.
- Do not commit personal information.

## Before opening a PR
```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
