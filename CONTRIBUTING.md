# Contributing to synapse-orchestrator

Thanks for your interest in contributing.

## Development setup

- Python: `cd python && pip install -e .[dev]`
- TypeScript: `cd typescript && npm install`

## Testing requirements

Please run tests for the language package(s) you touched before opening a PR.

- Python: `cd python && pytest`
- TypeScript: `cd typescript && npm run build`

## Core architecture note

The DAG and dependency analyzer is the core of this project. Changes around placeholder parsing, dependency extraction, topological planning, and stage construction must be tested carefully with both positive and negative cases.

At minimum, add or update tests for:

- dependency edge extraction from `$results.*` placeholders
- cycle detection and invalid dependency handling
- planner stage output determinism
- executor behavior for failures and downstream skip propagation

## Pull request checklist

- Keep changes focused and scoped.
- Add or update tests for behavior changes.
- Update docs and `CHANGELOG.md` when relevant.
- Fill out the PR template completely.
