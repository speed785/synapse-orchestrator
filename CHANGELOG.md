# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-05

### Added

- DependencyAnalyzer for parsing `$results.*` references and building dependency edges.
- Planner based on Kahn's algorithm for deterministic DAG stage planning.
- Parallel Executor implementations using `asyncio` (Python) and `Promise.all` (TypeScript).
- OpenAI, Anthropic, and LangChain integrations.
- Placeholder resolution for nested `$results.<id>.<path>` values.
- Error propagation with downstream skip behavior for failed dependencies.
- Observability support with execution reporting and logging hooks.
- TypeScript parity with the Python API and execution model.
- 100% test coverage.
