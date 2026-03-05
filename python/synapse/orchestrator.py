"""
Orchestrator — top-level entry point for Synapse.

Typical usage
-------------
::

    from synapse import Orchestrator, ToolCall

    async def main():
        orch = Orchestrator(tools={
            "fetch_user":   fetch_user,
            "fetch_orders": fetch_orders,
            "send_email":   send_email,
        })

        report = await orch.run([
            ToolCall(id="u",  name="fetch_user",   inputs={"user_id": 42}),
            ToolCall(id="o",  name="fetch_orders",  inputs={"user_id": 42}),
            ToolCall(id="em", name="send_email",
                     inputs={"to": "$results.u.email",
                              "subject": "Your orders",
                              "body": "$results.o"}),
        ])

        print(report)

The Orchestrator wires together:
  DependencyAnalyzer → Planner → Executor
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

from .dependency_analyzer import DependencyAnalyzer, DependencyGraph, ToolCall
from .executor import AsyncToolFn, CallResult, ExecutionReport, Executor
from .planner import ExecutionPlan, Planner


class Orchestrator:
    """
    High-level coordinator.  Accepts a tool registry and optional tuning
    parameters, then accepts lists of ToolCall objects, analyses their
    dependencies, plans execution, and runs them with maximum parallelism.

    Parameters
    ----------
    tools:
        Mapping from tool name → async callable.
    max_concurrency:
        Hard cap on simultaneous in-flight calls.
    default_timeout:
        Per-call timeout in seconds (applied when ToolCall.timeout is None).
    default_retries:
        Retry count applied when ToolCall.retries is 0.
    on_call_start:
        Optional hook called immediately before each call is dispatched.
    on_call_end:
        Optional hook called as soon as each call finishes.
    """

    def __init__(
        self,
        tools: dict[str, AsyncToolFn],
        *,
        max_concurrency: int = 16,
        default_timeout: float = 30.0,
        default_retries: int = 0,
        on_call_start: Callable[[ToolCall], None] | None = None,
        on_call_end: Callable[[CallResult], None] | None = None,
    ) -> None:
        self.tools = tools
        self._analyzer = DependencyAnalyzer()
        self._planner = Planner()
        self._executor = Executor(
            tools=tools,
            max_concurrency=max_concurrency,
            default_timeout=default_timeout,
            default_retries=default_retries,
            on_call_start=on_call_start,
            on_call_end=on_call_end,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def analyze(self, calls: list[ToolCall]) -> DependencyGraph:
        """Return the dependency graph without executing anything."""
        return self._analyzer.analyze(calls)

    def plan(self, calls: list[ToolCall]) -> ExecutionPlan:
        """Return the execution plan without executing anything."""
        graph = self._analyzer.analyze(calls)
        return self._planner.plan(graph)

    async def run(self, calls: list[ToolCall]) -> ExecutionReport:
        """
        Analyse, plan, and execute the given tool calls.

        Returns an ExecutionReport with per-call results, timing, and
        a speedup estimate (sum of individual durations / wall-clock time).
        """
        plan = self.plan(calls)
        return await self._executor.execute(plan)

    async def run_raw(
        self,
        raw_calls: list[dict[str, Any]],
    ) -> ExecutionReport:
        """
        Convenience wrapper that accepts plain dicts instead of ToolCall objects.

        Each dict must have at minimum ``id`` and ``name`` keys.  Other keys
        (``inputs``, ``depends_on``, ``timeout``, ``retries``, ``metadata``)
        are optional.
        """
        calls = [ToolCall(**c) for c in raw_calls]
        return await self.run(calls)
