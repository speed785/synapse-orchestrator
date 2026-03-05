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

from datetime import datetime
from typing import Any, Callable, Coroutine

from .dependency_analyzer import DependencyAnalyzer, DependencyGraph, ToolCall
from .executor import AsyncToolFn, CallResult, ExecutionReport, Executor
from .observability import OTelIntegration, SynapseLogger
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
        logger: SynapseLogger | None = None,
        enable_json_logs: bool = False,
        enable_otel: bool = False,
        debug: bool = False,
    ) -> None:
        self.tools = tools
        self.debug = debug
        self.logger = logger or SynapseLogger(enabled=True, emit_json=enable_json_logs)
        self.otel = OTelIntegration() if enable_otel else None
        self._analyzer = DependencyAnalyzer()
        self._planner = Planner(logger=self.logger)
        self._executor = Executor(
            tools=tools,
            max_concurrency=max_concurrency,
            default_timeout=default_timeout,
            default_retries=default_retries,
            on_call_start=on_call_start,
            on_call_end=on_call_end,
            logger=self.logger,
            otel=self.otel,
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
        graph = self.analyze(calls)
        plan = self._planner.plan(graph)
        report = await self._executor.execute(plan)

        if self.debug:
            print(self._ascii_dag(graph, plan))
            print(self._execution_timeline())

        return report

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

    def _ascii_dag(self, graph: DependencyGraph, plan: ExecutionPlan) -> str:
        lines = ["\n--- Synapse DAG ---"]
        for stage in plan.stages:
            calls = " | ".join(f"[{call.id}] {call.name}" for call in stage.calls)
            lines.append(f"Stage {stage.index}: {calls}")
        lines.append("Edges:")
        for call_id, deps in graph.edges.items():
            if deps:
                for dep in sorted(deps):
                    lines.append(f"  {dep} -> {call_id}")
            else:
                lines.append(f"  {call_id} (root)")
        return "\n".join(lines)

    def _execution_timeline(self) -> str:
        lines = ["--- Execution Timeline ---"]
        for event in self.logger.events:
            if event.event not in {
                "stage_started",
                "stage_completed",
                "tool_started",
                "tool_completed",
                "tool_failed",
                "execution_complete",
            }:
                continue
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
            stage = "-" if event.stage_index is None else str(event.stage_index)
            tool = event.tool_id or "-"
            latency = "-" if event.latency_ms is None else f"{event.latency_ms:.2f}ms"
            error = f" error={event.error}" if event.error else ""
            lines.append(
                f"{timestamp} stage={stage} tool={tool} event={event.event} latency={latency}{error}"
            )
        return "\n".join(lines)
