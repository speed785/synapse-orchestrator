"""
Synapse — Parallel Tool Call Orchestrator for AI Agents
=======================================================

Automatically detects data dependencies between tool calls and executes
them with maximum parallelism, dramatically reducing agent pipeline latency.

Quick start
-----------
::

    import asyncio
    from synapse import Orchestrator, ToolCall

    async def main():
        orch = Orchestrator(tools={
            "fetch_user":   fetch_user_fn,
            "fetch_orders": fetch_orders_fn,
            "send_email":   send_email_fn,
        })

        report = await orch.run([
            ToolCall(id="u",  name="fetch_user",   inputs={"user_id": 42}),
            ToolCall(id="o",  name="fetch_orders",  inputs={"user_id": 42}),
            ToolCall(id="em", name="send_email",
                     inputs={"to": "$results.u.email",
                              "body": "$results.o"}),
        ])
        print(report)

    asyncio.run(main())
"""

__version__ = "0.1.0"

from .dependency_analyzer import DependencyAnalyzer, DependencyGraph, ToolCall
from .executor import CallResult, ExecutionReport, Executor
from .observability import ExecutionMetrics, OTelIntegration, SynapseLogger
from .orchestrator import Orchestrator
from .planner import ExecutionPlan, Planner, Stage

__all__ = [
    # Core classes
    "Orchestrator",
    "ToolCall",
    # Analysis
    "DependencyAnalyzer",
    "DependencyGraph",
    # Planning
    "Planner",
    "ExecutionPlan",
    "Stage",
    # Execution
    "Executor",
    "CallResult",
    "ExecutionReport",
    "SynapseLogger",
    "ExecutionMetrics",
    "OTelIntegration",
]
