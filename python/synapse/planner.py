"""
Planner — converts a DependencyGraph into an ordered execution plan.

The plan is a list of *stages*. Each stage is a group of tool calls
that have no unresolved dependencies between them and can therefore
be launched concurrently.  Stages are computed via Kahn's topological
sort, partitioned by BFS level so that maximum parallelism is achieved
at every step.

Example
-------
Given the dependency graph::

    A ──┐
        ├──► C ──► E
    B ──┘
         D ──────► E (also)

The planner produces::

    Stage 0 (parallel): [A, B, D]
    Stage 1 (parallel): [C]
    Stage 2 (parallel): [E]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .dependency_analyzer import DependencyGraph, ToolCall
from .observability import SynapseLogger


@dataclass
class Stage:
    """A set of tool calls that can be executed concurrently."""

    index: int
    calls: list[ToolCall] = field(default_factory=list)

    def __repr__(self) -> str:
        ids = ", ".join(c.id for c in self.calls)
        return f"Stage({self.index}, [{ids}])"


@dataclass
class ExecutionPlan:
    """
    Ordered list of Stages produced by the Planner.

    Attributes:
        stages:         Ordered list from first to last.
        total_calls:    Total number of tool calls.
        parallelism:    Average number of calls per stage (higher = better).
        critical_path:  Sequence of call ids forming the longest dependency chain.
    """

    stages: list[Stage]
    total_calls: int
    parallelism: float
    critical_path: list[str]

    def __repr__(self) -> str:
        lines = [
            f"ExecutionPlan({self.total_calls} calls, "
            f"{len(self.stages)} stages, "
            f"avg parallelism={self.parallelism:.1f}x, "
            f"critical_path={self.critical_path})"
        ]
        for stage in self.stages:
            ids = [c.id for c in stage.calls]
            marker = " ← critical" if any(i in self.critical_path for i in ids) else ""
            lines.append(f"  Stage {stage.index}: {ids}{marker}")
        return "\n".join(lines)


class Planner:
    """
    Builds an ExecutionPlan from a DependencyGraph using BFS-level
    topological sort (Kahn's algorithm with level tracking).
    """

    def __init__(self, logger: SynapseLogger | None = None) -> None:
        self.logger = logger

    def plan(self, graph: DependencyGraph) -> ExecutionPlan:
        # in_degree: how many unresolved deps each node has
        in_degree: dict[str, int] = {
            cid: len(deps) for cid, deps in graph.edges.items()
        }

        # Start with all roots (zero in-degree)
        queue: deque[str] = deque(
            cid for cid, deg in in_degree.items() if deg == 0
        )

        stages: list[Stage] = []
        processed = 0

        while queue:
            # Everything currently in the queue shares the same BFS level
            level_size = len(queue)
            stage = Stage(index=len(stages))

            for _ in range(level_size):
                cid = queue.popleft()
                stage.calls.append(graph.nodes[cid])
                processed += 1

                # Reduce in-degree for all dependents
                for dependent_id in graph.rev_edges.get(cid, set()):
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)

            stages.append(stage)

        if processed != len(graph.nodes):
            raise ValueError(
                "Cycle detected during planning — not all nodes were processed. "
                f"({processed}/{len(graph.nodes)} processed)"
            )

        total = len(graph.nodes)
        parallelism = total / len(stages) if stages else 0.0
        critical = self._critical_path(graph)
        estimated_speedup = (
            total / len(critical)
            if critical
            else 1.0
        )

        if self.logger is not None:
            self.logger.log(
                "dag_planned",
                parallel_count=max((len(stage.calls) for stage in stages), default=0),
                stage_count=len(stages),
                critical_path=critical,
                estimated_speedup=estimated_speedup,
            )

        return ExecutionPlan(
            stages=stages,
            total_calls=total,
            parallelism=parallelism,
            critical_path=critical,
        )

    # ------------------------------------------------------------------
    # Critical path (longest chain by number of hops)
    # ------------------------------------------------------------------

    def _critical_path(self, graph: DependencyGraph) -> list[str]:
        """
        Returns the sequence of call ids forming the longest dependency
        chain (by hop count).  Ties are broken by id lexicographic order.
        """
        memo: dict[str, list[str]] = {}

        def longest(cid: str) -> list[str]:
            if cid in memo:
                return memo[cid]
            preds = graph.edges.get(cid, set())
            if not preds:
                memo[cid] = [cid]
                return [cid]
            best = max(
                (longest(p) for p in preds),
                key=lambda path: (len(path), path),
            )
            memo[cid] = best + [cid]
            return memo[cid]

        if not graph.nodes:
            return []

        return max(
            (longest(cid) for cid in graph.nodes),
            key=lambda path: (len(path), path),
        )
