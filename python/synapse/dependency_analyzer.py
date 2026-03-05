"""
Dependency Analyzer — detects data dependencies between tool calls.

A tool call B depends on A if any input value of B references an output
placeholder produced by A (e.g. "$results.tool_a.output_field").
We also support explicit `depends_on` declarations for cases where the
dependency is semantic rather than syntactic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Pattern that matches output references like $results.<tool_id>.<field>
_REF_PATTERN = re.compile(r"\$results\.([a-zA-Z0-9_\-]+)")


@dataclass
class ToolCall:
    """
    Represents a single planned tool call.

    Attributes:
        id:         Unique identifier for this call within an execution plan.
        name:       Tool/function name to invoke.
        inputs:     Keyword arguments passed to the tool.
        depends_on: Explicit list of tool-call IDs this call must wait for.
                    Synapse also detects implicit deps from $results.* refs.
        timeout:    Per-call timeout in seconds (None = global default).
        retries:    Number of retry attempts on transient failure.
        metadata:   Arbitrary caller-supplied metadata (not used for deps).
    """

    id: str
    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    timeout: float | None = None
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyGraph:
    """
    Result of dependency analysis.

    Attributes:
        nodes:      All ToolCall objects, keyed by id.
        edges:      Mapping from each call id → set of ids it depends on.
        rev_edges:  Reverse mapping: id → set of ids that depend on it.
    """

    nodes: dict[str, ToolCall]
    edges: dict[str, set[str]]
    rev_edges: dict[str, set[str]]

    def dependencies_of(self, call_id: str) -> set[str]:
        return self.edges.get(call_id, set())

    def dependents_of(self, call_id: str) -> set[str]:
        return self.rev_edges.get(call_id, set())

    def roots(self) -> list[str]:
        """Tool calls with no dependencies — can start immediately."""
        return [cid for cid, deps in self.edges.items() if not deps]

    def is_acyclic(self) -> bool:
        """Return True if the graph contains no cycles (valid DAG)."""
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            stack.add(node)
            for dep in self.edges.get(node, set()):
                if dep not in visited:
                    if not dfs(dep):
                        return False
                elif dep in stack:
                    return False
            stack.discard(node)
            return True

        return all(dfs(n) for n in self.nodes if n not in visited)


def _extract_refs(value: Any) -> set[str]:
    """
    Recursively walk a value (dict, list, str, …) and collect all
    tool-call IDs referenced via $results.<id> syntax.
    """
    refs: set[str] = set()
    if isinstance(value, str):
        for match in _REF_PATTERN.finditer(value):
            refs.add(match.group(1))
    elif isinstance(value, dict):
        for v in value.values():
            refs |= _extract_refs(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            refs |= _extract_refs(item)
    return refs


class DependencyAnalyzer:
    """
    Analyzes a list of ToolCall objects and builds a DependencyGraph.

    Dependency detection rules (applied in order, union of all matches):
    1. Explicit ``depends_on`` field on the ToolCall.
    2. ``$results.<tool_id>`` placeholder references inside ``inputs``.

    Raises:
        ValueError: If a referenced tool id does not exist in the plan,
                    or if the resulting graph contains a cycle.
    """

    def analyze(self, calls: list[ToolCall]) -> DependencyGraph:
        known_ids = {c.id for c in calls}
        nodes = {c.id: c for c in calls}
        edges: dict[str, set[str]] = {c.id: set() for c in calls}
        rev_edges: dict[str, set[str]] = {c.id: set() for c in calls}

        for call in calls:
            deps: set[str] = set(call.depends_on)
            deps |= _extract_refs(call.inputs)

            # Validate referenced IDs exist
            unknown = deps - known_ids
            if unknown:
                raise ValueError(
                    f"ToolCall '{call.id}' references unknown tool ids: {unknown}"
                )

            # A call cannot depend on itself
            deps.discard(call.id)

            edges[call.id] = deps
            for dep_id in deps:
                rev_edges[dep_id].add(call.id)

        graph = DependencyGraph(nodes=nodes, edges=edges, rev_edges=rev_edges)

        if not graph.is_acyclic():
            raise ValueError(
                "Circular dependency detected in tool call plan. "
                "Synapse requires an acyclic dependency graph."
            )

        return graph
