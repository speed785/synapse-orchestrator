/**
 * DependencyAnalyzer — detects data dependencies between tool calls.
 *
 * A tool call B depends on A when:
 *  1. B.dependsOn includes A's id (explicit declaration), OR
 *  2. Any string value inside B.inputs matches "$results.<A.id>[.<path>]"
 *     (implicit, reference-based dependency).
 */

/** Pattern matching $results.<toolId>[.<path>] placeholders. */
const REF_PATTERN = /\$results\.([a-zA-Z0-9_\-]+)/g;

/** A single planned tool call. */
export interface ToolCall {
  /** Unique identifier within the execution plan. */
  id: string;
  /** Tool/function name to invoke. */
  name: string;
  /** Arguments to pass to the tool. */
  inputs?: Record<string, unknown>;
  /** Explicit list of call ids this call must wait for. */
  dependsOn?: string[];
  /** Per-call timeout in milliseconds (overrides orchestrator default). */
  timeoutMs?: number;
  /** Number of retry attempts on transient failure. */
  retries?: number;
  /** Arbitrary metadata — not used for dependency detection. */
  metadata?: Record<string, unknown>;
}

/** Result of the dependency analysis phase. */
export interface DependencyGraph {
  /** All tool calls, keyed by id. */
  nodes: Map<string, ToolCall>;
  /** id → set of ids it depends on. */
  edges: Map<string, Set<string>>;
  /** id → set of ids that depend on it (reverse). */
  revEdges: Map<string, Set<string>>;
}

/** Returns all $results.<id> references found anywhere inside a value. */
function extractRefs(value: unknown): Set<string> {
  const refs = new Set<string>();
  if (typeof value === "string") {
    let m: RegExpExecArray | null;
    // Reset lastIndex before each iteration
    REF_PATTERN.lastIndex = 0;
    while ((m = REF_PATTERN.exec(value)) !== null) {
      refs.add(m[1]);
    }
  } else if (Array.isArray(value)) {
    for (const item of value) {
      for (const ref of extractRefs(item)) refs.add(ref);
    }
  } else if (value !== null && typeof value === "object") {
    for (const v of Object.values(value as Record<string, unknown>)) {
      for (const ref of extractRefs(v)) refs.add(ref);
    }
  }
  return refs;
}

/** Detect cycles in the dependency graph using DFS. */
function isAcyclic(edges: Map<string, Set<string>>): boolean {
  const visited = new Set<string>();
  const stack = new Set<string>();

  function dfs(node: string): boolean {
    visited.add(node);
    stack.add(node);
    for (const dep of edges.get(node) ?? []) {
      if (!visited.has(dep)) {
        if (!dfs(dep)) return false;
      } else if (stack.has(dep)) {
        return false;
      }
    }
    stack.delete(node);
    return true;
  }

  for (const node of edges.keys()) {
    if (!visited.has(node)) {
      if (!dfs(node)) return false;
    }
  }
  return true;
}

/** Analyzes a list of ToolCalls and builds a DependencyGraph. */
export class DependencyAnalyzer {
  analyze(calls: ToolCall[]): DependencyGraph {
    const knownIds = new Set(calls.map((c) => c.id));
    const nodes = new Map(calls.map((c) => [c.id, c]));
    const edges = new Map<string, Set<string>>(calls.map((c) => [c.id, new Set()]));
    const revEdges = new Map<string, Set<string>>(calls.map((c) => [c.id, new Set()]));

    for (const call of calls) {
      const deps = new Set<string>(call.dependsOn ?? []);

      // Add implicit deps from $results.* references
      for (const ref of extractRefs(call.inputs ?? {})) {
        deps.add(ref);
      }

      // Validate all referenced ids exist
      for (const dep of deps) {
        if (!knownIds.has(dep)) {
          throw new Error(
            `ToolCall '${call.id}' references unknown tool id '${dep}'`
          );
        }
      }

      // Self-dependency is a no-op
      deps.delete(call.id);

      edges.set(call.id, deps);
      for (const dep of deps) {
        revEdges.get(dep)!.add(call.id);
      }
    }

    const graph: DependencyGraph = { nodes, edges, revEdges };

    if (!isAcyclic(edges)) {
      throw new Error(
        "Circular dependency detected in tool call plan. Synapse requires an acyclic graph."
      );
    }

    return graph;
  }
}
