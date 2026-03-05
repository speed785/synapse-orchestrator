/**
 * Planner — converts a DependencyGraph into an ordered ExecutionPlan.
 *
 * Uses BFS-level topological sort (Kahn's algorithm) to group tool calls
 * into Stages.  All calls within a Stage have no mutual dependencies and
 * can be launched concurrently.
 */

import type { DependencyGraph, ToolCall } from "./dependencyAnalyzer.js";
import type { SynapseLogger } from "./observability.js";

/** A set of tool calls that can execute concurrently. */
export interface Stage {
  index: number;
  calls: ToolCall[];
}

/** The full ordered execution plan produced by the Planner. */
export interface ExecutionPlan {
  stages: Stage[];
  totalCalls: number;
  /** Average calls per stage — higher means more parallelism. */
  parallelism: number;
  /** Sequence of call ids forming the longest dependency chain. */
  criticalPath: string[];
}

/** Builds an ExecutionPlan from a DependencyGraph. */
export class Planner {
  constructor(private logger?: SynapseLogger) {}

  plan(graph: DependencyGraph): ExecutionPlan {
    // Compute in-degree for each node
    const inDegree = new Map<string, number>();
    for (const [id, deps] of graph.edges) {
      inDegree.set(id, deps.size);
    }

    // Start queue with all zero-in-degree nodes
    const queue: string[] = [];
    for (const [id, deg] of inDegree) {
      if (deg === 0) queue.push(id);
    }

    const stages: Stage[] = [];
    let processed = 0;

    while (queue.length > 0) {
      // Drain the current level into a Stage
      const levelSize = queue.length;
      const stage: Stage = { index: stages.length, calls: [] };

      for (let i = 0; i < levelSize; i++) {
        const id = queue.shift()!;
        stage.calls.push(graph.nodes.get(id)!);
        processed++;

        // Reduce in-degree of dependents
        for (const dependentId of graph.revEdges.get(id) ?? []) {
          const newDeg = (inDegree.get(dependentId) ?? 0) - 1;
          inDegree.set(dependentId, newDeg);
          if (newDeg === 0) {
            queue.push(dependentId);
          }
        }
      }

      stages.push(stage);
    }

    if (processed !== graph.nodes.size) {
      throw new Error(
        `Cycle detected during planning — ${processed}/${graph.nodes.size} nodes processed.`
      );
    }

    const totalCalls = graph.nodes.size;
    const parallelism = stages.length > 0 ? totalCalls / stages.length : 0;
    const criticalPath = this.computeCriticalPath(graph);
    const estimatedSpeedup = criticalPath.length > 0 ? totalCalls / criticalPath.length : 1;

    this.logger?.log("dag_planned", {
      parallelCount: Math.max(0, ...stages.map((stage) => stage.calls.length)),
      stageCount: stages.length,
      criticalPath,
      estimatedSpeedup,
    });

    return { stages, totalCalls, parallelism, criticalPath };
  }

  private computeCriticalPath(graph: DependencyGraph): string[] {
    const memo = new Map<string, string[]>();

    const longest = (id: string): string[] => {
      if (memo.has(id)) return memo.get(id)!;
      const preds = graph.edges.get(id) ?? new Set();
      if (preds.size === 0) {
        memo.set(id, [id]);
        return [id];
      }
      let best: string[] = [];
      for (const pred of preds) {
        const path = longest(pred);
        if (path.length > best.length) best = path;
      }
      const result = [...best, id];
      memo.set(id, result);
      return result;
    };

    let globalBest: string[] = [];
    for (const id of graph.nodes.keys()) {
      const p = longest(id);
      if (p.length > globalBest.length) globalBest = p;
    }
    return globalBest;
  }
}
