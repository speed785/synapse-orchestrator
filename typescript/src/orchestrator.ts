/**
 * Orchestrator — top-level entry point for Synapse (TypeScript).
 *
 * Wires together DependencyAnalyzer → Planner → Executor and exposes
 * a clean, ergonomic API.
 *
 * @example
 * ```ts
 * import { Orchestrator, ToolCall } from "synapse-orchestrator";
 *
 * const orch = new Orchestrator({
 *   tools: {
 *     fetchUser:   fetchUserFn,
 *     fetchOrders: fetchOrdersFn,
 *     sendEmail:   sendEmailFn,
 *   },
 * });
 *
 * const report = await orch.run([
 *   { id: "u",  name: "fetchUser",   inputs: { userId: 42 } },
 *   { id: "o",  name: "fetchOrders", inputs: { userId: 42 } },
 *   { id: "em", name: "sendEmail",
 *     inputs: { to: "$results.u.email", body: "$results.o" } },
 * ]);
 *
 * console.log(`Speedup: ${report.speedupEstimate.toFixed(2)}x`);
 * ```
 */

import { DependencyAnalyzer, ToolCall, DependencyGraph } from "./dependencyAnalyzer.js";
import { Planner, ExecutionPlan } from "./planner.js";
import { AsyncToolFn, ExecutionReport, Executor, ExecutorOptions } from "./executor.js";

export type { ToolCall, DependencyGraph } from "./dependencyAnalyzer.js";
export type { ExecutionPlan, Stage } from "./planner.js";
export type { AsyncToolFn, CallResult, ExecutionReport } from "./executor.js";

export interface OrchestratorOptions extends ExecutorOptions {
  tools: Record<string, AsyncToolFn>;
}

export class Orchestrator {
  private analyzer: DependencyAnalyzer;
  private planner: Planner;
  private executor: Executor;

  constructor(options: OrchestratorOptions) {
    this.analyzer = new DependencyAnalyzer();
    this.planner = new Planner();
    this.executor = new Executor(options.tools, options);
  }

  /** Build the dependency graph without executing. */
  analyze(calls: ToolCall[]): DependencyGraph {
    return this.analyzer.analyze(calls);
  }

  /** Build the execution plan without executing. */
  plan(calls: ToolCall[]): ExecutionPlan {
    const graph = this.analyzer.analyze(calls);
    return this.planner.plan(graph);
  }

  /** Analyze, plan, and execute. Returns a detailed ExecutionReport. */
  async run(calls: ToolCall[]): Promise<ExecutionReport> {
    const plan = this.plan(calls);
    return this.executor.execute(plan);
  }

  /**
   * Convenience method accepting plain objects — identical to `run` but
   * TypeScript types are relaxed to `unknown` for rapid prototyping.
   */
  async runRaw(rawCalls: Record<string, unknown>[]): Promise<ExecutionReport> {
    return this.run(rawCalls as unknown as ToolCall[]);
  }
}
