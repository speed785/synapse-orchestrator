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
import { OTelIntegration, SynapseLogger } from "./observability.js";

export type { ToolCall, DependencyGraph } from "./dependencyAnalyzer.js";
export type { ExecutionPlan, Stage } from "./planner.js";
export type { AsyncToolFn, CallResult, ExecutionReport } from "./executor.js";

export interface OrchestratorOptions extends ExecutorOptions {
  tools: Record<string, AsyncToolFn>;
  logger?: SynapseLogger;
  enableJsonLogs?: boolean;
  enableOtel?: boolean;
  debug?: boolean;
}

export class Orchestrator {
  private analyzer: DependencyAnalyzer;
  private planner: Planner;
  private executor: Executor;
  private logger: SynapseLogger;
  private debug: boolean;

  constructor(options: OrchestratorOptions) {
    if (options.logger) {
      this.logger = options.logger;
    } else if (options.enableJsonLogs !== undefined) {
      this.logger = new SynapseLogger({ enabled: true, emitJson: options.enableJsonLogs });
    } else {
      this.logger = new SynapseLogger({ enabled: true });
    }
    const otel = options.enableOtel ? new OTelIntegration() : undefined;
    this.debug = options.debug ?? false;
    this.analyzer = new DependencyAnalyzer();
    this.planner = new Planner(this.logger);
    const executorOptions: ExecutorOptions = { logger: this.logger };
    if (options.maxConcurrency !== undefined) {
      executorOptions.maxConcurrency = options.maxConcurrency;
    }
    if (options.defaultTimeoutMs !== undefined) {
      executorOptions.defaultTimeoutMs = options.defaultTimeoutMs;
    }
    if (options.defaultRetries !== undefined) {
      executorOptions.defaultRetries = options.defaultRetries;
    }
    if (options.onCallStart !== undefined) {
      executorOptions.onCallStart = options.onCallStart;
    }
    if (options.onCallEnd !== undefined) {
      executorOptions.onCallEnd = options.onCallEnd;
    }
    if (otel) {
      executorOptions.otel = otel;
    }
    this.executor = new Executor(options.tools, executorOptions);
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
    const graph = this.analyzer.analyze(calls);
    const plan = this.planner.plan(graph);
    const report = await this.executor.execute(plan);
    if (this.debug) {
      console.log(this.asciiDag(graph, plan));
      console.log(this.executionTimeline());
    }
    return report;
  }

  /**
   * Convenience method accepting plain objects — identical to `run` but
   * TypeScript types are relaxed to `unknown` for rapid prototyping.
   */
  async runRaw(rawCalls: Record<string, unknown>[]): Promise<ExecutionReport> {
    return this.run(rawCalls as unknown as ToolCall[]);
  }

  private asciiDag(graph: DependencyGraph, plan: ExecutionPlan): string {
    const lines = ["\n--- Synapse DAG ---"];
    for (const stage of plan.stages) {
      const calls = stage.calls.map((call) => `[${call.id}] ${call.name}`).join(" | ");
      lines.push(`Stage ${stage.index}: ${calls}`);
    }
    lines.push("Edges:");
    for (const [id, deps] of graph.edges) {
      if (deps.size === 0) {
        lines.push(`  ${id} (root)`);
      } else {
        for (const dep of deps) {
          lines.push(`  ${dep} -> ${id}`);
        }
      }
    }
    return lines.join("\n");
  }

  private executionTimeline(): string {
    const lines = ["--- Execution Timeline ---"];
    for (const event of this.logger.events) {
      if (!["stage_started", "stage_completed", "tool_started", "tool_completed", "tool_failed", "execution_complete"].includes(event.event)) {
        continue;
      }
      const timestamp = new Date(event.timestamp * 1000).toISOString().slice(11, 23);
      const stage = event.stageIndex ?? "-";
      const tool = event.toolId ?? "-";
      const latency = event.latencyMs !== undefined ? `${event.latencyMs.toFixed(2)}ms` : "-";
      const error = event.error ? ` error=${event.error}` : "";
      lines.push(`${timestamp} stage=${stage} tool=${tool} event=${event.event} latency=${latency}${error}`);
    }
    return lines.join("\n");
  }
}
