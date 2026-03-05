/**
 * Executor — runs an ExecutionPlan with maximum parallelism.
 *
 * Features:
 * - Promise.all per Stage for true concurrent execution.
 * - Per-call timeout via AbortController + Promise.race.
 * - Retry with exponential back-off.
 * - Concurrency cap via a simple semaphore.
 * - $results.<id>[.<path>] placeholder substitution between stages.
 * - Rich ExecutionReport with per-call timing and speedup estimate.
 */

import type { ToolCall } from "./dependencyAnalyzer.js";
import type { OTelIntegration, SynapseLogger } from "./observability.js";
import type { ExecutionPlan } from "./planner.js";

export type AsyncToolFn = (...args: unknown[]) => Promise<unknown>;

/** Result for a single tool call. */
export interface CallResult {
  callId: string;
  toolName: string;
  status: "success" | "error" | "timeout";
  output?: unknown;
  error?: string;
  durationMs: number;
  attempts: number;
}

/** Full report returned by Executor.execute(). */
export interface ExecutionReport {
  results: Map<string, CallResult>;
  totalDurationMs: number;
  stagesRun: number;
  parallelCalls: number;
  sequentialCalls: number;
  /** sum(individual durations) / wall clock — ≥ 1 means parallelism saved time. */
  speedupEstimate: number;
}

// ── Placeholder resolution ─────────────────────────────────────────────────────

const REF_FULL = /^\$results\.([a-zA-Z0-9_\-.\[\]]+)$/;
const REF_INLINE = /\$results\.([a-zA-Z0-9_\-.\[\]]+)/g;

function splitRef(rawRef: string, results: Map<string, unknown>): [string, string | undefined] {
  const parts = rawRef.split(".");
  for (let i = parts.length; i > 0; i--) {
    const candidate = parts.slice(0, i).join(".");
    if (results.has(candidate)) {
      const suffix = parts.slice(i).join(".");
      return [candidate, suffix || undefined];
    }
  }
  return [rawRef, undefined];
}

function getNestedValue(obj: unknown, path: string | undefined): unknown {
  if (!path || obj === undefined || obj === null) return obj;
  for (const part of path.split(".")) {
    const arrayMatch = part.match(/^(.+)\[(\d+)\]$/);
    if (arrayMatch) {
      obj = (obj as Record<string, unknown>)[arrayMatch[1]];
      obj = (obj as unknown[])[parseInt(arrayMatch[2], 10)];
    } else if (typeof obj === "object" && obj !== null) {
      obj = (obj as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
    if (obj === undefined) return undefined;
  }
  return obj;
}

function resolve(value: unknown, results: Map<string, unknown>): unknown {
  if (typeof value === "string") {
    const fullMatch = REF_FULL.exec(value);
    if (fullMatch) {
      const [id, path] = splitRef(fullMatch[1], results);
      return getNestedValue(results.get(id), path);
    }
    // Inline substitution (always produces a string)
    REF_INLINE.lastIndex = 0;
    return value.replace(REF_INLINE, (match, rawRef: string) => {
      const [id, path] = splitRef(rawRef, results);
      const v = getNestedValue(results.get(id), path);
      return v !== undefined ? String(v) : match;
    });
  }
  if (Array.isArray(value)) return value.map((item) => resolve(item, results));
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [
        k,
        resolve(v, results),
      ])
    );
  }
  return value;
}

// ── Semaphore ──────────────────────────────────────────────────────────────────

class Semaphore {
  private _count: number;
  private _queue: Array<() => void> = [];

  constructor(max: number) {
    this._count = max;
  }

  async acquire(): Promise<void> {
    if (this._count > 0) {
      this._count--;
      return;
    }
    return new Promise((resolve) => this._queue.push(resolve));
  }

  release(): void {
    if (this._queue.length > 0) {
      this._queue.shift()!();
    } else {
      this._count++;
    }
  }
}

// ── Executor ───────────────────────────────────────────────────────────────────

export interface ExecutorOptions {
  maxConcurrency?: number;
  defaultTimeoutMs?: number;
  defaultRetries?: number;
  onCallStart?: (call: ToolCall) => void;
  onCallEnd?: (result: CallResult) => void;
  logger?: SynapseLogger;
  otel?: OTelIntegration;
}

export class Executor {
  private tools: Map<string, AsyncToolFn>;
  private semaphore: Semaphore;
  private defaultTimeoutMs: number;
  private defaultRetries: number;
  private onCallStart: ((call: ToolCall) => void) | undefined;
  private onCallEnd: ((result: CallResult) => void) | undefined;
  private logger: SynapseLogger | undefined;
  private otel: OTelIntegration | undefined;

  constructor(tools: Record<string, AsyncToolFn>, options: ExecutorOptions = {}) {
    this.tools = new Map(Object.entries(tools));
    this.semaphore = new Semaphore(options.maxConcurrency ?? 16);
    this.defaultTimeoutMs = options.defaultTimeoutMs ?? 30_000;
    this.defaultRetries = options.defaultRetries ?? 0;
    this.onCallStart = options.onCallStart;
    this.onCallEnd = options.onCallEnd;
    this.logger = options.logger;
    this.otel = options.otel;
  }

  async execute(plan: ExecutionPlan): Promise<ExecutionReport> {
    const accumulated = new Map<string, unknown>();
    const allResults = new Map<string, CallResult>();
    let sumDurations = 0;
    let parallelCalls = 0;
    let sequentialCalls = 0;

    const wallStart = performance.now();

    for (const stage of plan.stages) {
      const stageStart = performance.now();
      this.logger?.log("stage_started", {
        stageIndex: stage.index,
        parallelCount: stage.calls.length,
        toolCount: stage.calls.length,
      });
      const stageSpan = this.otel?.startSpan("synapse.stage", {
        stageIndex: stage.index,
        toolCount: stage.calls.length,
      });
      if (stage.calls.length > 1) {
        parallelCalls += stage.calls.length;
      } else {
        sequentialCalls += stage.calls.length;
      }

      const stageResults = await Promise.all(
        stage.calls.map((call) => this.runCall(call, accumulated, stage.index, stage.calls.length))
      );

      const stageLatencyMs = performance.now() - stageStart;
      this.logger?.log("stage_completed", {
        stageIndex: stage.index,
        latencyMs: stageLatencyMs,
        parallelCount: stage.calls.length,
        toolCount: stage.calls.length,
      });
      this.otel?.endSpan(stageSpan, stageResults.every((result) => result.status === "success"));

      for (const result of stageResults) {
        allResults.set(result.callId, result);
        accumulated.set(result.callId, result.output);
        sumDurations += result.durationMs;
      }
    }

    const totalDurationMs = performance.now() - wallStart;
    const speedupEstimate = totalDurationMs > 0 ? sumDurations / totalDurationMs : 1;
    const avgParallelism = plan.stages.length > 0 ? plan.totalCalls / plan.stages.length : 0;
    this.logger?.recordExecution(speedupEstimate, avgParallelism);
    this.logger?.log("execution_complete", {
      latencyMs: totalDurationMs,
      parallelCount: parallelCalls,
      totalCalls: plan.totalCalls,
      speedupRatio: speedupEstimate,
    });

    return {
      results: allResults,
      totalDurationMs,
      stagesRun: plan.stages.length,
      parallelCalls,
      sequentialCalls,
      speedupEstimate,
    };
  }

  private async runCall(
    call: ToolCall,
    accumulated: Map<string, unknown>,
    stageIndex: number,
    parallelCount: number
  ): Promise<CallResult> {
    const toolFn = this.tools.get(call.name);
    if (!toolFn) {
      this.logger?.log("tool_failed", {
        stageIndex,
        toolId: call.id,
        parallelCount,
        error: `No tool registered for '${call.name}'`,
      });
      return {
        callId: call.id,
        toolName: call.name,
        status: "error",
        error: `No tool registered for '${call.name}'`,
        durationMs: 0,
        attempts: 0,
      };
    }

    const resolvedInputs = resolve(call.inputs ?? {}, accumulated) as Record<
      string,
      unknown
    >;
    const timeoutMs = call.timeoutMs ?? this.defaultTimeoutMs;
    const maxAttempts = 1 + (call.retries ?? this.defaultRetries);

    let lastError = "";
    let lastStatus: "error" | "timeout" = "error";
    let lastDuration = 0;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      this.onCallStart?.(call);
      this.logger?.log("tool_started", {
        stageIndex,
        toolId: call.id,
        parallelCount,
      });
      const toolSpan = this.otel?.startSpan("synapse.tool_call", {
        stageIndex,
        toolId: call.id,
        toolName: call.name,
        attempt,
      });
      const t0 = performance.now();

      try {
        await this.semaphore.acquire();
        let output: unknown;
        try {
          output = await Promise.race([
            toolFn(...Object.values(resolvedInputs)),
            new Promise<never>((_, reject) =>
              setTimeout(() => reject(new Error(`__timeout__`)), timeoutMs)
            ),
          ]);
        } finally {
          this.semaphore.release();
        }

        const durationMs = performance.now() - t0;
        const result: CallResult = {
          callId: call.id,
          toolName: call.name,
          status: "success",
          output,
          durationMs,
          attempts: attempt,
        };
        this.onCallEnd?.(result);
        this.logger?.recordToolResult(durationMs, false);
        this.logger?.log("tool_completed", {
          stageIndex,
          toolId: call.id,
          latencyMs: durationMs,
          parallelCount,
        });
        this.otel?.endSpan(toolSpan, true);
        return result;
      } catch (err: unknown) {
        lastDuration = performance.now() - t0;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg === "__timeout__") {
          lastStatus = "timeout";
          lastError = `Timed out after ${timeoutMs}ms`;
        } else {
          lastStatus = "error";
          lastError = msg;
        }
        this.otel?.endSpan(toolSpan, false, lastError);

        if (attempt < maxAttempts) {
          await new Promise((r) => setTimeout(r, 100 * 2 ** (attempt - 1)));
        }
      }
    }

    const result: CallResult = {
      callId: call.id,
      toolName: call.name,
      status: lastStatus,
      error: lastError,
      durationMs: lastDuration,
      attempts: maxAttempts,
    };
    this.onCallEnd?.(result);
    this.logger?.recordToolResult(lastDuration, true);
    this.logger?.log("tool_failed", {
      stageIndex,
      toolId: call.id,
      latencyMs: lastDuration,
      parallelCount,
      error: lastError,
      status: lastStatus,
    });
    return result;
  }
}
