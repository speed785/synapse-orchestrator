export type SynapseEventName =
  | "dag_planned"
  | "stage_started"
  | "stage_completed"
  | "tool_started"
  | "tool_completed"
  | "tool_failed"
  | "execution_complete";

export interface LogEvent {
  event: SynapseEventName;
  timestamp: number;
  stageIndex?: number;
  toolId?: string;
  latencyMs?: number;
  parallelCount?: number;
  error?: string;
  [key: string]: unknown;
}

function percentile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const pos = (ordered.length - 1) * q;
  const lower = Math.floor(pos);
  const upper = Math.min(lower + 1, ordered.length - 1);
  if (lower === upper) return ordered[lower];
  const weight = pos - lower;
  return ordered[lower] * (1 - weight) + ordered[upper] * weight;
}

export class ExecutionMetrics {
  totalExecutions = 0;
  totalToolCalls = 0;
  avgSpeedupRatio = 0;
  avgParallelism = 0;
  toolFailureRate = 0;
  p50LatencyMs = 0;
  p95LatencyMs = 0;
  p99LatencyMs = 0;

  private latencies: number[] = [];
  private failedTools = 0;

  recordExecution(speedupRatio: number, parallelism: number): void {
    this.totalExecutions += 1;
    const n = this.totalExecutions;
    this.avgSpeedupRatio = ((this.avgSpeedupRatio * (n - 1)) + speedupRatio) / n;
    this.avgParallelism = ((this.avgParallelism * (n - 1)) + parallelism) / n;
  }

  recordToolCall(latencyMs: number, failed: boolean): void {
    this.totalToolCalls += 1;
    this.latencies.push(latencyMs);
    if (failed) this.failedTools += 1;
    this.toolFailureRate = this.totalToolCalls > 0 ? this.failedTools / this.totalToolCalls : 0;
    this.p50LatencyMs = percentile(this.latencies, 0.5);
    this.p95LatencyMs = percentile(this.latencies, 0.95);
    this.p99LatencyMs = percentile(this.latencies, 0.99);
  }

  exportPrometheus(): string {
    return [
      "# HELP synapse_total_executions Number of completed executions.",
      "# TYPE synapse_total_executions counter",
      `synapse_total_executions ${this.totalExecutions}`,
      "# HELP synapse_total_tool_calls Number of tool calls executed.",
      "# TYPE synapse_total_tool_calls counter",
      `synapse_total_tool_calls ${this.totalToolCalls}`,
      "# HELP synapse_avg_speedup_ratio Average speedup ratio across executions.",
      "# TYPE synapse_avg_speedup_ratio gauge",
      `synapse_avg_speedup_ratio ${this.avgSpeedupRatio.toFixed(6)}`,
      "# HELP synapse_avg_parallelism Average parallelism across executions.",
      "# TYPE synapse_avg_parallelism gauge",
      `synapse_avg_parallelism ${this.avgParallelism.toFixed(6)}`,
      "# HELP synapse_tool_failure_rate Failure ratio across tool calls.",
      "# TYPE synapse_tool_failure_rate gauge",
      `synapse_tool_failure_rate ${this.toolFailureRate.toFixed(6)}`,
      "# HELP synapse_latency_ms Latency percentiles in milliseconds.",
      "# TYPE synapse_latency_ms gauge",
      `synapse_latency_ms{quantile="0.50"} ${this.p50LatencyMs.toFixed(6)}`,
      `synapse_latency_ms{quantile="0.95"} ${this.p95LatencyMs.toFixed(6)}`,
      `synapse_latency_ms{quantile="0.99"} ${this.p99LatencyMs.toFixed(6)}`,
      "",
    ].join("\n");
  }
}

export class SynapseLogger {
  readonly events: LogEvent[] = [];
  readonly metrics = new ExecutionMetrics();

  constructor(private options: { enabled?: boolean; emitJson?: boolean } = {}) {}

  log(event: SynapseEventName, payload: Omit<LogEvent, "event" | "timestamp"> = {}): void {
    if (this.options.enabled === false) return;
    const entry: LogEvent = { event, timestamp: Date.now() / 1000, ...payload };
    this.events.push(entry);
    if (this.options.emitJson) {
      console.log(JSON.stringify(entry));
    }
  }

  recordToolResult(latencyMs: number, failed: boolean): void {
    this.metrics.recordToolCall(latencyMs, failed);
  }

  recordExecution(speedupRatio: number, parallelism: number): void {
    this.metrics.recordExecution(speedupRatio, parallelism);
  }

  exportPrometheus(): string {
    return this.metrics.exportPrometheus();
  }
}

export interface SpanLike {
  setAttribute?(name: string, value: unknown): void;
  recordException?(error: unknown): void;
  end(): void;
}

export interface TracerLike {
  startSpan(name: string): SpanLike;
}

export class OTelIntegration {
  constructor(private tracer?: TracerLike) {}

  get enabled(): boolean {
    return this.tracer !== undefined;
  }

  startSpan(name: string, attributes: Record<string, unknown> = {}): SpanLike | undefined {
    if (!this.tracer) return undefined;
    const span = this.tracer.startSpan(name);
    for (const [key, value] of Object.entries(attributes)) {
      if (value !== undefined) {
        span.setAttribute?.(key, value);
      }
    }
    return span;
  }

  endSpan(span: SpanLike | undefined, ok: boolean, error?: string): void {
    if (!span) return;
    if (!ok && error) {
      span.recordException?.(new Error(error));
    }
    span.end();
  }
}
