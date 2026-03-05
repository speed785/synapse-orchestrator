from __future__ import annotations

import json
import importlib
import time
from dataclasses import dataclass, field
from typing import Any


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass
class ExecutionMetrics:
    total_executions: int = 0
    total_tool_calls: int = 0
    avg_speedup_ratio: float = 0.0
    avg_parallelism: float = 0.0
    tool_failure_rate: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    _latencies_ms: list[float] = field(default_factory=list, repr=False)
    _failed_tools: int = field(default=0, repr=False)

    def record_execution(self, *, speedup_ratio: float, parallelism: float) -> None:
        self.total_executions += 1
        n = self.total_executions
        self.avg_speedup_ratio = ((self.avg_speedup_ratio * (n - 1)) + speedup_ratio) / n
        self.avg_parallelism = ((self.avg_parallelism * (n - 1)) + parallelism) / n

    def record_tool_call(self, *, latency_ms: float, failed: bool) -> None:
        self.total_tool_calls += 1
        self._latencies_ms.append(latency_ms)
        if failed:
            self._failed_tools += 1
        self.tool_failure_rate = (
            self._failed_tools / self.total_tool_calls if self.total_tool_calls else 0.0
        )
        self.p50_latency_ms = _percentile(self._latencies_ms, 0.50)
        self.p95_latency_ms = _percentile(self._latencies_ms, 0.95)
        self.p99_latency_ms = _percentile(self._latencies_ms, 0.99)

    def export_prometheus(self) -> str:
        lines = [
            "# HELP synapse_total_executions Number of completed executions.",
            "# TYPE synapse_total_executions counter",
            f"synapse_total_executions {self.total_executions}",
            "# HELP synapse_total_tool_calls Number of tool calls executed.",
            "# TYPE synapse_total_tool_calls counter",
            f"synapse_total_tool_calls {self.total_tool_calls}",
            "# HELP synapse_avg_speedup_ratio Average speedup ratio across executions.",
            "# TYPE synapse_avg_speedup_ratio gauge",
            f"synapse_avg_speedup_ratio {self.avg_speedup_ratio:.6f}",
            "# HELP synapse_avg_parallelism Average parallelism across executions.",
            "# TYPE synapse_avg_parallelism gauge",
            f"synapse_avg_parallelism {self.avg_parallelism:.6f}",
            "# HELP synapse_tool_failure_rate Failure ratio across tool calls.",
            "# TYPE synapse_tool_failure_rate gauge",
            f"synapse_tool_failure_rate {self.tool_failure_rate:.6f}",
            "# HELP synapse_latency_ms Latency percentiles in milliseconds.",
            "# TYPE synapse_latency_ms gauge",
            f'synapse_latency_ms{{quantile="0.50"}} {self.p50_latency_ms:.6f}',
            f'synapse_latency_ms{{quantile="0.95"}} {self.p95_latency_ms:.6f}',
            f'synapse_latency_ms{{quantile="0.99"}} {self.p99_latency_ms:.6f}',
        ]
        return "\n".join(lines) + "\n"


@dataclass
class LogEvent:
    event: str
    timestamp: float
    stage_index: int | None = None
    tool_id: str | None = None
    latency_ms: float | None = None
    parallel_count: int | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": self.event,
            "timestamp": self.timestamp,
            "stage_index": self.stage_index,
            "tool_id": self.tool_id,
            "latency_ms": self.latency_ms,
            "parallel_count": self.parallel_count,
            "error": self.error,
        }
        payload.update(self.details)
        return payload


class SynapseLogger:
    def __init__(self, *, enabled: bool = True, emit_json: bool = False) -> None:
        self.enabled = enabled
        self.emit_json = emit_json
        self.events: list[LogEvent] = []
        self.metrics = ExecutionMetrics()

    def log(
        self,
        event: str,
        *,
        stage_index: int | None = None,
        tool_id: str | None = None,
        latency_ms: float | None = None,
        parallel_count: int | None = None,
        error: str | None = None,
        **details: Any,
    ) -> None:
        if not self.enabled:
            return
        item = LogEvent(
            event=event,
            timestamp=time.time(),
            stage_index=stage_index,
            tool_id=tool_id,
            latency_ms=latency_ms,
            parallel_count=parallel_count,
            error=error,
            details=details,
        )
        self.events.append(item)
        if self.emit_json:
            print(json.dumps(item.as_dict(), default=str, sort_keys=True))

    def record_tool_result(self, *, latency_ms: float, failed: bool) -> None:
        self.metrics.record_tool_call(latency_ms=latency_ms, failed=failed)

    def record_execution(self, *, speedup_ratio: float, parallelism: float) -> None:
        self.metrics.record_execution(
            speedup_ratio=speedup_ratio,
            parallelism=parallelism,
        )

    def export_prometheus(self) -> str:
        return self.metrics.export_prometheus()


class OTelIntegration:
    def __init__(self, service_name: str = "synapse-orchestrator") -> None:
        self._available = False
        self._trace = None
        self._status = None
        self._status_code = None
        self._tracer = None

        try:
            trace_module = importlib.import_module("opentelemetry.trace")
            self._trace = trace_module
            self._status = getattr(trace_module, "Status", None)
            self._status_code = getattr(trace_module, "StatusCode", None)
            self._tracer = trace_module.get_tracer(service_name)
            self._available = True
        except Exception:
            self._available = False

    @property
    def enabled(self) -> bool:
        return self._available

    def start_span(self, name: str, **attributes: Any) -> Any:
        if not self._available or self._tracer is None:
            return None
        span = self._tracer.start_span(name)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        return span

    def end_span(self, span: Any, *, ok: bool, error: str | None = None) -> None:
        if not self._available or span is None:
            return
        if error:
            span.record_exception(RuntimeError(error))
        if self._status is not None and self._status_code is not None:
            code = self._status_code.OK if ok else self._status_code.ERROR
            span.set_status(self._status(code, error or ""))
        span.end()
