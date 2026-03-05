"""
Executor — runs an ExecutionPlan with async concurrency.

Features
--------
* Fully async via asyncio.gather — zero threads needed.
* Per-call timeout and retry with exponential back-off.
* Concurrency cap via asyncio.Semaphore (don't overwhelm rate-limited APIs).
* $results.<id> placeholder substitution so later stages receive real values.
* Rich execution report with per-call timing and status.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .dependency_analyzer import ToolCall
from .planner import ExecutionPlan

# ── Type alias for an async tool implementation ───────────────────────────────
AsyncToolFn = Callable[..., Coroutine[Any, Any, Any]]

_REF_PATTERN = re.compile(r"\$results\.([a-zA-Z0-9_\-\.\[\]]+)")


# ── Per-call result ────────────────────────────────────────────────────────────

@dataclass
class CallResult:
    call_id: str
    tool_name: str
    status: str          # "success" | "error" | "timeout"
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    attempts: int = 1


# ── Full execution report ──────────────────────────────────────────────────────

@dataclass
class ExecutionReport:
    results: dict[str, CallResult]
    total_duration_ms: float
    stages_run: int
    parallel_calls: int   # calls that ran concurrently with at least one other
    sequential_calls: int
    speedup_estimate: float  # sum(durations) / wall_clock (≥1 means parallelism helped)

    def __repr__(self) -> str:
        lines = [
            "── Synapse Execution Report ──────────────────────────────────",
            f"  Wall clock : {self.total_duration_ms:.1f} ms",
            f"  Stages     : {self.stages_run}",
            f"  Parallel   : {self.parallel_calls} calls",
            f"  Sequential : {self.sequential_calls} calls",
            f"  Speedup    : {self.speedup_estimate:.2f}x",
            "  Results:",
        ]
        for r in self.results.values():
            icon = "✓" if r.status == "success" else "✗"
            lines.append(
                f"    {icon} [{r.call_id}] {r.tool_name} "
                f"— {r.status} in {r.duration_ms:.1f} ms"
                + (f" (attempt {r.attempts})" if r.attempts > 1 else "")
            )
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)


# ── Placeholder resolver ───────────────────────────────────────────────────────

def _resolve(value: Any, results: dict[str, Any]) -> Any:
    """Replace $results.<id>[.<path>] placeholders with actual values."""
    if isinstance(value, str):
        # Single full-match replacement preserves non-string types
        full = _REF_PATTERN.fullmatch(value)
        if full:
            ref_id, path = _split_ref(full.group(1), results)
            return _get_nested(results.get(ref_id), path)
        # Inline replacement (always produces a string)
        def replacer(m: re.Match[str]) -> str:
            ref_id, path = _split_ref(m.group(1), results)
            v = _get_nested(results.get(ref_id), path)
            return str(v) if v is not None else m.group(0)
        return _REF_PATTERN.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve(v, results) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve(item, results) for item in value]
    return value


def _split_ref(raw_ref: str, results: dict[str, Any]) -> tuple[str, str | None]:
    parts = raw_ref.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in results:
            suffix = ".".join(parts[i:])
            return candidate, (suffix or None)

    return raw_ref, None


def _get_nested(obj: Any, path: str | None) -> Any:
    if obj is None or path is None:
        return obj
    for part in path.split("."):
        if part.endswith("]"):
            key, _, idx = part.rstrip("]").partition("[")
            obj = obj[key][int(idx)]
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            break
    return obj


# ── Executor ───────────────────────────────────────────────────────────────────

class Executor:
    """
    Executes an ExecutionPlan against a registry of async tool functions.

    Parameters
    ----------
    tools:
        Mapping from tool name → async callable.
    max_concurrency:
        Maximum simultaneous in-flight tool calls.  Defaults to 16.
    default_timeout:
        Seconds before a call is cancelled if the ToolCall has no timeout.
    default_retries:
        Retry count if the ToolCall has no retries set.
    on_call_start / on_call_end:
        Optional hooks for observability / progress reporting.
    """

    def __init__(
        self,
        tools: dict[str, AsyncToolFn],
        *,
        max_concurrency: int = 16,
        default_timeout: float = 30.0,
        default_retries: int = 0,
        on_call_start: Callable[[ToolCall], None] | None = None,
        on_call_end: Callable[[CallResult], None] | None = None,
    ) -> None:
        self.tools = tools
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.default_timeout = default_timeout
        self.default_retries = default_retries
        self.on_call_start = on_call_start
        self.on_call_end = on_call_end

    async def execute(self, plan: ExecutionPlan) -> ExecutionReport:
        """Run the full plan and return a detailed ExecutionReport."""
        accumulated_results: dict[str, Any] = {}   # call_id → raw output
        call_results: dict[str, CallResult] = {}

        wall_start = time.perf_counter()
        sum_call_duration = 0.0
        parallel_calls = 0
        sequential_calls = 0

        for stage in plan.stages:
            is_parallel = len(stage.calls) > 1
            if is_parallel:
                parallel_calls += len(stage.calls)
            else:
                sequential_calls += len(stage.calls)

            stage_results = await asyncio.gather(
                *[
                    self._run_call(call, accumulated_results)
                    for call in stage.calls
                ],
                return_exceptions=False,
            )

            for result in stage_results:
                call_results[result.call_id] = result
                accumulated_results[result.call_id] = result.output
                sum_call_duration += result.duration_ms

        wall_ms = (time.perf_counter() - wall_start) * 1000
        speedup = sum_call_duration / wall_ms if wall_ms > 0 else 1.0

        return ExecutionReport(
            results=call_results,
            total_duration_ms=wall_ms,
            stages_run=len(plan.stages),
            parallel_calls=parallel_calls,
            sequential_calls=sequential_calls,
            speedup_estimate=speedup,
        )

    async def _run_call(
        self, call: ToolCall, accumulated_results: dict[str, Any]
    ) -> CallResult:
        tool_fn = self.tools.get(call.name)
        if tool_fn is None:
            return CallResult(
                call_id=call.id,
                tool_name=call.name,
                status="error",
                error=f"No tool registered for '{call.name}'",
            )

        resolved_inputs = _resolve(call.inputs, accumulated_results)
        timeout = call.timeout if call.timeout is not None else self.default_timeout
        max_attempts = 1 + (call.retries if call.retries else self.default_retries)

        last_error: str = ""
        last_status: str = "error"
        last_duration_ms: float = 0.0
        for attempt in range(1, max_attempts + 1):
            if self.on_call_start:
                self.on_call_start(call)
            t0 = time.perf_counter()
            try:
                async with self._semaphore:
                    output = await asyncio.wait_for(
                        tool_fn(**resolved_inputs), timeout=timeout
                    )
                duration_ms = (time.perf_counter() - t0) * 1000
                result = CallResult(
                    call_id=call.id,
                    tool_name=call.name,
                    status="success",
                    output=output,
                    duration_ms=duration_ms,
                    attempts=attempt,
                )
                if self.on_call_end:
                    self.on_call_end(result)
                return result
            except asyncio.TimeoutError:
                last_duration_ms = (time.perf_counter() - t0) * 1000
                last_error = f"Timed out after {timeout}s"
                last_status = "timeout"
            except Exception as exc:  # noqa: BLE001
                last_duration_ms = (time.perf_counter() - t0) * 1000
                last_error = str(exc)
                last_status = "error"

            if attempt < max_attempts:
                await asyncio.sleep(0.1 * (2 ** (attempt - 1)))  # back-off

        result = CallResult(
            call_id=call.id,
            tool_name=call.name,
            status=last_status,
            error=last_error,
            duration_ms=last_duration_ms,
            attempts=max_attempts,
        )
        if self.on_call_end:
            self.on_call_end(result)
        return result
