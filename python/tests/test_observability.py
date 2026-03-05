import asyncio
import types
from typing import Any

import pytest

from synapse import Orchestrator, SynapseLogger, ToolCall  # pyright: ignore[reportImplicitRelativeImport]
from synapse.observability import ExecutionMetrics, LogEvent, OTelIntegration, _percentile  # pyright: ignore[reportImplicitRelativeImport]


def test_execution_metrics_export_prometheus() -> None:
    metrics = ExecutionMetrics()
    metrics.record_execution(speedup_ratio=2.0, parallelism=1.5)
    metrics.record_tool_call(latency_ms=10.0, failed=False)
    metrics.record_tool_call(latency_ms=20.0, failed=True)

    output = metrics.export_prometheus()

    assert "synapse_total_executions 1" in output
    assert "synapse_total_tool_calls 2" in output
    assert "synapse_tool_failure_rate 0.500000" in output
    assert 'synapse_latency_ms{quantile="0.95"}' in output


def test_percentile_empty_values_returns_zero() -> None:
    assert _percentile([], 0.5) == 0.0


def test_logger_json_disabled_and_event_payload_paths(capsys: pytest.CaptureFixture[str]) -> None:
    disabled = SynapseLogger(enabled=False)
    disabled.log("ignored")
    assert disabled.events == []

    logger = SynapseLogger(enabled=True, emit_json=True)
    logger.log("evt", stage_index=1, tool_id="t", latency_ms=1.5, parallel_count=2, extra="v")
    payload = logger.events[0].as_dict()

    assert payload["event"] == "evt"
    assert payload["tool_id"] == "t"
    assert payload["extra"] == "v"
    assert "evt" in capsys.readouterr().out
    assert "synapse_total_tool_calls" in logger.export_prometheus()


def test_log_event_as_dict_contains_standard_fields() -> None:
    event = LogEvent(event="x", timestamp=123.0, error="err", details={"k": "v"})
    payload = event.as_dict()

    assert payload["event"] == "x"
    assert payload["timestamp"] == 123.0
    assert payload["error"] == "err"
    assert payload["k"] == "v"


def test_otel_integration_success_and_noop_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Span:
        def __init__(self) -> None:
            self.attrs: dict[str, Any] = {}
            self.status: Any = None
            self.exceptions: list[Exception] = []
            self.ended = False

        def set_attribute(self, key: str, value: Any) -> None:
            self.attrs[key] = value

        def set_status(self, status: Any) -> None:
            self.status = status

        def record_exception(self, exc: Exception) -> None:
            self.exceptions.append(exc)

        def end(self) -> None:
            self.ended = True

    class _Tracer:
        def __init__(self) -> None:
            self.last_span: _Span | None = None

        def start_span(self, name: str) -> _Span:
            self.last_span = _Span()
            return self.last_span

    class _StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    class _Status:
        def __init__(self, code: str, description: str) -> None:
            self.code = code
            self.description = description

    tracer = _Tracer()
    trace_module = types.SimpleNamespace(
        get_tracer=lambda service_name: tracer,
        Status=_Status,
        StatusCode=_StatusCode,
    )

    def fake_import(name: str) -> Any:
        assert name == "opentelemetry.trace"
        return trace_module

    monkeypatch.setattr("synapse.observability.importlib.import_module", fake_import)
    otel = OTelIntegration(service_name="svc")

    assert otel.enabled is True
    span = otel.start_span("span", a=1, b=None)
    assert tracer.last_span is not None
    assert tracer.last_span.attrs == {"a": 1}
    otel.end_span(span, ok=False, error="boom")
    assert tracer.last_span.status.code == "ERROR"
    assert tracer.last_span.status.description == "boom"
    assert tracer.last_span.exceptions
    assert tracer.last_span.ended is True

    monkeypatch.setattr(
        "synapse.observability.importlib.import_module",
        lambda name: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    disabled = OTelIntegration()
    assert disabled.enabled is False
    assert disabled.start_span("ignored") is None
    disabled.end_span(None, ok=True)


@pytest.mark.asyncio
async def test_orchestrator_logs_dag_and_execution_events() -> None:
    async def alpha() -> str:
        await asyncio.sleep(0.005)
        return "a"

    async def beta() -> str:
        await asyncio.sleep(0.005)
        return "b"

    async def merge(left: str, right: str) -> str:
        await asyncio.sleep(0.005)
        return left + right

    logger = SynapseLogger(enabled=True, emit_json=False)
    orch = Orchestrator(
        tools={"alpha": alpha, "beta": beta, "merge": merge},
        logger=logger,
    )

    report = await orch.run(
        [
            ToolCall(id="a", name="alpha"),
            ToolCall(id="b", name="beta"),
            ToolCall(id="m", name="merge", inputs={"left": "$results.a", "right": "$results.b"}),
        ]
    )

    event_names = [event.event for event in logger.events]

    assert "dag_planned" in event_names
    assert "stage_started" in event_names
    assert "stage_completed" in event_names
    assert "tool_started" in event_names
    assert "tool_completed" in event_names
    assert "execution_complete" in event_names
    assert report.results["m"].output == "ab"
    assert logger.metrics.total_executions == 1
    assert logger.metrics.total_tool_calls == 3


@pytest.mark.asyncio
async def test_debug_mode_prints_dag_and_timeline(capsys: pytest.CaptureFixture[str]) -> None:
    async def noop() -> str:
        return "ok"

    async def echo(value: str) -> str:
        return value

    orch = Orchestrator(
        tools={"noop": noop, "echo": echo},
        debug=True,
    )

    await orch.run(
        [
            ToolCall(id="n", name="noop"),
            ToolCall(id="e", name="echo", inputs={"value": "$results.n"}),
        ]
    )

    out = capsys.readouterr().out
    assert "--- Synapse DAG ---" in out
    assert "n -> e" in out
    assert "--- Execution Timeline ---" in out
