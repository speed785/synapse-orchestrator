import asyncio

import pytest

from synapse import Orchestrator, SynapseLogger, ToolCall  # pyright: ignore[reportImplicitRelativeImport]
from synapse.observability import ExecutionMetrics  # pyright: ignore[reportImplicitRelativeImport]


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

    orch = Orchestrator(
        tools={"noop": noop},
        debug=True,
    )

    await orch.run([ToolCall(id="n", name="noop")])

    out = capsys.readouterr().out
    assert "--- Synapse DAG ---" in out
    assert "--- Execution Timeline ---" in out
