import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from synapse import Orchestrator, ToolCall  # pyright: ignore[reportImplicitRelativeImport]
from synapse.dependency_analyzer import DependencyAnalyzer  # pyright: ignore[reportImplicitRelativeImport]
from synapse.executor import Executor, _resolve, _split_ref  # pyright: ignore[reportImplicitRelativeImport, reportPrivateUsage]
from synapse.observability import SynapseLogger  # pyright: ignore[reportImplicitRelativeImport]
from synapse.planner import Planner  # pyright: ignore[reportImplicitRelativeImport]


@pytest.mark.asyncio
async def test_executor_runs_independent_calls_in_parallel() -> None:
    async def slow_a() -> str:
        await asyncio.sleep(0.12)
        return "a"

    async def slow_b() -> str:
        await asyncio.sleep(0.12)
        return "b"

    async def combine(left: str, right: str) -> str:
        await asyncio.sleep(0.01)
        return left + right

    orchestrator = Orchestrator(
        tools={"slow_a": slow_a, "slow_b": slow_b, "combine": combine},
    )

    report = await orchestrator.run(
        [
            ToolCall(id="a", name="slow_a"),
            ToolCall(id="b", name="slow_b"),
            ToolCall(id="c", name="combine", inputs={"left": "$results.a", "right": "$results.b"}),
        ]
    )

    assert report.results["c"].output == "ab"
    assert report.total_duration_ms < 220
    assert report.parallel_calls == 2


@pytest.mark.asyncio
async def test_executor_honors_timeout() -> None:
    async def too_slow() -> str:
        await asyncio.sleep(0.05)
        return "done"

    orchestrator = Orchestrator(tools={"too_slow": too_slow}, default_timeout=0.01)

    report = await orchestrator.run([ToolCall(id="t", name="too_slow")])

    result = report.results["t"]
    assert result.status == "timeout"
    assert "Timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_executor_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient")
        return "ok"

    orchestrator = Orchestrator(tools={"flaky": flaky}, default_retries=1)

    report = await orchestrator.run([ToolCall(id="f", name="flaky")])

    result = report.results["f"]
    assert result.status == "success"
    assert result.output == "ok"
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_executor_resolves_placeholders_for_dotted_tool_ids() -> None:
    async def fetch_user() -> dict[str, str]:
        return {"email": "a@example.com"}

    async def send_email(to: str) -> str:
        return to

    orchestrator = Orchestrator(
        tools={"fetch_user": fetch_user, "send_email": send_email},
    )

    report = await orchestrator.run(
        [
            ToolCall(id="fetch.user", name="fetch_user"),
            ToolCall(id="email", name="send_email", inputs={"to": "$results.fetch.user.email"}),
        ]
    )

    assert report.results["email"].status == "success"
    assert report.results["email"].output == "a@example.com"


def test_executor_resolve_supports_nested_paths_lists_and_objects() -> None:
    results = {
        "id": {
            "nested": {
                "path": "value",
                "arr": [{"name": "first"}],
            }
        },
        "obj": SimpleNamespace(attr="obj-value"),
    }

    payload = {
        "inline": "prefix-$results.id.nested.path",
        "items": [
            "$results.id.nested.arr[0].name",
            "$results.obj.attr",
            "$results.id.nested.missing.leaf",
        ],
        "unknown_inline": "x-$results.unknown.path",
    }
    resolved = _resolve(payload, results)

    assert resolved["inline"] == "prefix-value"
    assert resolved["items"][0] == "first"
    assert resolved["items"][1] == "obj-value"
    assert resolved["items"][2] is None
    assert resolved["unknown_inline"] == "x-$results.unknown.path"
    assert _split_ref("missing.field", results) == ("missing.field", None)


class _FakeOTel:
    def __init__(self) -> None:
        self.started: list[tuple[str, dict[str, object]]] = []
        self.ended: list[tuple[bool, str | None]] = []

    def start_span(self, name: str, **attributes: object) -> object:
        self.started.append((name, attributes))
        return object()

    def end_span(self, span: object, *, ok: bool, error: str | None = None) -> None:
        assert span is not None
        self.ended.append((ok, error))


@pytest.mark.asyncio
async def test_executor_handles_missing_tool_and_report_repr() -> None:
    logger = SynapseLogger(enabled=True)
    analyzer = DependencyAnalyzer()
    planner = Planner()
    graph = analyzer.analyze([ToolCall(id="missing", name="not_registered")])
    plan = planner.plan(graph)

    report = await Executor(tools={}, logger=logger).execute(plan)

    assert report.results["missing"].status == "error"
    assert "No tool registered" in (report.results["missing"].error or "")
    report_repr = repr(report)
    assert "Synapse Execution Report" in report_repr
    assert "[missing]" in report_repr
    assert "" not in report_repr
    assert any(event.event == "tool_failed" for event in logger.events)


@pytest.mark.asyncio
async def test_executor_timeout_retry_exhaustion_and_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = SynapseLogger(enabled=True)
    otel = _FakeOTel()
    starts: list[str] = []
    ends: list[str] = []

    async def slow_tool() -> str:
        return "never"

    async def fake_wait_for(coro: Any, timeout: float) -> object:
        if hasattr(coro, "close"):
            coro.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    analyzer = DependencyAnalyzer()
    planner = Planner()
    graph = analyzer.analyze([ToolCall(id="t", name="slow", retries=1)])
    plan = planner.plan(graph)
    executor = Executor(
        tools={"slow": slow_tool},
        default_timeout=0.01,
        logger=logger,
        otel=cast(Any, otel),
        on_call_start=lambda call: starts.append(call.id),
        on_call_end=lambda result: ends.append(result.call_id),
    )

    report = await executor.execute(plan)

    result = report.results["t"]
    assert result.status == "timeout"
    assert result.attempts == 2
    assert starts == ["t", "t"]
    assert ends == ["t"]
    assert any(ok is False for ok, _ in otel.ended)


@pytest.mark.asyncio
async def test_executor_otel_end_span_on_success_and_exception() -> None:
    logger = SynapseLogger(enabled=True)
    otel = _FakeOTel()
    ended_calls: list[str] = []

    async def ok_tool() -> str:
        return "ok"

    async def bad_tool() -> str:
        raise RuntimeError("boom")

    graph_ok = DependencyAnalyzer().analyze([ToolCall(id="ok", name="ok")])
    plan_ok = Planner().plan(graph_ok)
    report_ok = await Executor(
        tools={"ok": ok_tool},
        logger=logger,
        otel=cast(Any, otel),
        on_call_end=lambda result: ended_calls.append(result.call_id),
    ).execute(plan_ok)
    assert report_ok.results["ok"].status == "success"
    assert ended_calls == ["ok"]
    assert any(ok is True and error is None for ok, error in otel.ended)

    analyzer = DependencyAnalyzer()
    planner = Planner()
    graph = analyzer.analyze([ToolCall(id="bad", name="bad")])
    plan = planner.plan(graph)
    report_bad = await Executor(tools={"bad": bad_tool}, logger=logger, otel=cast(Any, otel)).execute(plan)

    assert report_bad.results["bad"].status == "error"
    assert any(ok is False and error == "boom" for ok, error in otel.ended)
