import asyncio

import pytest

from synapse import Orchestrator, ToolCall


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
