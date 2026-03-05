import pytest

from synapse import Orchestrator, ToolCall  # pyright: ignore[reportImplicitRelativeImport]


def test_orchestrator_plan_method() -> None:
    async def source() -> str:
        return "x"

    async def sink(value: str) -> str:
        return value

    orch = Orchestrator(tools={"source": source, "sink": sink})
    plan = orch.plan(
        [
            ToolCall(id="s", name="source"),
            ToolCall(id="k", name="sink", inputs={"value": "$results.s"}),
        ]
    )

    assert len(plan.stages) == 2
    assert [call.id for call in plan.stages[0].calls] == ["s"]
    assert [call.id for call in plan.stages[1].calls] == ["k"]


@pytest.mark.asyncio
async def test_orchestrator_run_raw() -> None:
    async def add_one(x: int) -> int:
        return x + 1

    orch = Orchestrator(tools={"add_one": add_one})
    report = await orch.run_raw([{"id": "a", "name": "add_one", "inputs": {"x": 1}}])

    assert report.results["a"].status == "success"
    assert report.results["a"].output == 2
