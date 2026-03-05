import asyncio
from typing import Any

import pytest

from synapse.dependency_analyzer import ToolCall  # pyright: ignore[reportImplicitRelativeImport]
from synapse.integrations.langchain import SynapseAgentExecutor, _build_async_tool_adapter  # pyright: ignore[reportImplicitRelativeImport, reportPrivateUsage]


class _AInvokeTool:
    async def ainvoke(self, payload: dict[str, Any]) -> Any:
        return payload["x"] + 1


class _ARunTool:
    async def arun(self, **kwargs: Any) -> Any:
        return kwargs["x"] + 2


class _InvokeTool:
    def invoke(self, payload: dict[str, Any]) -> Any:
        return payload["x"] + 3


class _RunSyncTool:
    def run(self, **kwargs: Any) -> Any:
        return kwargs["x"] + 4


class _RunAsyncTool:
    async def _go(self, value: int) -> int:
        await asyncio.sleep(0)
        return value + 5

    def run(self, **kwargs: Any) -> Any:
        return self._go(kwargs["x"])


class _UnsupportedTool:
    pass


@pytest.mark.asyncio
async def test_build_async_tool_adapter_supports_all_known_shapes() -> None:
    assert await _build_async_tool_adapter(_AInvokeTool())(x=1) == 2
    assert await _build_async_tool_adapter(_ARunTool())(x=1) == 3
    assert await _build_async_tool_adapter(_InvokeTool())(x=1) == 4
    assert await _build_async_tool_adapter(_RunSyncTool())(x=1) == 5
    assert await _build_async_tool_adapter(_RunAsyncTool())(x=1) == 6

    with pytest.raises(ValueError, match="Unsupported LangChain tool type"):
        await _build_async_tool_adapter(_UnsupportedTool())(x=1)


class _AgentExecutor:
    def __init__(self) -> None:
        self.callbacks = ["cb"]
        self.memory = {"k": "v"}
        self.verbose = True
        self.tools = []
        self.invocations: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        self.invocations.append((args, kwargs))
        return "ainvoked"

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self.invocations.append((args, kwargs))
        return "invoked"


@pytest.mark.asyncio
async def test_synapse_agent_executor_passthrough_and_batches() -> None:
    async def plus_one(x: int) -> int:
        return x + 1

    agent = _AgentExecutor()
    synapse_executor = SynapseAgentExecutor(agent_executor=agent, tools={"plus_one": plus_one})

    report_async = await synapse_executor.arun_tool_batch(
        [ToolCall(id="a", name="plus_one", inputs={"x": 1})]
    )
    assert report_async.results["a"].output == 2

    assert synapse_executor.callbacks == ["cb"]
    assert synapse_executor.memory == {"k": "v"}
    assert synapse_executor.verbose is True

    assert await synapse_executor.ainvoke("x", y=1) == "ainvoked"
    assert synapse_executor.invoke("x", y=2) == "invoked"


def test_synapse_agent_executor_run_tool_batch_sync() -> None:
    async def plus_one(x: int) -> int:
        return x + 1

    agent = _AgentExecutor()
    synapse_executor = SynapseAgentExecutor(agent_executor=agent, tools={"plus_one": plus_one})

    report_sync = synapse_executor.run_tool_batch(
        [
            {"id": "b", "name": "plus_one", "inputs": {"x": 2}},
        ]
    )

    assert report_sync.results["b"].output == 3
