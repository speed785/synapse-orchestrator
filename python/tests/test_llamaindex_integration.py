import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from synapse.dependency_analyzer import ToolCall
from synapse.integrations.llamaindex import (
    SynapseFunctionCallingAgent,
    _build_async_tool_adapter,
    _to_tool_call,
)


class _AcallTool:
    async def acall(self, **kwargs: Any) -> Any:
        return kwargs["x"] + 1


class _CallableSyncTool:
    metadata: Any = None

    def __call__(self, **kwargs: Any) -> Any:
        return kwargs["x"] + 2


class _CallableAsyncTool:
    async def __call__(self, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        return kwargs["x"] + 3


class _CallMethodTool:
    def call(self, **kwargs: Any) -> Any:
        return kwargs["x"] + 4


class _CallMethodAsyncTool:
    async def _go(self, value: int) -> int:
        await asyncio.sleep(0)
        return value + 5

    def call(self, **kwargs: Any) -> Any:
        return self._go(kwargs["x"])


class _UnsupportedTool:
    pass


@pytest.mark.asyncio
async def test_build_async_tool_adapter_paths() -> None:
    assert await _build_async_tool_adapter(_AcallTool())(x=1) == 2
    assert await _build_async_tool_adapter(_CallableSyncTool())(x=1) == 3
    assert await _build_async_tool_adapter(_CallableAsyncTool())(x=1) == 4
    assert await _build_async_tool_adapter(_CallMethodTool())(x=1) == 5
    assert await _build_async_tool_adapter(_CallMethodAsyncTool())(x=1) == 6

    with pytest.raises(ValueError, match="Unsupported LlamaIndex tool type"):
        await _build_async_tool_adapter(_UnsupportedTool())(x=1)


def test_to_tool_call_supports_toolcall_dict_and_object() -> None:
    raw = ToolCall(id="a", name="tool", inputs={"x": 1})
    assert _to_tool_call(raw, 0) is raw

    as_dict = _to_tool_call({"id": "b", "name": "tool", "inputs": {"x": 2}}, 1)
    assert as_dict.id == "b"
    assert as_dict.inputs["x"] == 2

    obj = SimpleNamespace(id="c", tool_name="tool", kwargs={"x": 3}, depends_on=["a"])
    as_obj = _to_tool_call(obj, 2)
    assert as_obj.id == "c"
    assert as_obj.depends_on == ["a"]

    fallback = _to_tool_call(SimpleNamespace(name="tool", kwargs="invalid"), 3)
    assert fallback.id == "call_3"
    assert fallback.inputs == {}


class _FunctionCallingAgent:
    def __init__(self) -> None:
        self.callback_manager = "mgr"
        self.callbacks = ["cb"]
        self.llm = "llm"
        self.verbose = True
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        tool = _CallableSyncTool()
        tool.metadata = SimpleNamespace(name="plus_two")
        self.tools = [tool]

    async def achat(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return "achat"

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return "chat"

    async def astream_chat(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return "astream"

    def stream_chat(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return "stream"


@pytest.mark.asyncio
async def test_synapse_function_calling_agent_passthrough_and_parallel_batch() -> None:
    async def plus_one(x: int) -> int:
        return x + 1

    wrapped = SynapseFunctionCallingAgent(
        function_calling_agent=_FunctionCallingAgent(),
        tools={"plus_one": plus_one},
    )

    report = await wrapped.arun_tool_batch([
        {"id": "a", "name": "plus_one", "inputs": {"x": 1}},
        {"id": "b", "name": "plus_one", "inputs": {"x": 2}},
    ])
    assert report.results["a"].output == 2
    assert report.results["b"].output == 3
    assert report.parallel_calls == 2

    assert wrapped.callback_manager == "mgr"
    assert wrapped.callbacks == ["cb"]
    assert wrapped.llm == "llm"
    assert wrapped.verbose is True

    assert await wrapped.achat("x") == "achat"
    assert wrapped.chat("x") == "chat"
    assert await wrapped.astream_chat("x") == "astream"
    assert wrapped.stream_chat("x") == "stream"


def test_synapse_function_calling_agent_sync_batch_and_tool_autowiring() -> None:
    wrapped = SynapseFunctionCallingAgent(function_calling_agent=_FunctionCallingAgent())

    report = wrapped.run_tool_batch([
        {"id": "a", "name": "plus_two", "inputs": {"x": 1}},
    ])

    assert report.results["a"].output == 3
