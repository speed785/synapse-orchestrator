from __future__ import annotations

import inspect
from typing import Any

from ..dependency_analyzer import ToolCall
from ..executor import AsyncToolFn, ExecutionReport
from ..orchestrator import Orchestrator


def _build_async_tool_adapter(tool: Any) -> AsyncToolFn:
    async def _run_tool(**kwargs: Any) -> Any:
        if hasattr(tool, "acall"):
            return await tool.acall(**kwargs)
        if hasattr(tool, "__call__"):
            result = tool(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        if hasattr(tool, "call"):
            result = tool.call(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        raise ValueError(f"Unsupported LlamaIndex tool type: {type(tool)!r}")

    return _run_tool


def _to_tool_call(call: ToolCall | dict[str, Any] | Any, index: int) -> ToolCall:
    if isinstance(call, ToolCall):
        return call
    if isinstance(call, dict):
        return ToolCall(
            id=str(call.get("id", f"call_{index}")),
            name=str(call.get("name", call.get("tool_name", ""))),
            inputs=call.get("inputs", call.get("kwargs", {})) or {},
            depends_on=call.get("depends_on", []) or [],
        )

    tool_name = getattr(call, "tool_name", None) or getattr(call, "name", "")
    call_id = getattr(call, "id", f"call_{index}")
    kwargs = getattr(call, "kwargs", None) or getattr(call, "input", None) or {}
    depends_on = getattr(call, "depends_on", []) or []

    return ToolCall(
        id=str(call_id),
        name=str(tool_name),
        inputs=kwargs if isinstance(kwargs, dict) else {},
        depends_on=list(depends_on) if isinstance(depends_on, list) else [],
    )


class SynapseFunctionCallingAgent:
    def __init__(
        self,
        function_calling_agent: Any,
        *,
        tools: dict[str, AsyncToolFn] | None = None,
        **orchestrator_kwargs: Any,
    ) -> None:
        self.function_calling_agent = function_calling_agent

        resolved_tools = tools or {
            str(getattr(tool, "metadata", tool).name): _build_async_tool_adapter(tool)
            for tool in getattr(function_calling_agent, "tools", [])
        }

        self._orchestrator = Orchestrator(tools=resolved_tools, **orchestrator_kwargs)

    @property
    def callback_manager(self) -> Any:
        return getattr(self.function_calling_agent, "callback_manager", None)

    @property
    def callbacks(self) -> Any:
        return getattr(self.function_calling_agent, "callbacks", None)

    @property
    def llm(self) -> Any:
        return getattr(self.function_calling_agent, "llm", None)

    @property
    def verbose(self) -> Any:
        return getattr(self.function_calling_agent, "verbose", None)

    async def arun_tool_batch(self, tool_calls: list[ToolCall | dict[str, Any] | Any]) -> ExecutionReport:
        calls = [_to_tool_call(call, index) for index, call in enumerate(tool_calls)]
        return await self._orchestrator.run(calls)

    def run_tool_batch(self, tool_calls: list[ToolCall | dict[str, Any] | Any]) -> ExecutionReport:
        import asyncio

        calls = [_to_tool_call(call, index) for index, call in enumerate(tool_calls)]
        return asyncio.run(self._orchestrator.run(calls))

    async def achat(self, *args: Any, **kwargs: Any) -> Any:
        return await self.function_calling_agent.achat(*args, **kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        return self.function_calling_agent.chat(*args, **kwargs)

    async def astream_chat(self, *args: Any, **kwargs: Any) -> Any:
        return await self.function_calling_agent.astream_chat(*args, **kwargs)

    def stream_chat(self, *args: Any, **kwargs: Any) -> Any:
        return self.function_calling_agent.stream_chat(*args, **kwargs)
