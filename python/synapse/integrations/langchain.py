from __future__ import annotations

import inspect
from typing import Any

from ..dependency_analyzer import ToolCall
from ..executor import AsyncToolFn, ExecutionReport
from ..orchestrator import Orchestrator


def _build_async_tool_adapter(tool: Any) -> AsyncToolFn:
    async def _run_tool(**kwargs: Any) -> Any:
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(kwargs)
        if hasattr(tool, "arun"):
            return await tool.arun(**kwargs)
        if hasattr(tool, "invoke"):
            return tool.invoke(kwargs)
        if hasattr(tool, "run"):
            result = tool.run(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        raise ValueError(f"Unsupported LangChain tool type: {type(tool)!r}")

    return _run_tool


class SynapseAgentExecutor:
    def __init__(
        self,
        agent_executor: Any,
        *,
        tools: dict[str, AsyncToolFn] | None = None,
        **orchestrator_kwargs: Any,
    ) -> None:
        self.agent_executor = agent_executor

        resolved_tools = tools or {
            tool.name: _build_async_tool_adapter(tool)
            for tool in getattr(agent_executor, "tools", [])
        }

        self._orchestrator = Orchestrator(
            tools=resolved_tools,
            **orchestrator_kwargs,
        )

    @property
    def callbacks(self) -> Any:
        return getattr(self.agent_executor, "callbacks", None)

    @property
    def memory(self) -> Any:
        return getattr(self.agent_executor, "memory", None)

    @property
    def verbose(self) -> Any:
        return getattr(self.agent_executor, "verbose", None)

    async def arun_tool_batch(self, tool_calls: list[ToolCall | dict[str, Any]]) -> ExecutionReport:
        calls = [c if isinstance(c, ToolCall) else ToolCall(**c) for c in tool_calls]
        return await self._orchestrator.run(calls)

    def run_tool_batch(self, tool_calls: list[ToolCall | dict[str, Any]]) -> ExecutionReport:
        import asyncio

        calls = [c if isinstance(c, ToolCall) else ToolCall(**c) for c in tool_calls]
        return asyncio.run(self._orchestrator.run(calls))

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self.agent_executor.ainvoke(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.agent_executor.invoke(*args, **kwargs)
