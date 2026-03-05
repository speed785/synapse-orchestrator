from __future__ import annotations

import inspect
from typing import Any

from ..dependency_analyzer import ToolCall
from ..executor import AsyncToolFn, ExecutionReport
from ..orchestrator import Orchestrator


def _task_id(task: Any, index: int) -> str:
    if isinstance(task, dict):
        return str(task.get("id", task.get("task_id", f"task_{index}")))
    value = getattr(task, "id", None) or getattr(task, "task_id", None)
    return str(value or f"task_{index}")


def _task_name(task: Any) -> str:
    if isinstance(task, dict):
        return str(task.get("name", task.get("tool", "run_task")))
    value = getattr(task, "name", None) or getattr(task, "tool", None)
    return str(value or "run_task")


def _task_inputs(task: Any) -> dict[str, Any]:
    if isinstance(task, dict):
        raw = task.get("inputs", task.get("kwargs", task.get("input", {})))
        return raw if isinstance(raw, dict) else {}
    raw = getattr(task, "inputs", None) or getattr(task, "kwargs", None) or getattr(task, "input", None) or {}
    return raw if isinstance(raw, dict) else {}


def _task_dependencies(task: Any) -> list[str]:
    if isinstance(task, dict):
        deps = task.get("depends_on", [])
        context = task.get("context", [])
    else:
        deps = getattr(task, "depends_on", [])
        context = getattr(task, "context", [])

    resolved: list[str] = []
    for dep in deps if isinstance(deps, list) else []:
        if isinstance(dep, str):
            resolved.append(dep)
        else:
            dep_id = getattr(dep, "id", None) or getattr(dep, "task_id", None)
            if dep_id:
                resolved.append(str(dep_id))

    for dep in context if isinstance(context, list) else []:
        if isinstance(dep, str):
            resolved.append(dep)
        else:
            dep_id = getattr(dep, "id", None) or getattr(dep, "task_id", None)
            if dep_id:
                resolved.append(str(dep_id))

    seen: set[str] = set()
    unique: list[str] = []
    for dep in resolved:
        if dep not in seen:
            seen.add(dep)
            unique.append(dep)
    return unique


class SynapseCrewTaskExecutor:
    def __init__(
        self,
        *,
        task_executors: dict[str, AsyncToolFn] | None = None,
        task_runner: Any | None = None,
        **orchestrator_kwargs: Any,
    ) -> None:
        self._task_runner = task_runner

        async def _run_task(task: Any) -> Any:
            if self._task_runner is not None:
                result = self._task_runner(task=task)
                if inspect.isawaitable(result):
                    return await result
                return result
            if hasattr(task, "execute"):
                result = task.execute()
                if inspect.isawaitable(result):
                    return await result
                return result
            raise ValueError("Task runner is required when tasks do not implement execute().")

        tools = dict(task_executors or {})
        if "run_task" not in tools:
            tools["run_task"] = _run_task

        self._orchestrator = Orchestrator(tools=tools, **orchestrator_kwargs)

    def _to_calls(self, tasks: list[Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        task_by_id: dict[str, Any] = {}

        for index, task in enumerate(tasks):
            task_id = _task_id(task, index)
            task_by_id[task_id] = task

        for index, task in enumerate(tasks):
            task_id = _task_id(task, index)
            name = _task_name(task)
            inputs = _task_inputs(task)
            if name == "run_task":
                inputs = {"task": task_by_id[task_id]}
            calls.append(
                ToolCall(
                    id=task_id,
                    name=name,
                    inputs=inputs,
                    depends_on=_task_dependencies(task),
                )
            )
        return calls

    async def arun_tasks(self, tasks: list[Any]) -> ExecutionReport:
        calls = self._to_calls(tasks)
        return await self._orchestrator.run(calls)

    def run_tasks(self, tasks: list[Any]) -> ExecutionReport:
        import asyncio

        calls = self._to_calls(tasks)
        return asyncio.run(self._orchestrator.run(calls))
