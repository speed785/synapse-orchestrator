import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from synapse.integrations.crewai import (
    SynapseCrewTaskExecutor,
    _task_dependencies,
    _task_id,
    _task_inputs,
    _task_name,
)


def test_task_helpers_cover_dict_and_object_variants() -> None:
    assert _task_id({"id": "a"}, 0) == "a"
    assert _task_id(SimpleNamespace(task_id="b"), 1) == "b"
    assert _task_name({"name": "task"}) == "task"
    assert _task_name(SimpleNamespace(tool="runner")) == "runner"
    assert _task_inputs({"inputs": {"x": 1}}) == {"x": 1}
    assert _task_inputs(SimpleNamespace(kwargs={"y": 2})) == {"y": 2}
    assert _task_inputs(SimpleNamespace(kwargs="bad")) == {}


def test_task_dependencies_normalizes_and_deduplicates() -> None:
    dep_obj = SimpleNamespace(id="dep_obj")
    context_obj = SimpleNamespace(task_id="ctx_obj")

    deps = _task_dependencies(
        {
            "depends_on": ["dep_a", dep_obj, "dep_a"],
            "context": ["ctx_a", context_obj, "ctx_a"],
        }
    )

    assert deps == ["dep_a", "dep_obj", "ctx_a", "ctx_obj"]


@pytest.mark.asyncio
async def test_synapse_crewai_executor_parallel_with_named_executors() -> None:
    async def plus_one(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    executor = SynapseCrewTaskExecutor(task_executors={"plus_one": plus_one})

    report = await executor.arun_tasks(
        [
            {"id": "a", "name": "plus_one", "inputs": {"x": 1}},
            {"id": "b", "name": "plus_one", "inputs": {"x": 2}},
            {
                "id": "c",
                "name": "plus_one",
                "inputs": {"x": "$results.a"},
                "depends_on": ["a"],
            },
        ]
    )

    assert report.results["a"].output == 2
    assert report.results["b"].output == 3
    assert report.results["c"].output == 3
    assert report.parallel_calls == 2


class _ExecutableSyncTask:
    id = "sync"

    def execute(self) -> str:
        return "sync-result"


class _ExecutableAsyncTask:
    id = "async"

    async def execute(self) -> str:
        await asyncio.sleep(0)
        return "async-result"


def test_synapse_crewai_executor_sync_run_tasks_and_execute_fallbacks() -> None:
    executor = SynapseCrewTaskExecutor()
    report = executor.run_tasks([_ExecutableSyncTask(), _ExecutableAsyncTask()])
    assert report.results["sync"].output == "sync-result"
    assert report.results["async"].output == "async-result"


@pytest.mark.asyncio
async def test_synapse_crewai_executor_task_runner_modes_and_error_path() -> None:
    async def async_runner(*, task: Any) -> str:
        await asyncio.sleep(0)
        return f"async-{getattr(task, 'id', 'x')}"

    sync_executor = SynapseCrewTaskExecutor(task_runner=lambda task: f"sync-{task.id}")
    async_executor = SynapseCrewTaskExecutor(task_runner=async_runner)
    default_executor = SynapseCrewTaskExecutor()

    sync_report = await sync_executor.arun_tasks([SimpleNamespace(id="s1")])
    async_report = await async_executor.arun_tasks([SimpleNamespace(id="a1")])
    error_report = await default_executor.arun_tasks([SimpleNamespace(id="e1")])

    assert sync_report.results["s1"].output == "sync-s1"
    assert async_report.results["a1"].output == "async-a1"
    assert error_report.results["e1"].status == "error"
    assert "Task runner is required" in (error_report.results["e1"].error or "")
