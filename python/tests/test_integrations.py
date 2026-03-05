import json
import sys
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from synapse.integrations.anthropic import SynapseAnthropic  # pyright: ignore[reportImplicitRelativeImport]
from synapse.integrations.openai import SynapseOpenAI  # pyright: ignore[reportImplicitRelativeImport]


class _MockOpenAIClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **params: Any) -> Any:
        self.calls.append(params)
        if len(self.calls) == 1:
            tool_calls = [
                SimpleNamespace(
                    id="tc_ok",
                    function=SimpleNamespace(name="ok_tool", arguments=json.dumps({})),
                ),
                SimpleNamespace(
                    id="tc_bad",
                    function=SimpleNamespace(name="bad_tool", arguments=json.dumps({})),
                ),
            ]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))]
            )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))]
        )


class _MockAnthropicClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **params: Any) -> Any:
        self.calls.append(params)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="tool_use", id="a_ok", name="ok_tool", input={}),
                    SimpleNamespace(type="tool_use", id="a_bad", name="bad_tool", input={}),
                ],
                stop_reason="tool_use",
            )

        return SimpleNamespace(content=[], stop_reason="end_turn")


@pytest.mark.asyncio
async def test_openai_integration_includes_error_details_in_tool_messages() -> None:
    async def ok_tool() -> dict[str, str]:
        return {"status": "ok"}

    async def bad_tool() -> None:
        raise RuntimeError("boom")

    client = _MockOpenAIClient()
    synapse = SynapseOpenAI(
        openai_client=client,
        tools={"ok_tool": ok_tool, "bad_tool": bad_tool},
    )

    _, reports = await synapse.chat(
        model="gpt-test",
        messages=[{"role": "user", "content": "run tools"}],
        tools=[],
    )

    assert len(reports) == 1
    second_request_messages = client.calls[1]["messages"]
    tool_messages = [
        m
        for m in second_request_messages
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert len(tool_messages) == 2

    bad_message = next(m for m in tool_messages if m["tool_call_id"] == "tc_bad")
    bad_payload = json.loads(bad_message["content"])
    assert bad_payload["error"] == "boom"
    assert bad_payload["status"] == "error"


@pytest.mark.asyncio
async def test_anthropic_integration_includes_error_details_in_tool_results() -> None:
    async def ok_tool() -> dict[str, str]:
        return {"status": "ok"}

    async def bad_tool() -> None:
        raise RuntimeError("anthropic boom")

    client = _MockAnthropicClient()
    synapse = SynapseAnthropic(
        anthropic_client=client,
        tools={"ok_tool": ok_tool, "bad_tool": bad_tool},
    )

    _, reports = await synapse.messages(
        model="claude-test",
        max_tokens=256,
        messages=[{"role": "user", "content": "run tools"}],
        tools=[],
    )

    assert len(reports) == 1
    second_request_messages = client.calls[1]["messages"]
    user_turn = second_request_messages[-1]
    tool_results = user_turn["content"]
    bad_result = next(r for r in tool_results if r["tool_use_id"] == "a_bad")
    bad_payload = json.loads(bad_result["content"])
    assert bad_payload["error"] == "anthropic boom"
    assert bad_payload["status"] == "error"


@pytest.mark.asyncio
async def test_integrations_raise_on_retry_exhaustion_rounds() -> None:
    class _NeverEndingOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, **params: Any) -> Any:
            tool_calls = [
                SimpleNamespace(
                    id="tc",
                    function=SimpleNamespace(name="ok_tool", arguments=json.dumps({})),
                )
            ]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))]
            )

    class _NeverEndingAnthropicClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.messages = SimpleNamespace(create=self._create)

        async def _create(self, **params: Any) -> Any:
            self.calls.append(params)
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", id="a", name="ok_tool", input={})],
                stop_reason="tool_use",
            )

    async def ok_tool() -> str:
        return "ok"

    openai = SynapseOpenAI(
        openai_client=_NeverEndingOpenAIClient(),
        tools={"ok_tool": ok_tool},
        max_rounds=1,
    )
    anthropic_client = _NeverEndingAnthropicClient()
    anthropic = SynapseAnthropic(
        anthropic_client=anthropic_client,
        tools={"ok_tool": ok_tool},
        max_rounds=1,
    )

    with pytest.raises(RuntimeError, match="Exceeded max_rounds=1"):
        await openai.chat(model="gpt", messages=[{"role": "user", "content": "go"}], tools=[])

    with pytest.raises(RuntimeError, match="Exceeded max_rounds=1"):
        await anthropic.messages(
            model="claude",
            max_tokens=32,
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            system="sys",
        )
    assert anthropic_client.calls[0]["system"] == "sys"


def test_integrations_init_fallback_when_langchain_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    package_path = Path(__file__).resolve().parents[1] / "synapse" / "integrations" / "__init__.py"
    source = package_path.read_text(encoding="utf-8")

    real_import = builtins.__import__

    def fake_import(name: str, globals_obj: Any = None, locals_obj: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name == "langchain" and level == 1 and fromlist == ("SynapseAgentExecutor",):
            raise RuntimeError("langchain unavailable")
        if name == "llamaindex" and level == 1 and fromlist == ("SynapseFunctionCallingAgent",):
            raise RuntimeError("llamaindex unavailable")
        if name == "crewai" and level == 1 and fromlist == ("SynapseCrewTaskExecutor",):
            raise RuntimeError("crewai unavailable")
        return real_import(name, globals_obj, locals_obj, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = type(sys)("tmp_integrations_init")
    module.__package__ = "synapse.integrations"
    exec(compile(source, str(package_path), "exec"), module.__dict__)

    assert module.SynapseAgentExecutor is None
    assert module.SynapseFunctionCallingAgent is None
    assert module.SynapseCrewTaskExecutor is None
