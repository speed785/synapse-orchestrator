"""
OpenAI Integration — auto-orchestrate OpenAI function/tool calls.

Drop-in wrapper around the OpenAI client that intercepts tool call
responses, builds a Synapse execution plan, and executes calls in
the optimal order before returning results back to the conversation.

Usage
-----
::

    from openai import AsyncOpenAI
    from synapse.integrations.openai import SynapseOpenAI

    client = SynapseOpenAI(
        openai_client=AsyncOpenAI(),
        tools={
            "get_weather":  get_weather,
            "get_forecast": get_forecast,
            "send_alert":   send_alert,
        },
    )

    response, report = await client.chat(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather and 5-day forecast for NYC?"}],
        tools=openai_tool_schemas,
    )

    print(report)   # shows parallelism, speedup, per-call timing
"""

from __future__ import annotations

import json
from typing import Any

from ..dependency_analyzer import ToolCall
from ..executor import AsyncToolFn, ExecutionReport
from ..orchestrator import Orchestrator


def _openai_calls_to_synapse(tool_calls: list[Any]) -> list[ToolCall]:
    """
    Convert a list of OpenAI ChatCompletionMessageToolCall objects into
    Synapse ToolCall objects.

    Synapse performs its own dependency detection via $results.* references
    in inputs — if the model embeds such references in its arguments,
    they will be honoured automatically.
    """
    return [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            inputs=json.loads(tc.function.arguments),
        )
        for tc in tool_calls
    ]


def _build_tool_messages(report: ExecutionReport) -> list[dict[str, Any]]:
    """
    Build the list of role=tool messages to return to the model after
    all tool calls have been executed.
    """
    messages = []
    for call_id, result in report.results.items():
        content = (
            json.dumps(result.output)
            if result.output is not None
            else json.dumps({"error": result.error})
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }
        )
    return messages


class SynapseOpenAI:
    """
    Wraps an AsyncOpenAI client and intercepts tool call responses so
    Synapse can execute them in parallel.

    Parameters
    ----------
    openai_client:
        An ``openai.AsyncOpenAI`` instance (caller owns lifecycle).
    tools:
        Mapping from tool name → async callable.
    max_rounds:
        Maximum agentic rounds (model call → tool execution → …).
    orchestrator_kwargs:
        Keyword arguments forwarded to ``Orchestrator.__init__``.
    """

    def __init__(
        self,
        openai_client: Any,
        tools: dict[str, AsyncToolFn],
        *,
        max_rounds: int = 10,
        **orchestrator_kwargs: Any,
    ) -> None:
        self._client = openai_client
        self._orch = Orchestrator(tools, **orchestrator_kwargs)
        self.max_rounds = max_rounds

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[Any, list[ExecutionReport]]:
        """
        Run a full agentic loop with automatic parallel tool execution.

        Returns
        -------
        (final_response, reports):
            ``final_response`` is the last ChatCompletion where the model
            produced text rather than tool calls.
            ``reports`` is one ExecutionReport per round of tool calls.
        """
        history = list(messages)
        all_reports: list[ExecutionReport] = []

        for _ in range(self.max_rounds):
            response = await self._client.chat.completions.create(
                model=model,
                messages=history,
                tools=tools,
                **kwargs,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                # Model is done — return the final response
                return response, all_reports

            history.append(message)

            synapse_calls = _openai_calls_to_synapse(message.tool_calls)
            report = await self._orch.run(synapse_calls)
            all_reports.append(report)

            tool_messages = _build_tool_messages(report)
            history.extend(tool_messages)

        raise RuntimeError(
            f"Exceeded max_rounds={self.max_rounds} without a final response."
        )
