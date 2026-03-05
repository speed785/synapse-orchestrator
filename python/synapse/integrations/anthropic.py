"""
Anthropic Integration — auto-orchestrate Anthropic tool use.

Wraps the Anthropic client and transparently parallelises all tool
calls returned inside a ``tool_use`` content block.

Usage
-----
::

    import anthropic
    from synapse.integrations.anthropic import SynapseAnthropic

    client = SynapseAnthropic(
        anthropic_client=anthropic.AsyncAnthropic(),
        tools={
            "search_web":    search_web,
            "read_document": read_document,
            "summarize":     summarize,
        },
    )

    response, reports = await client.messages(
        model="claude-opus-4-5",
        max_tokens=4096,
        system="You are a research assistant.",
        messages=[{"role": "user", "content": "Research quantum computing."}],
        tools=anthropic_tool_schemas,
    )

    for report in reports:
        print(report)
"""

from __future__ import annotations

from typing import Any

from ..dependency_analyzer import ToolCall
from ..executor import AsyncToolFn, ExecutionReport
from ..orchestrator import Orchestrator


def _extract_tool_use_blocks(content: list[Any]) -> list[Any]:
    """Return only ``tool_use`` content blocks from a message."""
    return [block for block in content if getattr(block, "type", None) == "tool_use"]


def _anthropic_calls_to_synapse(tool_use_blocks: list[Any]) -> list[ToolCall]:
    """
    Convert Anthropic tool_use content blocks into Synapse ToolCall objects.
    """
    return [
        ToolCall(
            id=block.id,
            name=block.name,
            inputs=block.input if isinstance(block.input, dict) else {},
        )
        for block in tool_use_blocks
    ]


def _build_tool_result_content(report: ExecutionReport) -> list[dict[str, Any]]:
    """
    Build the ``tool_result`` content blocks list for the user-turn
    message that carries tool outputs back to Claude.
    """
    results = []
    for call_id, result in report.results.items():
        if result.status == "success":
            import json
            content = json.dumps(result.output) if not isinstance(result.output, str) else result.output
        else:
            import json
            content = json.dumps({"error": result.error, "status": result.status})

        results.append(
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content,
            }
        )
    return results


class SynapseAnthropic:
    """
    Wraps an ``anthropic.AsyncAnthropic`` client and executes all tool
    calls in parallel via Synapse.

    Parameters
    ----------
    anthropic_client:
        An ``anthropic.AsyncAnthropic`` instance.
    tools:
        Mapping from tool name → async callable.
    max_rounds:
        Maximum agentic rounds before giving up.
    orchestrator_kwargs:
        Keyword arguments forwarded to ``Orchestrator.__init__``.
    """

    def __init__(
        self,
        anthropic_client: Any,
        tools: dict[str, AsyncToolFn],
        *,
        max_rounds: int = 10,
        **orchestrator_kwargs: Any,
    ) -> None:
        self._client = anthropic_client
        self._orch = Orchestrator(tools, **orchestrator_kwargs)
        self.max_rounds = max_rounds

    async def messages(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, list[ExecutionReport]]:
        """
        Run the full agentic loop with parallel tool execution.

        Returns
        -------
        (final_message, reports):
            ``final_message`` is the last Message object where the model
            produced only text (``stop_reason == "end_turn"``).
            ``reports`` is one ExecutionReport per round of tool calls.
        """
        history = list(messages)
        all_reports: list[ExecutionReport] = []

        create_kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )
        if system is not None:
            create_kwargs["system"] = system

        for _ in range(self.max_rounds):
            response = await self._client.messages.create(
                messages=history,
                **create_kwargs,
            )

            tool_blocks = _extract_tool_use_blocks(response.content)

            if not tool_blocks or response.stop_reason == "end_turn":
                return response, all_reports

            # Add assistant message with tool use blocks to history
            history.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

            synapse_calls = _anthropic_calls_to_synapse(tool_blocks)
            report = await self._orch.run(synapse_calls)
            all_reports.append(report)

            tool_results = _build_tool_result_content(report)
            history.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        raise RuntimeError(
            f"Exceeded max_rounds={self.max_rounds} without a final response."
        )
