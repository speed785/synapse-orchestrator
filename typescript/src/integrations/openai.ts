/**
 * OpenAI Integration — auto-orchestrate OpenAI tool calls via Synapse.
 *
 * @example
 * ```ts
 * import OpenAI from "openai";
 * import { SynapseOpenAI } from "synapse-orchestrator/integrations/openai";
 *
 * const client = new SynapseOpenAI({
 *   openaiClient: new OpenAI(),
 *   tools: {
 *     getWeather:  getWeatherFn,
 *     getForecast: getForecastFn,
 *     sendAlert:   sendAlertFn,
 *   },
 * });
 *
 * const [response, reports] = await client.chat({
 *   model: "gpt-4o",
 *   messages: [{ role: "user", content: "What's the weather in NYC?" }],
 *   tools: openAIToolSchemas,
 * });
 * ```
 */

import { Orchestrator } from "../orchestrator.js";
import type { AsyncToolFn, ExecutionReport } from "../executor.js";
import type { ToolCall } from "../dependencyAnalyzer.js";
import type { OrchestratorOptions } from "../orchestrator.js";

// ── Minimal OpenAI type shims (avoids a hard dep on the openai package) ───────

interface OpenAIToolCall {
  id: string;
  function: { name: string; arguments: string };
}

interface OpenAIMessage {
  tool_calls?: OpenAIToolCall[];
  [key: string]: unknown;
}

interface OpenAIChoice {
  message: OpenAIMessage;
}

interface OpenAIResponse {
  choices: OpenAIChoice[];
}

interface OpenAIClient {
  chat: {
    completions: {
      create(params: Record<string, unknown>): Promise<OpenAIResponse>;
    };
  };
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function openAICallsToSynapse(toolCalls: OpenAIToolCall[]): ToolCall[] {
  return toolCalls.map((tc) => ({
    id: tc.id,
    name: tc.function.name,
    inputs: JSON.parse(tc.function.arguments) as Record<string, unknown>,
  }));
}

function buildToolMessages(report: ExecutionReport): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = [];
  for (const [callId, result] of report.results) {
    messages.push({
      role: "tool",
      tool_call_id: callId,
      content:
        result.status === "success"
          ? JSON.stringify(result.output)
          : JSON.stringify({ error: result.error ?? "Unknown tool error", status: result.status }),
    });
  }
  return messages;
}

// ── Main class ─────────────────────────────────────────────────────────────────

export interface SynapseOpenAIOptions extends Omit<OrchestratorOptions, "tools"> {
  openaiClient: OpenAIClient;
  tools: Record<string, AsyncToolFn>;
  maxRounds?: number;
}

export class SynapseOpenAI {
  private client: OpenAIClient;
  private orch: Orchestrator;
  private maxRounds: number;

  constructor(options: SynapseOpenAIOptions) {
    this.client = options.openaiClient;
    this.orch = new Orchestrator({ ...options });
    this.maxRounds = options.maxRounds ?? 10;
  }

  async chat(params: {
    model: string;
    messages: Array<Record<string, unknown>>;
    tools: Array<Record<string, unknown>>;
    [key: string]: unknown;
  }): Promise<[OpenAIResponse, ExecutionReport[]]> {
    const { model, messages, tools, ...rest } = params;
    const history = [...messages];
    const allReports: ExecutionReport[] = [];

    for (let round = 0; round < this.maxRounds; round++) {
      const response = await this.client.chat.completions.create({
        model,
        messages: history,
        tools,
        ...rest,
      });

      const message = response.choices[0].message;

      if (!message.tool_calls || message.tool_calls.length === 0) {
        return [response, allReports];
      }

      history.push(message as Record<string, unknown>);

      const synapseCalls = openAICallsToSynapse(message.tool_calls);
      const report = await this.orch.run(synapseCalls);
      allReports.push(report);

      const toolMessages = buildToolMessages(report);
      history.push(...toolMessages);
    }

    throw new Error(`Exceeded maxRounds=${this.maxRounds} without a final response.`);
  }
}
