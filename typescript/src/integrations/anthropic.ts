/**
 * Anthropic Integration — auto-orchestrate Anthropic tool use via Synapse.
 *
 * @example
 * ```ts
 * import Anthropic from "@anthropic-ai/sdk";
 * import { SynapseAnthropic } from "synapse-orchestrator/integrations/anthropic";
 *
 * const client = new SynapseAnthropic({
 *   anthropicClient: new Anthropic(),
 *   tools: {
 *     searchWeb:    searchWebFn,
 *     readDocument: readDocumentFn,
 *     summarize:    summarizeFn,
 *   },
 * });
 *
 * const [response, reports] = await client.messages({
 *   model: "claude-opus-4-5",
 *   maxTokens: 4096,
 *   messages: [{ role: "user", content: "Research quantum computing." }],
 *   tools: anthropicToolSchemas,
 * });
 * ```
 */

import { Orchestrator } from "../orchestrator.js";
import type { AsyncToolFn, ExecutionReport } from "../executor.js";
import type { ToolCall } from "../dependencyAnalyzer.js";
import type { OrchestratorOptions } from "../orchestrator.js";

// ── Minimal Anthropic type shims ──────────────────────────────────────────────

interface ContentBlock {
  type: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  [key: string]: unknown;
}

interface AnthropicMessage {
  id: string;
  type: string;
  role: string;
  content: ContentBlock[];
  stop_reason: string | null;
  [key: string]: unknown;
}

interface AnthropicClient {
  messages: {
    create(params: Record<string, unknown>): Promise<AnthropicMessage>;
  };
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function extractToolUseBlocks(content: ContentBlock[]): ContentBlock[] {
  return content.filter((block) => block.type === "tool_use");
}

function anthropicCallsToSynapse(blocks: ContentBlock[]): ToolCall[] {
  return blocks.map((block) => ({
    id: block.id!,
    name: block.name!,
    inputs: block.input ?? {},
  }));
}

function buildToolResultContent(report: ExecutionReport): ContentBlock[] {
  const results: ContentBlock[] = [];
  for (const [callId, result] of report.results) {
    const content =
      result.status === "success"
        ? typeof result.output === "string"
          ? result.output
          : JSON.stringify(result.output)
        : JSON.stringify({ error: result.error, status: result.status });

    results.push({
      type: "tool_result",
      tool_use_id: callId,
      content,
    });
  }
  return results;
}

// ── Main class ─────────────────────────────────────────────────────────────────

export interface SynapseAnthropicOptions extends Omit<OrchestratorOptions, "tools"> {
  anthropicClient: AnthropicClient;
  tools: Record<string, AsyncToolFn>;
  maxRounds?: number;
}

export class SynapseAnthropic {
  private client: AnthropicClient;
  private orch: Orchestrator;
  private maxRounds: number;

  constructor(options: SynapseAnthropicOptions) {
    this.client = options.anthropicClient;
    this.orch = new Orchestrator({ ...options });
    this.maxRounds = options.maxRounds ?? 10;
  }

  async messages(params: {
    model: string;
    maxTokens: number;
    messages: Array<Record<string, unknown>>;
    tools: Array<Record<string, unknown>>;
    system?: string;
    [key: string]: unknown;
  }): Promise<[AnthropicMessage, ExecutionReport[]]> {
    const { model, maxTokens, messages, tools, system, ...rest } = params;
    const history = [...messages];
    const allReports: ExecutionReport[] = [];

    const createParams: Record<string, unknown> = {
      model,
      max_tokens: maxTokens,
      tools,
      ...rest,
    };
    if (system) createParams.system = system;

    for (let round = 0; round < this.maxRounds; round++) {
      const response = await this.client.messages.create({
        ...createParams,
        messages: history,
      });

      const toolBlocks = extractToolUseBlocks(response.content);

      if (toolBlocks.length === 0 || response.stop_reason === "end_turn") {
        return [response, allReports];
      }

      history.push({
        role: "assistant",
        content: response.content,
      });

      const synapseCalls = anthropicCallsToSynapse(toolBlocks);
      const report = await this.orch.run(synapseCalls);
      allReports.push(report);

      const toolResults = buildToolResultContent(report);
      history.push({
        role: "user",
        content: toolResults,
      });
    }

    throw new Error(`Exceeded maxRounds=${this.maxRounds} without a final response.`);
  }
}
