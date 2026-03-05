/**
 * Synapse — Parallel Tool Call Orchestrator for AI Agents
 * =========================================================
 *
 * @packageDocumentation
 */

export { Orchestrator } from "./orchestrator.js";
export type { OrchestratorOptions } from "./orchestrator.js";

export type { ToolCall, DependencyGraph } from "./dependencyAnalyzer.js";
export { DependencyAnalyzer } from "./dependencyAnalyzer.js";

export type { Stage, ExecutionPlan } from "./planner.js";
export { Planner } from "./planner.js";

export type { AsyncToolFn, CallResult, ExecutionReport, ExecutorOptions } from "./executor.js";
export { Executor } from "./executor.js";

export { SynapseOpenAI } from "./integrations/openai.js";
export type { SynapseOpenAIOptions } from "./integrations/openai.js";
export { SynapseAnthropic } from "./integrations/anthropic.js";
export type { SynapseAnthropicOptions } from "./integrations/anthropic.js";
