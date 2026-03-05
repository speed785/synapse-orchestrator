import { Orchestrator, type ToolCall } from "../src/index.js";

const SOURCES = ["wikipedia", "arxiv", "github", "news", "patents"];

async function fetchSource(source: string, query: string): Promise<Record<string, unknown>> {
  await new Promise((resolve) => setTimeout(resolve, 200));
  return {
    source,
    query,
    results: [`Result 1 from ${source}`, `Result 2 from ${source}`],
    count: 2,
  };
}

async function aggregate(
  s1: Record<string, unknown>,
  s2: Record<string, unknown>,
  s3: Record<string, unknown>,
  s4: Record<string, unknown>,
  s5: Record<string, unknown>
): Promise<Record<string, unknown>> {
  await new Promise((resolve) => setTimeout(resolve, 100));
  const all = [s1, s2, s3, s4, s5];
  const results = all.flatMap((v) => v.results as string[]);
  const total = all.reduce((sum, v) => sum + Number(v.count), 0);
  return { total, results };
}

async function formatReport(aggregated: Record<string, unknown>, title: string): Promise<string> {
  await new Promise((resolve) => setTimeout(resolve, 50));
  const lines = [
    `# ${title}`,
    `Total results: ${aggregated.total}`,
    "",
    ...((aggregated.results as string[]).map((result, index) => `${index + 1}. ${result}`)),
  ];
  return lines.join("\n");
}

async function main(): Promise<void> {
  const query = "quantum computing breakthroughs";
  const calls: ToolCall[] = [
    ...SOURCES.map((source) => ({
      id: `fetch_${source}`,
      name: "fetchSource",
      inputs: { source, query },
    })),
    {
      id: "agg",
      name: "aggregate",
      inputs: {
        s1: "$results.fetch_wikipedia",
        s2: "$results.fetch_arxiv",
        s3: "$results.fetch_github",
        s4: "$results.fetch_news",
        s5: "$results.fetch_patents",
      },
    },
    {
      id: "report",
      name: "formatReport",
      inputs: {
        aggregated: "$results.agg",
        title: `Research Report: ${query}`,
      },
    },
  ];

  const orchestrator = new Orchestrator({
    tools: {
      fetchSource,
      aggregate,
      formatReport,
    },
  });

  const report = await orchestrator.run(calls);
  const output = report.results.get("report")?.output;
  console.log(String(output));
  console.log(`Speedup estimate: ${report.speedupEstimate.toFixed(2)}x`);
}

void main();
