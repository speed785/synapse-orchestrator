"""
Example 2 — Fan-out / Fan-in Pattern
=====================================

A classic "gather then aggregate" pattern that benefits enormously from
parallelism.  An AI agent fetches data from 5 independent sources and
then aggregates everything into a report.

Dependency graph:

  fetch_s1 ─┐
  fetch_s2 ─┤
  fetch_s3 ─┼──► aggregate ──► format_report
  fetch_s4 ─┤
  fetch_s5 ─┘

Sequential would take 5×200ms + 100ms + 50ms = 1150ms.
Synapse runs all 5 fetches in parallel: 200ms + 100ms + 50ms = 350ms.
Theoretical speedup ≈ 3.3×.
"""

import asyncio
import sys
import time

sys.path.insert(0, "python")

from synapse import Orchestrator, ToolCall

SOURCES = ["wikipedia", "arxiv", "github", "news", "patents"]


async def fetch_source(source: str, query: str) -> dict:
    """Simulate fetching from a data source (200 ms each)."""
    await asyncio.sleep(0.20)
    return {
        "source": source,
        "query": query,
        "results": [f"Result 1 from {source}", f"Result 2 from {source}"],
        "count": 2,
    }


async def aggregate(
    s1: dict, s2: dict, s3: dict, s4: dict, s5: dict
) -> dict:
    """Merge results from all five sources (100 ms)."""
    await asyncio.sleep(0.10)
    all_results = []
    for src in [s1, s2, s3, s4, s5]:
        all_results.extend(src["results"])
    return {
        "total": sum(s["count"] for s in [s1, s2, s3, s4, s5]),
        "results": all_results,
    }


async def format_report(aggregated: dict, title: str) -> str:
    """Render the final report (50 ms)."""
    await asyncio.sleep(0.05)
    lines = [f"# {title}", f"Total results: {aggregated['total']}", ""]
    for i, r in enumerate(aggregated["results"], 1):
        lines.append(f"{i}. {r}")
    return "\n".join(lines)


async def main():
    query = "quantum computing breakthroughs"

    # ── Build the ToolCall list programmatically ──────────────────────────────
    fetch_calls = [
        ToolCall(
            id=f"fetch_{src}",
            name="fetch_source",
            inputs={"source": src, "query": query},
        )
        for src in SOURCES
    ]

    aggregate_call = ToolCall(
        id="agg",
        name="aggregate",
        inputs={
            "s1": "$results.fetch_wikipedia",
            "s2": "$results.fetch_arxiv",
            "s3": "$results.fetch_github",
            "s4": "$results.fetch_news",
            "s5": "$results.fetch_patents",
        },
    )

    report_call = ToolCall(
        id="report",
        name="format_report",
        inputs={
            "aggregated": "$results.agg",
            "title": f"Research Report: {query}",
        },
    )

    all_calls = fetch_calls + [aggregate_call, report_call]

    # ── Create the orchestrator ────────────────────────────────────────────────
    orch = Orchestrator(
        tools={
            "fetch_source":  fetch_source,
            "aggregate":     aggregate,
            "format_report": format_report,
        }
    )

    # Show the plan
    plan = orch.plan(all_calls)
    print("Execution plan:")
    print(plan)
    print()

    # ── Run with Synapse ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    exec_report = await orch.run(all_calls)
    synapse_ms = (time.perf_counter() - t0) * 1000

    final_report = exec_report.results["report"].output
    print("Generated report:")
    print("─" * 50)
    print(final_report)
    print("─" * 50)

    print(exec_report)

    # ── Compare against fully sequential ─────────────────────────────────────
    t0 = time.perf_counter()
    results = {}
    for src in SOURCES:
        results[src] = await fetch_source(src, query)
    agg = await aggregate(
        results["wikipedia"], results["arxiv"], results["github"],
        results["news"], results["patents"]
    )
    _ = await format_report(agg, f"Research Report: {query}")
    sequential_ms = (time.perf_counter() - t0) * 1000

    print(f"\nSpeedup: {sequential_ms / synapse_ms:.2f}x "
          f"({sequential_ms:.0f} ms → {synapse_ms:.0f} ms)")


if __name__ == "__main__":
    asyncio.run(main())
