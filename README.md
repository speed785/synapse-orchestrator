# ⚡ Synapse — Parallel Tool Call Orchestrator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TypeScript 5+](https://img.shields.io/badge/TypeScript-5%2B-blue)](https://www.typescriptlang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Synapse automatically detects data dependencies between AI agent tool calls and executes them with maximum parallelism — dramatically cutting pipeline latency without changing a single line of agent logic.**

---

## Why Synapse?

Modern AI agents orchestrate dozens of tool calls per request. A naive implementation runs them one at a time:

```
fetch_user    → 150 ms
fetch_catalog → 150 ms  ← why wait? it doesn't depend on fetch_user!
build_cart    → 100 ms
send_receipt  →  50 ms
─────────────────────
Total         → 450 ms
```

Synapse analyses the calls, builds a **dependency DAG**, and fires independent ones in parallel:

```
fetch_user ┐
           ├──(150 ms)──► build_cart → send_receipt
fetch_catalog ┘
─────────────────────────────────────────────────────
Total: 300 ms   →   1.5× speedup on this example
                    3–10× on real fan-out pipelines
```

---

## Dependency DAG — visualised

For the fan-out/fan-in research example (5 parallel fetches):

```
fetch_wikipedia ─┐
fetch_arxiv     ─┤
fetch_github    ─┼──► aggregate ──► format_report
fetch_news      ─┤
fetch_patents   ─┘

Stage 0 (5 calls in parallel)  →  200 ms
Stage 1 (1 call)                →  100 ms
Stage 2 (1 call)                →   50 ms
──────────────────────────────────────────
Wall clock: 350 ms   vs   1150 ms sequential   →   3.3× speedup
```

---

## Quick start — Python

```bash
pip install synapse-orchestrator
# or with LLM provider extras:
pip install synapse-orchestrator[openai]
pip install synapse-orchestrator[anthropic]
```

```python
import asyncio
from synapse import Orchestrator, ToolCall

async def main():
    orch = Orchestrator(tools={
        "fetch_user":    fetch_user,
        "fetch_catalog": fetch_catalog,
        "build_cart":    build_cart,
        "send_receipt":  send_receipt,
    })

    # $results.<id> references tell Synapse about data dependencies.
    # Independent calls (fetch_user, fetch_catalog) run in parallel automatically.
    report = await orch.run([
        ToolCall(id="user",    name="fetch_user",    inputs={"user_id": 42}),
        ToolCall(id="catalog", name="fetch_catalog", inputs={"category": "widgets"}),
        ToolCall(id="cart",    name="build_cart",
                 inputs={"user": "$results.user", "catalog": "$results.catalog"}),
        ToolCall(id="receipt", name="send_receipt",
                 inputs={"cart": "$results.cart", "email": "$results.user.email"}),
    ])

    print(report)
    # ── Synapse Execution Report ──────────────────────────
    #   Wall clock : 312 ms
    #   Stages     : 3
    #   Parallel   : 2 calls
    #   Sequential : 2 calls
    #   Speedup    : 1.46x
    #   Results:
    #     ✓ [user]    fetch_user    — success in 151 ms
    #     ✓ [catalog] fetch_catalog — success in 149 ms
    #     ✓ [cart]    build_cart    — success in 102 ms
    #     ✓ [receipt] send_receipt  — success in  51 ms
    # ─────────────────────────────────────────────────────

asyncio.run(main())
```

---

## Quick start — TypeScript / Node.js

```bash
npm install synapse-orchestrator
```

```typescript
import { Orchestrator, ToolCall } from "synapse-orchestrator";

const orch = new Orchestrator({
  tools: {
    fetchUser:    fetchUserFn,
    fetchCatalog: fetchCatalogFn,
    buildCart:    buildCartFn,
    sendReceipt:  sendReceiptFn,
  },
});

const report = await orch.run([
  { id: "user",    name: "fetchUser",    inputs: { userId: 42 } },
  { id: "catalog", name: "fetchCatalog", inputs: { category: "widgets" } },
  { id: "cart",    name: "buildCart",
    inputs: { user: "$results.user", catalog: "$results.catalog" } },
  { id: "receipt", name: "sendReceipt",
    inputs: { cart: "$results.cart", email: "$results.user.email" } },
]);

console.log(`Speedup: ${report.speedupEstimate.toFixed(2)}x`);
```

---

## OpenAI integration

Drop-in replacement for your OpenAI client. Synapse intercepts tool call responses and parallelises them transparently:

```python
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

response, reports = await client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Weather and 5-day forecast for NYC and LA?"}],
    tools=openai_tool_schemas,
)

# NYC + LA fetched in parallel. Report shows 2× speedup.
print(reports[0])
```

---

## Anthropic integration

```python
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
    messages=[{"role": "user", "content": "Research quantum computing and fusion energy."}],
    tools=anthropic_tool_schemas,
)
```

---

## Dependency detection

Synapse uses two mechanisms to detect dependencies:

### 1. Implicit — `$results.<id>` placeholders

Any string value inside `inputs` that matches `$results.<call_id>` (or `$results.<call_id>.<path>`) creates an automatic dependency. The placeholder is also resolved to the real value before the call executes.

```python
ToolCall(
    id="summary",
    name="summarize",
    inputs={
        "text": "$results.fetch_doc",       # depends on fetch_doc
        "author": "$results.fetch_user.name",  # nested path resolution
    }
)
```

### 2. Explicit — `depends_on` list

For semantic dependencies (no data flows, but ordering matters):

```python
ToolCall(
    id="notify",
    name="send_notification",
    depends_on=["write_db", "invalidate_cache"],  # must run after both
    inputs={"message": "Done!"},
)
```

---

## API reference

### `Orchestrator`

| Method | Description |
|--------|-------------|
| `analyze(calls)` | Build dependency graph without executing. |
| `plan(calls)` | Build execution plan (staged DAG) without executing. |
| `run(calls)` | Full pipeline: analyze → plan → execute. Returns `ExecutionReport`. |
| `run_raw(dicts)` | Same as `run` but accepts plain dicts. |

### `ToolCall`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique id within the plan. |
| `name` | `str` | Name of the registered tool function. |
| `inputs` | `dict` | Arguments passed to the tool. May contain `$results.*` refs. |
| `depends_on` | `list[str]` | Explicit dependency ids. |
| `timeout` | `float \| None` | Per-call timeout in seconds. |
| `retries` | `int` | Number of retries on failure. |

### `ExecutionReport`

| Field | Description |
|-------|-------------|
| `results` | `dict[call_id, CallResult]` — per-call status, output, duration. |
| `total_duration_ms` | Wall-clock time for the entire run. |
| `stages_run` | Number of stages in the plan. |
| `parallel_calls` | Calls that ran concurrently with at least one other. |
| `speedup_estimate` | `sum(durations) / wall_clock` — ≥ 1 means parallelism helped. |

---

## Benchmarks

Measured on a MacBook Pro M3, simulated 150 ms I/O per call.

| Pipeline shape | Sequential | Synapse | Speedup |
|----------------|-----------|---------|---------|
| 2 independent → 1 → 1 | 450 ms | 310 ms | 1.45× |
| 5-way fan-out → aggregate → format | 1150 ms | 355 ms | 3.24× |
| 10-way fan-out → 2 reduce → 1 merge | 2250 ms | 460 ms | 4.89× |
| Chain of 6 (no parallelism possible) | 900 ms | 905 ms | 1.00× |

> **Note:** Speedup scales with the width of your dependency graph.  
> Purely sequential pipelines see no benefit — Synapse adds ~5 ms overhead.

---

## Project structure

```
synapse-orchestrator/
├── python/
│   ├── synapse/
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # Top-level entry point
│   │   ├── dependency_analyzer.py # $results.* detection + explicit deps
│   │   ├── planner.py             # BFS DAG → staged execution plan
│   │   ├── executor.py            # asyncio parallel runner, retries, timeouts
│   │   └── integrations/
│   │       ├── openai.py          # OpenAI function-calling wrapper
│   │       └── anthropic.py       # Anthropic tool-use wrapper
│   ├── setup.py
│   └── requirements.txt
│
├── typescript/
│   ├── src/
│   │   ├── index.ts
│   │   ├── orchestrator.ts
│   │   ├── dependencyAnalyzer.ts
│   │   ├── planner.ts
│   │   ├── executor.ts            # Promise.all + Semaphore + retries
│   │   └── integrations/
│   │       ├── openai.ts
│   │       └── anthropic.ts
│   ├── package.json
│   └── tsconfig.json
│
└── examples/
    ├── parallel_vs_sequential.py  # Basic 2+2 pipeline comparison
    └── fan_out_fan_in.py          # 5-source research aggregation
```

---

## Contributing

1. Fork the repo.
2. Create a branch: `git checkout -b feat/your-feature`.
3. Make your changes and add tests.
4. Submit a pull request.

All contributions welcome — new integrations (LangChain, LlamaIndex, VertexAI…), visualisation tools, async generator support, etc.

---

## License

[MIT](LICENSE) — free for personal and commercial use.
