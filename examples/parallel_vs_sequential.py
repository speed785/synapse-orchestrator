"""
Example 1 — Parallel vs Sequential Execution
=============================================

Demonstrates how Synapse automatically identifies independent tool calls
and runs them concurrently, while correctly sequencing dependent calls.

Simulated pipeline:
  ┌─────────────┐   ┌─────────────────┐
  │ fetch_user  │   │  fetch_catalog  │  ← no deps, run in parallel
  └──────┬──────┘   └────────┬────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
          ┌─────────────────┐
          │  build_cart     │  ← depends on both above
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │  send_receipt   │  ← depends on build_cart
          └─────────────────┘

Without Synapse (sequential): ~600 ms
With Synapse (parallel):       ~300 ms  →  ~2x speedup
"""

import asyncio
import sys
import time

# Allow running the example without installing the package
sys.path.insert(0, "python")

from synapse import Orchestrator, ToolCall

# ── Simulated tool functions (each sleeps to mimic real I/O) ─────────────────

async def fetch_user(user_id: int) -> dict:
    await asyncio.sleep(0.15)  # 150 ms simulated DB call
    return {"id": user_id, "name": "Alice", "email": "alice@example.com"}


async def fetch_catalog(category: str) -> list:
    await asyncio.sleep(0.15)  # 150 ms simulated API call
    return [
        {"sku": "A1", "name": "Widget Pro", "price": 29.99},
        {"sku": "A2", "name": "Gadget Plus", "price": 49.99},
    ]


async def build_cart(user: dict, catalog: list) -> dict:
    await asyncio.sleep(0.10)  # 100 ms build logic
    return {
        "user": user["name"],
        "items": catalog[:1],
        "total": catalog[0]["price"],
    }


async def send_receipt(cart: dict, email: str) -> dict:
    await asyncio.sleep(0.05)  # 50 ms email send
    return {"sent": True, "to": email, "total": cart["total"]}


# ── Sequential baseline ───────────────────────────────────────────────────────

async def run_sequential() -> float:
    print("\n--- Sequential execution (baseline) ---")
    t0 = time.perf_counter()

    user = await fetch_user(42)
    catalog = await fetch_catalog("widgets")
    cart = await build_cart(user, catalog)
    receipt = await send_receipt(cart, user["email"])

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Receipt sent: {receipt}")
    print(f"Sequential wall-clock: {elapsed:.0f} ms")
    return elapsed


# ── Synapse parallel execution ────────────────────────────────────────────────

async def run_with_synapse() -> float:
    print("\n--- Synapse parallel execution ---")

    orch = Orchestrator(
        tools={
            "fetch_user":    fetch_user,
            "fetch_catalog": fetch_catalog,
            "build_cart":    build_cart,
            "send_receipt":  send_receipt,
        },
        on_call_start=lambda c: print(f"  → starting  [{c.id}] {c.name}"),
        on_call_end=lambda r: print(
            f"  ✓ finished  [{r.call_id}] {r.tool_name} ({r.duration_ms:.0f} ms)"
        ),
    )

    # Print the plan before executing
    plan = orch.plan([
        ToolCall(id="user",    name="fetch_user",    inputs={"user_id": 42}),
        ToolCall(id="catalog", name="fetch_catalog", inputs={"category": "widgets"}),
        ToolCall(id="cart",    name="build_cart",
                 inputs={"user": "$results.user", "catalog": "$results.catalog"}),
        ToolCall(id="receipt", name="send_receipt",
                 inputs={"cart": "$results.cart", "email": "$results.user.email"}),
    ])

    print("\nExecution plan:")
    print(plan)

    t0 = time.perf_counter()
    report = await orch.run([
        ToolCall(id="user",    name="fetch_user",    inputs={"user_id": 42}),
        ToolCall(id="catalog", name="fetch_catalog", inputs={"category": "widgets"}),
        ToolCall(id="cart",    name="build_cart",
                 inputs={"user": "$results.user", "catalog": "$results.catalog"}),
        ToolCall(id="receipt", name="send_receipt",
                 inputs={"cart": "$results.cart", "email": "$results.user.email"}),
    ])
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n{report}")
    print(f"Synapse wall-clock: {elapsed:.0f} ms")
    return elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    seq_ms  = await run_sequential()
    par_ms  = await run_with_synapse()

    speedup = seq_ms / par_ms
    print(f"\n{'='*55}")
    print(f"  Sequential : {seq_ms:.0f} ms")
    print(f"  Synapse    : {par_ms:.0f} ms")
    print(f"  Speedup    : {speedup:.2f}x")
    print(f"{'='*55}")


if __name__ == "__main__":
    asyncio.run(main())
