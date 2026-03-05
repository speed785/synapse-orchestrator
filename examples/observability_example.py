import asyncio

from synapse import Orchestrator, SynapseLogger, ToolCall


async def fetch_user(user_id: int) -> dict:
    await asyncio.sleep(0.05)
    return {"id": user_id, "email": "alice@example.com"}


async def fetch_orders(user_id: int) -> list[dict]:
    await asyncio.sleep(0.05)
    return [{"order_id": "o-1", "user_id": user_id}]


async def summarize(user: dict, orders: list[dict]) -> dict:
    await asyncio.sleep(0.03)
    return {"email": user["email"], "order_count": len(orders)}


async def main() -> None:
    logger = SynapseLogger(emit_json=True)
    orchestrator = Orchestrator(
        tools={
            "fetch_user": fetch_user,
            "fetch_orders": fetch_orders,
            "summarize": summarize,
        },
        logger=logger,
        debug=True,
    )

    report = await orchestrator.run(
        [
            ToolCall(id="user", name="fetch_user", inputs={"user_id": 42}),
            ToolCall(id="orders", name="fetch_orders", inputs={"user_id": 42}),
            ToolCall(
                id="summary",
                name="summarize",
                inputs={"user": "$results.user", "orders": "$results.orders"},
            ),
        ]
    )

    print(report)
    print("\nPrometheus metrics:\n")
    print(logger.export_prometheus())


if __name__ == "__main__":
    asyncio.run(main())
