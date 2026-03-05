import { Orchestrator, type ToolCall } from "../src/index.js";

async function fetchUser(userId: number): Promise<Record<string, unknown>> {
  await new Promise((resolve) => setTimeout(resolve, 150));
  return { id: userId, name: "Alice", email: "alice@example.com" };
}

async function fetchCatalog(category: string): Promise<Array<Record<string, unknown>>> {
  await new Promise((resolve) => setTimeout(resolve, 150));
  return [
    { sku: "A1", name: "Widget Pro", price: 29.99, category },
    { sku: "A2", name: "Gadget Plus", price: 49.99, category },
  ];
}

async function buildCart(
  user: Record<string, unknown>,
  catalog: Array<Record<string, unknown>>
): Promise<Record<string, unknown>> {
  await new Promise((resolve) => setTimeout(resolve, 100));
  return {
    user: user.name,
    items: [catalog[0]],
    total: catalog[0].price,
  };
}

async function sendReceipt(
  cart: Record<string, unknown>,
  email: string
): Promise<Record<string, unknown>> {
  await new Promise((resolve) => setTimeout(resolve, 50));
  return { sent: true, to: email, total: cart.total };
}

async function runSequential(): Promise<number> {
  const start = performance.now();
  const user = await fetchUser(42);
  const catalog = await fetchCatalog("widgets");
  const cart = await buildCart(user, catalog);
  await sendReceipt(cart, String(user.email));
  return performance.now() - start;
}

async function runWithSynapse(): Promise<number> {
  const orchestrator = new Orchestrator({
    tools: { fetchUser, fetchCatalog, buildCart, sendReceipt },
  });

  const calls: ToolCall[] = [
    { id: "user", name: "fetchUser", inputs: { userId: 42 } },
    { id: "catalog", name: "fetchCatalog", inputs: { category: "widgets" } },
    {
      id: "cart",
      name: "buildCart",
      inputs: { user: "$results.user", catalog: "$results.catalog" },
    },
    {
      id: "receipt",
      name: "sendReceipt",
      inputs: { cart: "$results.cart", email: "$results.user.email" },
    },
  ];

  const start = performance.now();
  const report = await orchestrator.run(calls);
  const elapsed = performance.now() - start;
  console.log(`Speedup estimate: ${report.speedupEstimate.toFixed(2)}x`);
  return elapsed;
}

async function main(): Promise<void> {
  const sequentialMs = await runSequential();
  const synapseMs = await runWithSynapse();
  console.log(`Sequential: ${sequentialMs.toFixed(0)}ms`);
  console.log(`Synapse: ${synapseMs.toFixed(0)}ms`);
  console.log(`Observed speedup: ${(sequentialMs / synapseMs).toFixed(2)}x`);
}

void main();
