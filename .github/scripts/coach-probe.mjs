// Sends a handful of typical coach-panel inputs to the deployed
// /translate/coach and prints every SSE event raw, with timings - for
// seeing exactly what the coach returns when the app looks stuck.
const BASE = process.env.BACKEND_URL;
const session = `coach-probe-${Date.now()}`;

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return res;
}

async function coach(label, body) {
  const t0 = Date.now();
  const res = await post("/translate/coach", { session_id: session, draft_thread: [], ...body });
  console.log(`\n=== ${label} (HTTP ${res.status}) ===`);
  const text = await res.text();
  for (const line of text.split("\n")) {
    if (line.trim()) console.log(`  ${line}`);
  }
  console.log(`  [${Date.now() - t0} ms]`);
}

// A partner message to talk about.
const turn = await post("/chat/turn", { session_id: session, message: "Hola, ¿cómo estás?" });
console.log(`chat turn HTTP ${turn.status}`);
const history = await (await fetch(`${BASE}/chat/history?session_id=${session}`)).json();
const partner = history.messages.filter((m) => m.role === "assistant").at(-1);
console.log(`partner: ${partner?.id} ${partner?.text}`);

await coach("cold spanish", { text: "estoy bien, y tu?", thread_message_id: null });
await coach("english want-to-say", { text: "how do I say I'm tired but happy", thread_message_id: null });
await coach("spanglish", { text: "yo quiero going to the beach mañana", thread_message_id: null });
await coach("question", { text: "what does cómo estás mean literally?", thread_message_id: null });
await coach("question, thread open", { text: "and estás?", thread_message_id: partner?.id ?? null });
await coach("check", { text: "Estoy bien gracias", thread_message_id: null, check: true });

await post(`/chat/history/clear?session_id=${session}`, {});
