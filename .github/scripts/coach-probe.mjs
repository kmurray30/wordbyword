// Diagnostic against the deployed backend: one chat turn in a throwaway
// session (so there's a partner message), then "Check with coach" on a few
// versions of a reply - right, right but sloppy punctuation/capitals, and
// missing an accent - printing the coach's ready flag and reply for each.
// Clears the session afterwards.
const BASE = process.env.BACKEND_URL;
const session = `coach-probe-${Date.now()}`;
const post = async (path, body) => {
  const res = await fetch(`${BASE}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  return res;
};

const turn = await (await post("/chat/turn", { session_id: session, message: "Hola, ¿qué te gusta hacer en verano?" })).json();
console.log(`partner: ${turn.text}`);

const check = async (label, text) => {
  const res = await post("/translate/coach", { text, session_id: session, thread_message_id: null, message_draft: text, draft_thread: [], check: true });
  const body = await res.text();
  const events = body.split("\n\n").filter(Boolean).map((block) => {
    const ev = /^event: (.*)$/m.exec(block)?.[1];
    const data = /^data: (.*)$/m.exec(block)?.[1];
    return [ev, data ? JSON.parse(data) : null];
  });
  const ready = events.find(([e]) => e === "ready")?.[1]?.ready;
  const reply = events.find(([e]) => e === "reply_complete")?.[1]?.reply;
  const err = events.find(([e]) => e === "error")?.[1];
  console.log(`\n=== ${label}: ${JSON.stringify(text)}\n  HTTP ${res.status} ready=${ready}${err ? ` error=${JSON.stringify(err)}` : ""}\n  coach: ${reply}`);
  console.log(`  suggestion block: ${/\[\[/.test(reply ?? "") ? "yes" : "no"}`);
};

await check("right", "Me encanta nadar en el mar.");
await check("right, sloppy punctuation/capitals", "me encanta nadar en el mar");
await check("missing accent", "Me encanta nadar en el mar y tu?");
await check("real mistake", "Me gusto nadar en la mar");
await post(`/chat/history/clear?session_id=${session}`, {});
