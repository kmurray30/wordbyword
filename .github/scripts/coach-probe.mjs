// Diagnostic for "the coach isn't responding": prints the SHAPE of one
// session's threads (ids, roles, kinds, text lengths - never the text, since
// this repo's logs are public), to see where the coach's replies were saved.
const BASE = process.env.BACKEND_URL;
const session = process.env.PROBE_SESSION;
const history = await (await fetch(`${BASE}/chat/history?session_id=${session}`)).json();
console.log(`messages: ${history.messages.length}`);
for (const m of history.messages) {
  const thread = (m.thread ?? []).map((e) => `#${e.id} ${e.role}/${e.kind} (${e.text.length} chars)`);
  console.log(`[${m.id}] ${m.role} (${m.text.length} chars)${thread.length ? `\n    thread: ${thread.join(", ")}` : ""}`);
}
