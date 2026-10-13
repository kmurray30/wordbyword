// Diagnostic against the deployed backend: glosses for one learner message
// (one chat turn in a throwaway session, then POST /chat/message/glosses on
// the learner's message), printing each word's meaning and each phrase -
// to check words get their own dictionary meaning, not their idiom's.
// Clears the session afterwards.
const BASE = process.env.BACKEND_URL;
const session = `gloss-probe-${Date.now()}`;
const text = "Me encanta seguir la broma con mis amigos, pero echo de menos a mi familia.";
const post = async (path, body) => (await fetch(`${BASE}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })).json();

const turn = await post("/chat/turn", { session_id: session, message: text });
const g = await post("/chat/message/glosses", { message_id: turn.user_message_id });
console.log(`=== ${text}`);
for (const w of g.words ?? []) console.log(`  ${text.slice(w.start, w.end).padEnd(10)} -> ${w.gloss}${w.grammar ? `   [${w.grammar}]` : ""}`);
for (const p of g.phrases ?? []) console.log(`  PHRASE "${text.slice(p.start, p.end)}" -> ${p.gloss}${p.note ? `   (${p.note})` : ""}`);
await post(`/chat/history/clear?session_id=${session}`, {});
