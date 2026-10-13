// Diagnostic against the deployed backend: one chat turn in a throwaway
// session, then prints the word/phrase glosses for the partner's reply and
// for the learner's own message (POST /chat/message/glosses), then clears
// the session.
const BASE = process.env.BACKEND_URL;
const session = `gloss-probe-${Date.now()}`;
const post = async (path, body) => {
  const res = await fetch(`${BASE}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  return { status: res.status, json: await res.json().catch(() => null) };
};
const show = (label, text, g) => {
  console.log(`\n=== ${label}: ${text}`);
  if (!g) return console.log("  (no glosses)");
  for (const w of g.words) console.log(`  ${text.slice(w.start, w.end)} -> ${w.gloss}${w.grammar ? `   [${w.grammar}]` : ""}`);
  for (const p of g.phrases) console.log(`  PHRASE "${text.slice(p.start, p.end)}" -> ${p.gloss}${p.note ? `   (${p.note})` : ""}`);
};

const t0 = Date.now();
const turn = await post("/chat/turn", { session_id: session, message: "¡Hola! Ayer fui a la playa con mis amigos y comimos paella. ¿Qué tal tu día?" });
console.log(`turn HTTP ${turn.status} in ${Date.now() - t0} ms; translation: ${turn.json?.translation}`);
show("reply", turn.json.text, turn.json.glosses);
const t1 = Date.now();
const mine = await post("/chat/message/glosses", { message_id: turn.json.user_message_id });
console.log(`\nuser glosses HTTP ${mine.status} in ${Date.now() - t1} ms`);
show("mine", "¡Hola! Ayer fui a la playa con mis amigos y comimos paella. ¿Qué tal tu día?", mine.json);
const t2 = Date.now();
const again = await post("/chat/message/glosses", { message_id: turn.json.user_message_id });
console.log(`\nsecond request (saved) HTTP ${again.status} in ${Date.now() - t2} ms`);
const hist = await (await fetch(`${BASE}/chat/history?session_id=${session}`)).json();
console.log(`history carries glosses: ${hist.messages.map((m) => `${m.role}=${!!m.glosses}`).join(", ")}`);
await post(`/chat/history/clear?session_id=${session}`, {});
