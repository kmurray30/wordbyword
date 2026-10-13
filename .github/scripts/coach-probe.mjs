// Diagnostic against the deployed backend, one chat turn per language in a
// throwaway session: the partner's reply (should be in that language), its
// glosses (word meanings and phrases) and readings (romaji/pinyin/Korean
// romanization), and one coach reply. Clears each session afterwards.
const BASE = process.env.BACKEND_URL;
const post = async (path, body) => {
  const res = await fetch(`${BASE}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text, status: res.status };
  }
};

const cases = [
  { language: "es", message: "Puede ser que llueva mañana, pero me encanta seguir la broma." },
  { language: "fr", message: "Salut ! J'aime faire la fête le week-end. Et toi ?" },
  { language: "ja", message: "こんにちは！週末は何をしますか？" },
  { language: "ko", message: "안녕하세요! 주말에 뭐 해요?" },
  { language: "zh-Hant", message: "你好！你周末喜欢做什么？" },
];

let failures = 0;
for (const { language, message } of cases) {
  const session = `lang-probe-${language}-${Date.now()}`;
  console.log(`\n===== ${language}: ${message}`);
  const turn = await post("/chat/turn", { session_id: session, message, language });
  if (!turn.text) {
    console.log("  TURN FAILED:", JSON.stringify(turn).slice(0, 300));
    failures++;
    continue;
  }
  console.log(`  partner: ${turn.text}`);
  console.log(`  translation: ${turn.translation}`);
  const words = turn.glosses?.words ?? [];
  console.log(`  words (${words.length}): ${words.map((w) => `${turn.text.slice(w.start, w.end)}=${w.gloss}`).join(" | ")}`);
  for (const p of turn.glosses?.phrases ?? []) console.log(`  PHRASE "${turn.text.slice(p.start, p.end)}" -> ${p.gloss}`);
  if (!words.length) failures++;
  const r = await post("/translate/readings", { texts: [turn.text], language });
  const item = r.items?.[0];
  if (item?.readings?.length) {
    console.log(`  shown: ${item.text}`);
    console.log(`  readings: ${item.readings.map((x) => `${item.text.slice(x.start, x.end)}${x.kana ? `(${x.kana})` : ""}:${x.roman}`).join(" ")}`);
  }
  const mine = await post("/chat/message/glosses", { message_id: turn.user_message_id });
  console.log(`  learner words: ${(mine.words ?? []).map((w) => `${message.slice(w.start, w.end)}=${w.gloss}`).join(" | ")}`);
  const coach = await fetch(`${BASE}/translate/coach`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text: "How do I say: I like to read books", session_id: session, language }),
  }).then((res) => res.text());
  const reply = coach.split("\n").filter((l) => l.startsWith("data:") && coach.includes("reply_complete")).map((l) => JSON.parse(l.slice(5))).find((d) => d.reply)?.reply;
  console.log(`  coach: ${reply ?? coach.slice(0, 300)}`);
  if (!reply || !reply.includes("[[")) failures++;
  await post(`/chat/history/clear?session_id=${session}`, {});
}
console.log(`\n${failures ? `FAILURES: ${failures}` : "ALL LANGUAGES OK"}`);
process.exit(failures ? 1 : 0);
