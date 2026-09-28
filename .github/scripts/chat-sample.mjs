// Ad hoc sampler for comparing chat quality across WORD_WEIGHTING_ENABLED
// on/off - sends several distinct conversational turns in one session (a
// real back-and-forth, not isolated one-shots) and prints every reply so a
// human can eyeball them side by side. Deliberately skips the browser/TTS
// checks in e2e-check.mjs - this only cares about reply quality, and the
// Playwright drive-the-page step is slow and has an unrelated flake.
const BACKEND_URL = process.env.BACKEND_URL || "https://server-production-f688.up.railway.app";
const LABEL = process.env.SAMPLE_LABEL || "run";
const sessionId = `sample-${LABEL}-${crypto.randomUUID()}`;

const messages = [
  "Hi, what hobbies do you like?",
  "What did you do last weekend?",
  "Do you have a favorite food?",
  "Tell me about your family.",
  "What's your favorite movie?",
  "Hola, ¿cómo estás?",
];

async function chatTurn(message) {
  const res = await fetch(`${BACKEND_URL}/chat/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) {
    throw new Error(`chat/turn failed (${res.status}): ${await res.text()}`);
  }
  return res.json();
}

async function main() {
  console.log(`[${LABEL}] session ${sessionId}`);
  for (const message of messages) {
    const started = Date.now();
    const { text } = await chatTurn(message);
    console.log(`[${LABEL}] > ${message}`);
    console.log(`[${LABEL}] < ${text}  (${Date.now() - started}ms)`);
    console.log("");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
