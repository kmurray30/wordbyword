// Ad hoc sampler for eyeballing chat reply quality (e.g. after a prompt
// change) - sends several distinct conversational turns in one session (a
// real back-and-forth, not isolated one-shots) and prints every reply so a
// human can eyeball them side by side. Deliberately skips the browser/TTS
// checks in e2e-check.mjs - this only cares about reply quality, and the
// Playwright drive-the-page step is slow and has an unrelated flake.
const BACKEND_URL = process.env.BACKEND_URL || "https://server-production-f688.up.railway.app";
const LABEL = process.env.SAMPLE_LABEL || "gentle-steering-v2-prompt";
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
  // A push to this branch rebuilds the Railway services. The `server`
  // container itself can still be restarting, in which case Railway's edge
  // returns a 502 "Application failed to respond" - seen live: a run
  // failed on its very first request because `server`'s own redeploy
  // hadn't finished yet. Retry through any 5xx, since that's "not ready
  // yet", not a real application error.
  const maxAttempts = 18;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const res = await fetch(`${BACKEND_URL}/chat/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    if (res.ok) return res.json();
    const body = await res.text();
    if (res.status >= 500 && attempt < maxAttempts) {
      console.log(`  (backend not ready yet, status ${res.status}, attempt ${attempt}/${maxAttempts}, retrying in 10s)`);
      await new Promise((r) => setTimeout(r, 10_000));
      continue;
    }
    throw new Error(`chat/turn failed (${res.status}): ${body}`);
  }
}

async function main() {
  // Railway does a graceful rolling deploy: the OLD container keeps serving
  // successfully for ~110s while the new one builds, so a request fired
  // right after this push's git-triggered redeploy starts can get a real,
  // error-free 200 from the STALE code - no retry logic catches that,
  // since nothing failed. Confirmed live: a run's first reply was
  // byte-identical to the previous (pre-fix) run's. Wait out that window
  // before sending anything for real.
  const startupDelayMs = Number(process.env.STARTUP_DELAY_MS ?? 130_000);
  if (startupDelayMs > 0) {
    console.log(`[${LABEL}] waiting ${startupDelayMs}ms for the triggering deploy to roll over...`);
    await new Promise((r) => setTimeout(r, startupDelayMs));
  }

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
