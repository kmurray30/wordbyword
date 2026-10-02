// One-off admin tool: reads the live /settings, and - if model_provider
// and/or word_weighting_enabled env vars are set - PUTs a partial update
// (only the fields given; the other is left untouched). Always prints the
// resulting settings so a run triggered with nothing set (e.g. the push
// trigger, which has no workflow_dispatch inputs to read) still reports
// current state rather than doing nothing silently.
//
// If TEST_MESSAGE is also set, sends ONE direct, un-retried /chat/turn
// request and prints the raw status + body - chat-sample.mjs's retry loop
// treats any 5xx (including a real, persistent application error like a
// rejected API key) as "backend still waking up" and keeps retrying for
// minutes before finally surfacing the real body, which makes diagnosing
// a genuine failure slow; this gives an immediate answer instead.
const BACKEND_URL = process.env.BACKEND_URL;
const MODEL_PROVIDER = (process.env.MODEL_PROVIDER || "").trim();
const WORD_WEIGHTING_ENABLED = (process.env.WORD_WEIGHTING_ENABLED || "").trim();
const TEST_MESSAGE = (process.env.TEST_MESSAGE || "").trim();

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

async function main() {
  const body = {};
  if (MODEL_PROVIDER) {
    if (!["local", "openai"].includes(MODEL_PROVIDER)) {
      fail(`model_provider must be "local" or "openai", got ${JSON.stringify(MODEL_PROVIDER)}`);
    }
    body.model_provider = MODEL_PROVIDER;
  }
  if (WORD_WEIGHTING_ENABLED) {
    if (!["true", "false"].includes(WORD_WEIGHTING_ENABLED)) {
      fail(`word_weighting_enabled must be "true" or "false", got ${JSON.stringify(WORD_WEIGHTING_ENABLED)}`);
    }
    body.word_weighting_enabled = WORD_WEIGHTING_ENABLED === "true";
  }

  if (Object.keys(body).length === 0) {
    console.log("No fields given - just reading current settings.");
    const res = await fetch(`${BACKEND_URL}/settings`);
    if (!res.ok) fail(`GET /settings returned ${res.status}: ${await res.text()}`);
    console.log(JSON.stringify(await res.json(), null, 2));
    await printHealthLlm();
    return;
  }

  console.log(`PUT /settings with ${JSON.stringify(body)} ...`);
  const res = await fetch(`${BACKEND_URL}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) fail(`PUT /settings returned ${res.status}: ${await res.text()}`);
  const result = await res.json();
  console.log("New settings:");
  console.log(JSON.stringify(result, null, 2));

  if ("model_provider" in body && result.model_provider !== body.model_provider) {
    fail(`model_provider didn't take - requested ${body.model_provider}, server reports ${result.model_provider}`);
  }
  if ("word_weighting_enabled" in body && result.word_weighting_enabled !== body.word_weighting_enabled) {
    fail(
      `word_weighting_enabled didn't take - requested ${body.word_weighting_enabled}, server reports ${result.word_weighting_enabled}`
    );
  }
  await printHealthLlm();
}

async function printHealthLlm() {
  const res = await fetch(`${BACKEND_URL}/health/llm`);
  console.log(`\nGET /health/llm -> ${res.status}: ${await res.text()}`);
}

async function testChat() {
  console.log(`\nSending one direct /chat/turn request: ${JSON.stringify(TEST_MESSAGE)} ...`);
  const res = await fetch(`${BACKEND_URL}/chat/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: TEST_MESSAGE, session_id: `admin-test-${Date.now()}` }),
  });
  const body = await res.text();
  console.log(`Status: ${res.status}`);
  console.log(`Body: ${body}`);
  if (!res.ok) fail(`/chat/turn returned ${res.status}`);
}

async function run() {
  await main();
  if (TEST_MESSAGE) await testChat();
}

run();
