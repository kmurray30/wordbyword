// One-off admin tool: reads the live /settings, and - if model_provider
// and/or word_weighting_enabled env vars are set - PUTs a partial update
// (only the fields given; the other is left untouched). Always prints the
// resulting settings so a run triggered with nothing set (e.g. the push
// trigger, which has no workflow_dispatch inputs to read) still reports
// current state rather than doing nothing silently.
const BACKEND_URL = process.env.BACKEND_URL;
const MODEL_PROVIDER = (process.env.MODEL_PROVIDER || "").trim();
const WORD_WEIGHTING_ENABLED = (process.env.WORD_WEIGHTING_ENABLED || "").trim();

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
}

main();
