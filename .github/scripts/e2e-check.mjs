// On-demand end-to-end check against the deployed Railway services. Runs on
// GitHub Actions (not in the dev sandbox, which can't reach *.up.railway.app
// at all - its egress policy blocks that domain outright). Two stages:
//  1. Hit the backend's /chat/turn directly - isolates "is the LLM path
//     actually working" from any frontend issue.
//  2. If that passes, drive the real site with Playwright end to end.
import { chromium } from "playwright";

const BACKEND_URL = process.env.BACKEND_URL;
const FRONTEND_URL = process.env.FRONTEND_URL;
const CHAT_TIMEOUT_MS = 100_000;

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

async function checkBackendDirect() {
  console.log(`[1/2] Checking backend health at ${BACKEND_URL}/health ...`);
  const health = await fetch(`${BACKEND_URL}/health`);
  if (!health.ok) fail(`/health returned ${health.status}`);
  console.log("  health OK");

  console.log(`[1/2] Sending a direct /chat/turn request (up to ${CHAT_TIMEOUT_MS / 1000}s) ...`);
  const start = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(`${BACKEND_URL}/chat/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Hola" }),
      signal: controller.signal,
    });
  } catch (err) {
    fail(`/chat/turn request failed/timed out after ${Date.now() - start}ms: ${err}`);
  } finally {
    clearTimeout(timer);
  }
  const elapsed = Date.now() - start;
  if (!res.ok) {
    const body = await res.text();
    fail(`/chat/turn returned ${res.status} after ${elapsed}ms: ${body}`);
  }
  const data = await res.json();
  if (!data.text || data.text.trim().length === 0) {
    fail(`/chat/turn returned 200 but empty text: ${JSON.stringify(data)}`);
  }
  console.log(`  OK in ${elapsed}ms - reply: ${JSON.stringify(data.text)}`);
  console.log(`  tokens: ${data.tokens?.length ?? 0}`);
}

async function checkBrowserEndToEnd() {
  console.log(`[2/2] Driving ${FRONTEND_URL} with Playwright ...`);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 800 } });
  const consoleErrors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  try {
    await page.goto(FRONTEND_URL, { waitUntil: "load", timeout: 30_000 });
    await page.waitForSelector("text=wordbyword", { timeout: 15_000 });
    await page.screenshot({ path: "e2e-1-loaded.png" });

    await page.fill("textarea", "Hola");
    await page.keyboard.press("Enter");

    await page.waitForSelector(".chat-message--assistant", { timeout: CHAT_TIMEOUT_MS });
    await page.screenshot({ path: "e2e-2-chat-reply.png" });

    const replyText = await page.locator(".chat-message--assistant").first().innerText();
    if (!replyText || replyText.trim().length === 0) {
      fail("assistant message rendered but has no text");
    }
    console.log(`  OK - assistant replied: ${JSON.stringify(replyText)}`);

    // Hover a word token and confirm the translation popover appears.
    const wordToken = page.locator(".chat-message--assistant .word-token").first();
    if ((await wordToken.count()) > 0) {
      await wordToken.hover();
      await page.waitForSelector(".chat-message--assistant .translate-popover", { timeout: 8_000 });
      console.log("  OK - hover translation popover works");
    } else {
      console.log("  (no trackable word tokens in this reply - skipping hover check)");
    }

    if (consoleErrors.length > 0) {
      console.log("  Browser console errors seen:", consoleErrors);
    }
  } finally {
    await browser.close();
  }
}

await checkBackendDirect();
await checkBrowserEndToEnd();
console.log("ALL CHECKS PASSED");
