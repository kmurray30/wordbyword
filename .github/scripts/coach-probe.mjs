// Diagnostic for "the coach isn't responding" on iPhone: drives the LIVE
// app in WebKit (Safari's engine) with iPhone emulation, asks the coach in
// the "Your next message" panel, and prints what the panel shows plus every
// console message and page error.
import { webkit, devices } from "playwright";

const FRONTEND = process.env.FRONTEND_URL;
const browser = await webkit.launch();
const ctx = await browser.newContext({ ...devices["iPhone 13"] });
const page = await ctx.newPage();
page.on("console", (m) => console.log(`console.${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => console.log(`PAGEERROR: ${e}\n${e.stack ?? ""}`));
page.on("requestfailed", (r) => console.log(`requestfailed: ${r.url()} ${r.failure()?.errorText}`));
page.on("response", (r) => {
  if (/translate\/coach|tts\/speak/.test(r.url())) console.log(`response ${r.status()} ${r.url()}`);
});

await page.goto(FRONTEND);
await page.waitForSelector(".next-message__textarea", { timeout: 60000 });
await page.waitForSelector(".llm-gate", { state: "hidden", timeout: 60000 }).catch(() => {});

async function ask(text) {
  await page.tap(".next-message__textarea");
  await page.fill(".next-message__textarea", text);
  await page.tap(".next-message__ask");
  await page
    .waitForSelector('.next-message .talk-line--coach:last-child[data-status="done"], .next-message .talk-line--coach:last-child[data-status="error"]', { timeout: 30000 })
    .catch(() => console.log("  (no finished coach line within 30s)"));
  const lines = await page.$$eval(".next-message .talk-line", (els) =>
    els.map((e) => `${e.classList.contains("talk-line--you") ? "YOU" : "COACH"} [${e.dataset.status ?? ""}] ${e.textContent}`),
  );
  console.log(`\n=== ${text}\n${lines.join("\n")}`);
}

await ask("hola, me gusta running");
await ask("how do I say I'm tired?");
await page.screenshot({ path: "coach-webkit.png", fullPage: true });
await browser.close();
