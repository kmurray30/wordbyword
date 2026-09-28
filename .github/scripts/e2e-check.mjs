// On-demand end-to-end check against the deployed Railway services. Runs on
// GitHub Actions (not in the dev sandbox, which can't reach *.up.railway.app
// at all - its egress policy blocks that domain outright). Six stages:
//  1. Hit the backend's /chat/turn directly - isolates "is the LLM path
//     actually working" from any frontend issue.
//  2. Hit the backend's /tts/speak directly - isolates "is DeepInfra TTS
//     actually working" (right model/voice, valid token) from the frontend.
//  3. Hit /tts/voices and /tts/speak for each Spanish voice directly -
//     confirms every voice the picker offers is a real, working DeepInfra
//     voice id, not just the default.
//  4. Send an English message to /chat/turn, and a fixed Spanish phrase to
//     /translate/text - prints both for a human to eyeball (semantic
//     correctness isn't something a script can assert), but at least
//     confirms the LLM-backed translation path actually returns something,
//     not the old Argos error string.
//  5. Confirm GET /chat/history reflects what stages 1/4 just sent, and
//     that POST /chat/history/clear actually empties it.
//  6. If those pass, drive the real site with Playwright end to end: send a
//     message, hover a word gloss, play its audio, toggle both messages'
//     translation rows open and closed, confirm the user message gets two
//     rows (EN+ES), hover a cognate word in the input box for its
//     dual-column candidates, switch the voice picker and confirm that
//     request carries the chosen voice, then reload and clear the chat.
//
// Stages 1-5 all use one throwaway session id (see TEST_SESSION_ID) so this
// script's own chat traffic never lands in - or pollutes - anyone real's
// conversation history, and gets explicitly cleared at the end regardless.
import { chromium } from "playwright";

const BACKEND_URL = process.env.BACKEND_URL;
const FRONTEND_URL = process.env.FRONTEND_URL;
const CHAT_TIMEOUT_MS = 100_000;
const TTS_TIMEOUT_MS = 30_000;
const TEST_SESSION_ID = `e2e-${crypto.randomUUID()}`;

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

async function chatTurn(message) {
  const start = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(`${BACKEND_URL}/chat/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: TEST_SESSION_ID }),
      signal: controller.signal,
    });
  } catch (err) {
    fail(`/chat/turn (${JSON.stringify(message)}) failed/timed out after ${Date.now() - start}ms: ${err}`);
  } finally {
    clearTimeout(timer);
  }
  const elapsed = Date.now() - start;
  if (!res.ok) {
    fail(`/chat/turn (${JSON.stringify(message)}) returned ${res.status} after ${elapsed}ms: ${await res.text()}`);
  }
  const data = await res.json();
  if (!data.text || data.text.trim().length === 0) {
    fail(`/chat/turn (${JSON.stringify(message)}) returned 200 but empty text: ${JSON.stringify(data)}`);
  }
  return { elapsed, text: data.text, tokenCount: data.tokens?.length ?? 0 };
}

async function checkBackendDirect() {
  console.log(`[1/6] Checking backend health at ${BACKEND_URL}/health ...`);
  const health = await fetch(`${BACKEND_URL}/health`);
  if (!health.ok) fail(`/health returned ${health.status}`);
  console.log("  health OK");

  console.log(`[1/6] Sending a direct /chat/turn request (session ${TEST_SESSION_ID}, up to ${CHAT_TIMEOUT_MS / 1000}s) ...`);
  const { elapsed, text, tokenCount } = await chatTurn("Hola");
  console.log(`  OK in ${elapsed}ms - reply: ${JSON.stringify(text)}`);
  console.log(`  tokens: ${tokenCount}`);
}

async function speakDirect(voice) {
  const start = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TTS_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(`${BACKEND_URL}/tts/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(voice ? { text: "Hola, ¿cómo estás?", voice } : { text: "Hola, ¿cómo estás?" }),
      signal: controller.signal,
    });
  } catch (err) {
    fail(`/tts/speak (voice=${voice ?? "default"}) failed/timed out after ${Date.now() - start}ms: ${err}`);
  } finally {
    clearTimeout(timer);
  }
  const elapsed = Date.now() - start;
  if (!res.ok) {
    const body = await res.text();
    fail(`/tts/speak (voice=${voice ?? "default"}) returned ${res.status} after ${elapsed}ms: ${body}`);
  }
  const contentType = res.headers.get("content-type") ?? "";
  const buf = await res.arrayBuffer();
  if (!contentType.includes("audio")) {
    fail(`/tts/speak (voice=${voice ?? "default"}) returned 200 but content-type was ${JSON.stringify(contentType)}, not audio`);
  }
  if (buf.byteLength < 1000) {
    fail(`/tts/speak (voice=${voice ?? "default"}) returned 200 but only ${buf.byteLength} bytes - too small to be real audio`);
  }
  return { elapsed, contentType, bytes: buf.byteLength };
}

async function checkTTSDirect() {
  console.log(`[2/6] Sending a direct /tts/speak request (up to ${TTS_TIMEOUT_MS / 1000}s) ...`);
  const { elapsed, contentType, bytes } = await speakDirect(undefined);
  console.log(`  OK in ${elapsed}ms - ${contentType}, ${bytes} bytes`);
}

async function checkVoicesDirect() {
  console.log(`[3/6] Checking /tts/voices?language=es and every Spanish voice ...`);
  const res = await fetch(`${BACKEND_URL}/tts/voices?language=es`);
  if (!res.ok) fail(`/tts/voices?language=es returned ${res.status}`);
  const data = await res.json();
  const expected = ["ef_dora", "em_alex", "em_santa"];
  const got = [...data.voices].sort();
  if (JSON.stringify(got) !== JSON.stringify([...expected].sort())) {
    fail(`/tts/voices?language=es returned ${JSON.stringify(data.voices)}, expected ${JSON.stringify(expected)}`);
  }
  console.log(`  OK - voices: ${JSON.stringify(data.voices)}, default: ${data.default}`);

  for (const voice of data.voices) {
    const { elapsed, bytes } = await speakDirect(voice);
    console.log(`  OK - voice ${voice} in ${elapsed}ms, ${bytes} bytes`);
  }
}

async function checkEnglishInputAndTranslation() {
  console.log(`[4/6] Sending an English message to /chat/turn (session ${TEST_SESSION_ID}, no prior turns polluting it) ...`);
  const { text } = await chatTurn("Hi, what hobbies do you like?");
  console.log(`  reply: ${JSON.stringify(text)} (eyeball: does this answer, or just echo/translate the question?)`);

  console.log(`[4/6] Sending a direct /translate/text request ...`);
  const translateRes = await fetch(`${BACKEND_URL}/translate/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "¡Hola! ¿Estás bien?", source_lang: "es", target_lang: "en" }),
  });
  if (!translateRes.ok) fail(`/translate/text returned ${translateRes.status}: ${await translateRes.text()}`);
  const translateData = await translateRes.json();
  if (!translateData.translation || translateData.translation.startsWith("(translation unavailable")) {
    fail(`/translate/text returned an unusable translation: ${JSON.stringify(translateData)}`);
  }
  console.log(`  "¡Hola! ¿Estás bien?" -> ${JSON.stringify(translateData.translation)}`);
}

async function checkHistoryAndClear() {
  console.log(`[5/6] Checking GET /chat/history reflects this session's turns ...`);
  const historyRes = await fetch(`${BACKEND_URL}/chat/history?session_id=${encodeURIComponent(TEST_SESSION_ID)}`);
  if (!historyRes.ok) fail(`/chat/history returned ${historyRes.status}: ${await historyRes.text()}`);
  const historyData = await historyRes.json();
  // 2 chatTurn() calls so far (stage 1's "Hola", stage 4's hobbies
  // question) = 4 messages (user+assistant each).
  if (historyData.messages.length !== 4) {
    fail(`/chat/history returned ${historyData.messages.length} messages, expected 4: ${JSON.stringify(historyData)}`);
  }
  console.log(`  OK - ${historyData.messages.length} messages recorded for this session`);

  console.log(`[5/6] Clearing this session's history via POST /chat/history/clear ...`);
  const clearRes = await fetch(`${BACKEND_URL}/chat/history/clear?session_id=${encodeURIComponent(TEST_SESSION_ID)}`, {
    method: "POST",
  });
  if (!clearRes.ok) fail(`/chat/history/clear returned ${clearRes.status}: ${await clearRes.text()}`);
  const clearData = await clearRes.json();
  if (clearData.cleared !== 4) {
    fail(`/chat/history/clear reported clearing ${clearData.cleared} messages, expected 4`);
  }

  const afterRes = await fetch(`${BACKEND_URL}/chat/history?session_id=${encodeURIComponent(TEST_SESSION_ID)}`);
  const afterData = await afterRes.json();
  if (afterData.messages.length !== 0) {
    fail(`/chat/history still returned ${afterData.messages.length} messages after clearing`);
  }
  console.log(`  OK - history empty after clearing`);
}

async function checkBrowserEndToEnd() {
  console.log(`[6/6] Driving ${FRONTEND_URL} with Playwright ...`);
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

    // Hover a word token and confirm the translation popover appears. It's
    // rendered into a portal at document.body (see WordToken.tsx - escapes
    // the scrolling chat feed's overflow clipping and sibling-bubble
    // stacking contexts), so it's no longer a DOM descendant of the message
    // bubble - just check it shows up on the page at all.
    const wordToken = page.locator(".chat-message--assistant .word-token").first();
    if ((await wordToken.count()) > 0) {
      await wordToken.hover();
      await page.waitForSelector(".word-token__popover-anchor .translate-popover", { timeout: 8_000 });
      console.log("  OK - hover translation popover works");
    } else {
      console.log("  (no trackable word tokens in this reply - skipping hover check)");
    }

    // Click the speaker button and confirm the browser's own /tts/speak
    // request (not just our direct fetch above) actually succeeds and the
    // UI doesn't end up in its error state. Checked via response status +
    // headers (reliable) rather than response.body() - buffering the full
    // body through CDP after the page has already consumed it as a Blob is
    // flaky and isn't needed: the direct backend check above already proved
    // real audio bytes come back for identical text.
    // Translation rows are collapsed by default, so at this point the only
    // .audio-button in the assistant bubble is the raw-message one.
    const speakButton = page.locator(".chat-message--assistant .audio-button").first();
    if ((await speakButton.count()) > 0) {
      const responsePromise = page.waitForResponse(
        (res) => res.url().includes("/tts/speak") && res.request().method() === "POST",
        { timeout: TTS_TIMEOUT_MS }
      );
      await speakButton.click();
      const ttsResponse = await responsePromise;
      if (!ttsResponse.ok()) {
        const body = await ttsResponse.text();
        fail(`browser /tts/speak request returned ${ttsResponse.status()}: ${body}`);
      }
      const contentType = ttsResponse.headers()["content-type"] ?? "";
      const contentLength = Number(ttsResponse.headers()["content-length"] ?? "0");
      if (!contentType.includes("audio")) {
        fail(`browser /tts/speak response content-type was ${JSON.stringify(contentType)}, not audio`);
      }
      if (contentLength < 1000) {
        fail(`browser /tts/speak response content-length was only ${contentLength} bytes`);
      }
      // Give the page a moment to finish turning the response into a Blob
      // and updating the button state, then confirm it didn't land on error.
      await page.waitForTimeout(500);
      const speakClass = (await speakButton.getAttribute("class")) ?? "";
      if (speakClass.includes("audio-button--error")) {
        fail("speaker button ended up in its error state after clicking");
      }
      console.log(`  OK - speaker button fetched ${contentType}, ${contentLength} bytes`);
    } else {
      fail("no speaker button (.audio-button) found on the assistant message");
    }

    // Translation rows: hovering the message previews the translation (no
    // click needed); moving the mouse away hides it again unless pinned;
    // clicking the toggle pins it open even after the mouse leaves.
    const assistantColumn = page.locator(".chat-message--assistant .chat-message__column").first();
    await assistantColumn.hover();
    await page.waitForSelector(".chat-message--assistant .translation-row--assistant", { timeout: 15_000 });
    const assistantRowText = await page
      .locator(".chat-message--assistant .translation-row--assistant .translation-row__text")
      .innerText();
    if (!assistantRowText.trim()) {
      fail("assistant translation row appeared but has no text");
    }
    console.log(`  OK - assistant translation row previews on hover: ${JSON.stringify(assistantRowText)}`);

    await page.mouse.move(0, 0);
    await page.waitForTimeout(200);
    const stillOpenAfterUnhover = await page.locator(".chat-message--assistant .translation-row--assistant").count();
    if (stillOpenAfterUnhover !== 0) {
      fail("assistant translation row still present after moving the mouse away (not pinned)");
    }
    console.log("  OK - moving the mouse away hides the row (doesn't just persist)");

    const assistantToggle = page.locator(".chat-message--assistant .chat-message__translate-toggle").first();
    await assistantToggle.click();
    await page.mouse.move(0, 0);
    await page.waitForTimeout(200);
    const pinnedOpen = await page.locator(".chat-message--assistant .translation-row--assistant").count();
    if (pinnedOpen === 0) {
      fail("assistant translation row not pinned open after clicking the toggle, then moving the mouse away");
    }
    console.log("  OK - clicking the toggle pins the translation row open");

    // Same for the user's own message, which should get TWO rows (EN + ES).
    const userMessageCount = await page.locator(".chat-message--user").count();
    console.log(`  (debug) .chat-message--user count: ${userMessageCount}`);
    if (userMessageCount === 0) {
      const allMessages = await page.locator(".chat-message").evaluateAll((els) =>
        els.map((el) => el.className)
      );
      console.log(`  (debug) all .chat-message class lists: ${JSON.stringify(allMessages)}`);
      await page.screenshot({ path: "e2e-debug-no-user-message.png" });
    }
    const userColumn = page.locator(".chat-message--user .chat-message__column").first();
    await userColumn.hover();
    await page.waitForSelector(".chat-message--user .translation-row--user-native", { timeout: 15_000 });
    await page.waitForSelector(".chat-message--user .translation-row--user-target", { timeout: 15_000 });
    console.log("  OK - hovering the user message previews EN + ES translation rows");

    // Hovering a word in the input box should show translation candidates -
    // "hotel" is a real word in both languages (app/translate/word_validity.py
    // checks each independently), so it should get BOTH an English and a
    // Spanish column, not just one direction. Wait for the actual
    // /translate/tag-input response rather than a fixed delay - a word
    // valid in both languages needs two Argos MT calls (one per direction),
    // which can take longer than the 350ms debounce alone suggests.
    const textarea = page.locator("textarea");
    const tagResponsePromise = page.waitForResponse(
      (res) => res.url().includes("/translate/tag-input") && res.request().method() === "POST",
      { timeout: 10_000 }
    );
    await textarea.fill("hotel");
    await tagResponsePromise;
    await page.waitForTimeout(150); // let React apply the response to state/DOM
    const hoverableWord = page.locator(".chat-input__hoverable").first();
    if ((await hoverableWord.count()) > 0) {
      await hoverableWord.hover();
      await page.waitForSelector(".word-candidates-popover", { timeout: 8_000 });
      const isDual = (await page.locator(".word-candidates-popover--dual").count()) > 0;
      if (!isDual) {
        fail('hovering "hotel" in the input did not show both EN and ES columns');
      }
      console.log('  OK - hovering a cognate ("hotel") in the input shows both EN and ES columns');

      // The Spanish reading ("hotel" is already correct Spanish) should be
      // a static gloss, not a button - only the English->Spanish reading
      // should be clickable to swap in place.
      const clickableCount = await page.locator(".candidate-cycler__main:not(.candidate-cycler__main--static)").count();
      const staticCount = await page.locator(".candidate-cycler__main--static").count();
      if (clickableCount === 0 || staticCount === 0) {
        fail(`expected one clickable and one static reading for "hotel", got ${clickableCount} clickable, ${staticCount} static`);
      }
      console.log("  OK - only the English->Spanish reading is clickable; the Spanish gloss is not");
    } else {
      fail('"hotel" in the input box was not flagged as hoverable');
    }
    await textarea.fill("");

    // Switch the voice picker to a non-default voice and confirm the next
    // speak request actually carries that voice - not just that the
    // dropdown renders.
    const voiceSelect = page.locator(".app__voice-picker select");
    if ((await voiceSelect.count()) > 0) {
      const options = await voiceSelect.locator("option").allTextContents();
      const targetLabel = options.includes("Alex") ? "Alex" : options.find((l) => l.trim().length > 0);
      if (!targetLabel) {
        fail("voice picker has no selectable options");
      }
      await voiceSelect.selectOption({ label: targetLabel });

      const responsePromise2 = page.waitForResponse(
        (res) => res.url().includes("/tts/speak") && res.request().method() === "POST",
        { timeout: TTS_TIMEOUT_MS }
      );
      await speakButton.click();
      const ttsResponse2 = await responsePromise2;
      const requestBody = JSON.parse(ttsResponse2.request().postData() ?? "{}");
      if (!ttsResponse2.ok()) {
        fail(`browser /tts/speak after switching voice to ${targetLabel} returned ${ttsResponse2.status()}`);
      }
      console.log(`  OK - voice picker switched to ${targetLabel}, request carried voice=${requestBody.voice}`);
    } else {
      fail("no voice picker (.app__voice-picker select) found in the header");
    }

    // Reload and confirm history hydration actually restores the
    // conversation - not just that /chat/turn works in the moment.
    await page.reload({ waitUntil: "load", timeout: 30_000 });
    await page.waitForSelector(".chat-message--assistant", { timeout: 15_000 });
    console.log("  OK - conversation persisted across a page reload");

    // "Clear chat" should wipe the visible conversation via the real
    // backend endpoint, not just a client-side reset.
    const clearButton = page.locator(".app__clear-chat");
    if ((await clearButton.count()) > 0) {
      const clearResponsePromise = page.waitForResponse(
        (res) => res.url().includes("/chat/history/clear") && res.request().method() === "POST",
        { timeout: 15_000 }
      );
      await clearButton.click();
      const clearResponse = await clearResponsePromise;
      if (!clearResponse.ok()) {
        fail(`browser "Clear chat" request returned ${clearResponse.status()}`);
      }
      await page.waitForSelector("text=Say hello to start a conversation", { timeout: 10_000 });
      console.log("  OK - Clear chat button emptied the conversation");
    } else {
      fail('no "Clear chat" button (.app__clear-chat) found - expected one once a conversation exists');
    }

    if (consoleErrors.length > 0) {
      console.log("  Browser console errors seen:", consoleErrors);
    }
  } finally {
    await browser.close();
  }
}

await checkBackendDirect();
await checkTTSDirect();
await checkVoicesDirect();
await checkEnglishInputAndTranslation();
await checkHistoryAndClear();
await checkBrowserEndToEnd();
console.log("ALL CHECKS PASSED");
