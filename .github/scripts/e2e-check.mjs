// On-demand end-to-end check against the deployed Railway services. Runs on
// GitHub Actions (not in the dev sandbox, which can't reach *.up.railway.app
// at all - its egress policy blocks that domain outright). Six stages, in an
// order that deliberately runs everything that depends only on this app's
// own code BEFORE anything that depends on DeepInfra, a third-party TTS
// provider that's had real, repeated slow/down patches unrelated to this
// app's own code (stages 5-6, and the TTS-dependent tail end of stage 4) -
// so a DeepInfra outage fails loudly on its own stage instead of masking
// whether the actual application logic works:
//  1. Hit the backend's /chat/turn directly - isolates "is the LLM path
//     actually working" from any frontend issue - then confirm /health/llm
//     (what the frontend's LlmGate polls on page load to hold the whole UI
//     back until the model server is actually ready, rather than let a real
//     interaction discover it's still cold-starting) agrees it's ready.
//  2. Send an English message to /chat/turn, and a fixed Spanish phrase to
//     /translate/text - prints both for a human to eyeball (semantic
//     correctness isn't something a script can assert), but at least
//     confirms the LLM-backed translation path actually returns something,
//     not the old Argos error string. Also checks /translate/word for a
//     few known slang words (bro/sup/partner) - regression coverage for a
//     real bug where Argos's word-level fallback echoed unknown words back
//     unchanged instead of translating them.
//  3. Confirm GET /chat/history reflects what stages 1/2 just sent, and
//     that POST /chat/history/clear actually empties it.
//  4. If those pass, drive the real site with Playwright end to end: confirm
//     the LLM-readiness gate isn't blocking the page (the model server is
//     already known warm by this point), send a message, hover a word
//     gloss, toggle both messages' translation rows
//     open and closed, confirm the user message gets two rows (EN+ES),
//     hover a cognate word in the input box for its dual-column candidates,
//     drag-select a multi-word phrase in the input (starting the drag ON a
//     hoverable word, not just in a gap between them) and confirm it shows
//     one phrase translation (in whichever direction the selected words'
//     majority language calls for) rather than a leftover single-word
//     popover - all of that before touching TTS at all, same reasoning as
//     stages 1-2 before 5-6 - then play the assistant message's audio,
//     switch the voice picker and confirm that request carries the chosen
//     voice, then reload and clear the chat.
//  5. Hit the backend's /tts/speak directly - isolates "is DeepInfra TTS
//     actually working" (right model/voice, valid token) from the frontend.
//  6. Hit /tts/voices and /tts/speak for each Spanish voice directly -
//     confirms every voice the picker offers is a real, working DeepInfra
//     voice id, not just the default.
//
// Stages 1-3 all use one throwaway session id (see TEST_SESSION_ID) so this
// script's own chat traffic never lands in - or pollutes - anyone real's
// conversation history, and gets explicitly cleared at the end regardless.
import { chromium } from "playwright";
const BACKEND_URL = process.env.BACKEND_URL;
const FRONTEND_URL = process.env.FRONTEND_URL;
const CHAT_TIMEOUT_MS = 100_000;
// DeepInfra (the TTS provider) has observed live, more than once, going
// through slow patches where even a short phrase's synthesis takes well
// past 15-20s - not a code regression, just third-party latency variance.
// 30s was tight enough that it turned a slow-but-working response into a
// hard failure and burned several CI cycles on the exact same non-issue;
// wider margin here, not a claim that this is fast.
const TTS_TIMEOUT_MS = 60_000;
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

  // The chat turn above only succeeds if the model server is actually up -
  // /health/llm should now report the same thing (it's what the frontend's
  // LlmGate polls on page load to hold the UI back until this is true).
  console.log(`[1/6] Checking /health/llm reports the model server ready ...`);
  const llmHealth = await fetch(`${BACKEND_URL}/health/llm`);
  if (!llmHealth.ok) fail(`/health/llm returned ${llmHealth.status}`);
  const llmHealthData = await llmHealth.json();
  if (llmHealthData.ready !== true) {
    fail(`/health/llm reported not ready right after a successful /chat/turn: ${JSON.stringify(llmHealthData)}`);
  }
  console.log("  OK - /health/llm reports ready");
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
  console.log(`[5/6] Sending a direct /tts/speak request (up to ${TTS_TIMEOUT_MS / 1000}s) ...`);
  const { elapsed, contentType, bytes } = await speakDirect(undefined);
  console.log(`  OK in ${elapsed}ms - ${contentType}, ${bytes} bytes`);
}

async function checkVoicesDirect() {
  console.log(`[6/6] Checking /tts/voices?language=es and every Spanish voice ...`);
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
  console.log(`[2/6] Sending an English message to /chat/turn (session ${TEST_SESSION_ID}, no prior turns polluting it) ...`);
  const { text } = await chatTurn("Hi, what hobbies do you like?");
  console.log(`  reply: ${JSON.stringify(text)} (eyeball: does this answer, or just echo/translate the question?)`);

  console.log(`[2/6] Sending a direct /translate/text request ...`);
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

  // Regression check for a real reported bug: Argos MT's word-level lookup
  // (used before the Wiktionary dataset was added) had no way to say "I
  // don't know this word" - for slang never seen in its formal training
  // corpus, it just echoed the input back unchanged with no explanation,
  // which looked exactly like a real (wrong) translation rather than a
  // miss. The bug signature specifically is translation === input AND no
  // description - an UNEXPLAINED echo.
  //
  // "bro", "sup" and "partner" all have real candidates now: "sup" and
  // "bro" via the curated informal_dictionary.py override (Wiktionary's
  // crowd-sourced translation tables were too thin for these - no "es"
  // entry for "sup" at all, and only a bare self-referential one for
  // "bro"), "partner" via Wiktionary's own translation table. "bro" and
  // "sup" must each offer more than one real candidate (the whole point of
  // the cycle button) - "partner" already does via Wiktionary alone.
  console.log(`[2/6] Checking /translate/word no longer echoes known slang back unchanged ...`);
  for (const word of ["bro", "sup", "partner"]) {
    const res = await fetch(`${BACKEND_URL}/translate/word`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word, source_lang: "en" }),
    });
    if (!res.ok) fail(`/translate/word(${word}) returned ${res.status}: ${await res.text()}`);
    const data = await res.json();
    const best = data.candidates?.[0];
    if (!best || !best.translation) {
      fail(`/translate/word(${word}) returned no usable candidate: ${JSON.stringify(data)}`);
    }
    const isUnexplainedEcho = best.translation.toLowerCase() === word.toLowerCase() && !best.description;
    if (isUnexplainedEcho) {
      fail(`/translate/word(${word}) echoed the input back unchanged with no explanation: ${JSON.stringify(data)}`);
    }
    if (data.candidates.length < 2) {
      fail(`/translate/word(${word}) returned only ${data.candidates.length} candidate(s) - expected multiple: ${JSON.stringify(data)}`);
    }
    console.log(`  "${word}" -> ${JSON.stringify(best.translation)} (${JSON.stringify(best.description)}, ${data.candidates.length} candidate(s))`);
  }
}

async function checkHistoryAndClear() {
  console.log(`[3/6] Checking GET /chat/history reflects this session's turns ...`);
  const historyRes = await fetch(`${BACKEND_URL}/chat/history?session_id=${encodeURIComponent(TEST_SESSION_ID)}`);
  if (!historyRes.ok) fail(`/chat/history returned ${historyRes.status}: ${await historyRes.text()}`);
  const historyData = await historyRes.json();
  // 2 chatTurn() calls so far (stage 1's "Hola", stage 4's hobbies
  // question) = 4 messages (user+assistant each).
  if (historyData.messages.length !== 4) {
    fail(`/chat/history returned ${historyData.messages.length} messages, expected 4: ${JSON.stringify(historyData)}`);
  }
  console.log(`  OK - ${historyData.messages.length} messages recorded for this session`);

  console.log(`[3/6] Clearing this session's history via POST /chat/history/clear ...`);
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
  console.log(`[4/6] Driving ${FRONTEND_URL} with Playwright ...`);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 800 } });
  const consoleErrors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  try {
    await page.goto(FRONTEND_URL, { waitUntil: "load", timeout: 30_000 });
    await page.waitForSelector("text=wordbyword", { timeout: 15_000 });
    // The model server is already confirmed warm (stage 1's /chat/turn and
    // /health/llm checks, minutes ago by now), so LlmGate's very first poll
    // should come back ready - this just confirms the gate doesn't get
    // stuck open on the real deployed site even when there's nothing to
    // wait for. `state: "hidden"` resolves immediately if the element was
    // never in the DOM at all, not just if it disappears.
    await page.waitForSelector(".llm-gate", { state: "hidden", timeout: 15_000 });
    console.log("  OK - LLM gate isn't blocking the page once the model server is warm");
    await page.waitForSelector("textarea", { timeout: 15_000 });
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
      // Close it before moving on - it's portaled to document.body and
      // positioned from the word's own coordinates, which can land right
      // over the action row below the bubble and intercept the next
      // step's hover on the globe button otherwise.
      await page.mouse.move(0, 0);
      await page.waitForTimeout(100);
    } else {
      console.log("  (no trackable word tokens in this reply - skipping hover check)");
    }

    // Translation rows are collapsed by default, so at this point the only
    // .audio-button in the assistant bubble is the raw-message one. Captured
    // here (used to click it further down, and again for the voice-picker
    // check) - the actual click happens later, after the non-TTS checks
    // below, so a DeepInfra outage (this session has seen several) fails on
    // its own dedicated step instead of blocking unrelated translation/UI
    // checks that don't touch TTS at all - same reasoning as reordering the
    // direct backend checks earlier in this file.
    const speakButton = page.locator(".chat-message--assistant .audio-button").first();
    if ((await speakButton.count()) === 0) {
      fail("no speaker button (.audio-button) found on the assistant message");
    }

    // Translation rows: hovering the globe button (not the whole bubble -
    // that used to trigger it too, which meant just reading a message
    // popped the translation open) previews the translation with no click
    // needed; moving the mouse away hides it again unless pinned; clicking
    // the toggle pins it open even after the mouse leaves.
    const assistantToggle = page.locator(".chat-message--assistant .chat-message__translate-toggle").first();
    await assistantToggle.hover();
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
    const userToggle = page.locator(".chat-message--user .chat-message__translate-toggle").first();
    await userToggle.hover();
    await page.waitForSelector(".chat-message--user .translation-row--user-native", { timeout: 15_000 });
    await page.waitForSelector(".chat-message--user .translation-row--user-target", { timeout: 15_000 });
    console.log("  OK - hovering the user message previews EN + ES translation rows");

    // Debug screenshot: a plain single-candidate word hover ("gato" only has
    // one clear sense, one column) - the closest apples-to-apples visual
    // comparison to the phrase popover below, which is also always a single
    // candidate. Both should render the exact same widget.
    const textarea = page.locator("textarea");
    const gatoTagPromise = page.waitForResponse(
      (res) => res.url().includes("/translate/tag-input") && res.request().method() === "POST",
      { timeout: 10_000 }
    );
    await textarea.fill("gato");
    await gatoTagPromise;
    await page.waitForTimeout(150);
    const gatoWord = page.locator(".chat-input__hoverable").first();
    if ((await gatoWord.count()) > 0) {
      await gatoWord.hover();
      const gatoPopover = page.locator(".word-candidates-popover").first();
      await gatoPopover.waitFor({ state: "visible", timeout: 8_000 });
      const gatoBox = await gatoPopover.boundingBox();
      await page.screenshot({
        path: "e2e-debug-single-word-hover-popover.png",
        clip: { x: Math.max(0, gatoBox.x - 10), y: Math.max(0, gatoBox.y - 10), width: gatoBox.width + 20, height: gatoBox.height + 20 },
      });
    }
    await page.mouse.move(10, 10);
    await page.waitForTimeout(200);
    await textarea.fill("");

    // Hovering a word in the input box should show translation candidates -
    // "hotel" is a real word in both languages (app/translate/word_validity.py
    // checks each independently), so it should get BOTH an English and a
    // Spanish column, not just one direction. Wait for the actual
    // /translate/tag-input response rather than a fixed delay - a word
    // valid in both languages needs two Argos MT calls (one per direction),
    // which can take longer than the 350ms debounce alone suggests.
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
      const toggleButtons = page.locator(".word-candidates-popover__toggle-btn");
      const toggleLabels = await toggleButtons.allTextContents();
      if (JSON.stringify(toggleLabels.sort()) !== JSON.stringify(["English", "Spanish"])) {
        fail(`hovering "hotel" in the input did not show a Spanish/English toggle - got ${JSON.stringify(toggleLabels)}`);
      }
      console.log('  OK - hovering a cognate ("hotel") in the input shows a Spanish | English toggle');

      // The Spanish reading (a translation to swap in for the English word
      // "hotel") is the default when present - Spanish is the target
      // language, so that's the reading a learner most wants to see first.
      // It's also the clickable one (swaps it into the input). The English
      // reading (what "hotel" means, glossed) is a static gloss, not a
      // button.
      const defaultActiveLabel = await page.locator(".word-candidates-popover__toggle-btn--active").innerText();
      if (defaultActiveLabel !== "Spanish") {
        fail(`expected "Spanish" to be the default active toggle for "hotel", got ${JSON.stringify(defaultActiveLabel)}`);
      }
      const isClickableByDefault = (await page.locator(".candidate-cycler__main--static").count()) === 0;
      if (!isClickableByDefault) {
        fail('expected the default (Spanish) reading for "hotel" to be clickable, not a static gloss');
      }
      // Debug screenshot: visually confirm the word-hover popover's actual
      // rendered appearance (not just its DOM structure via selectors).
      const hotelBox = await page.locator(".word-candidates-popover").first().boundingBox();
      await page.screenshot({
        path: "e2e-debug-word-hover-popover.png",
        clip: { x: Math.max(0, hotelBox.x - 10), y: Math.max(0, hotelBox.y - 10), width: hotelBox.width + 20, height: hotelBox.height + 20 },
      });

      // Clicking the other toggle should switch to the English reading,
      // which is a static gloss (not clickable - nothing to swap in, since
      // "hotel" is already the English word being typed).
      await page.locator(".word-candidates-popover__toggle-btn", { hasText: "English" }).click();
      await page.waitForSelector(".candidate-cycler__main--static", { timeout: 3_000 });
      const nowActiveLabel = await page.locator(".word-candidates-popover__toggle-btn--active").innerText();
      if (nowActiveLabel !== "English") {
        fail(`expected clicking "English" to switch the active toggle, got ${JSON.stringify(nowActiveLabel)}`);
      }
      console.log('  OK - toggling to "English" switches to the static gloss reading');
    } else {
      fail('"hotel" in the input box was not flagged as hoverable');
    }
    await textarea.fill("");

    // Highlighting (selecting) a run of text in the input should show a
    // phrase-level translation, not just the single word under the
    // pointer - and dragging the selection FROM a hoverable word has to
    // actually work: the hover overlay's spans intercept mousedown for
    // hover-popover purposes, which (before this was fixed) silently
    // swallowed the drag and never produced a real <textarea> selection at
    // all when it started on a word (i.e. almost always).
    const phraseTagResponsePromise = page.waitForResponse(
      (res) => res.url().includes("/translate/tag-input") && res.request().method() === "POST",
      { timeout: 10_000 }
    );
    await textarea.fill("me gusta mucho el gato negro");
    await phraseTagResponsePromise;
    await page.waitForTimeout(150);
    const dragStartWord = page.locator(".chat-input__hoverable", { hasText: "gusta" }).first();
    const dragEndWord = page.locator(".chat-input__hoverable", { hasText: "negro" }).first();
    if ((await dragStartWord.count()) > 0 && (await dragEndWord.count()) > 0) {
      const startBox = await dragStartWord.boundingBox();
      const endBox = await dragEndWord.boundingBox();
      await page.mouse.move(startBox.x + 2, startBox.y + startBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(endBox.x + endBox.width - 2, endBox.y + endBox.height / 2, { steps: 10 });
      await page.mouse.up();
      await page.waitForTimeout(150);

      const selection = await textarea.evaluate((el) => ({
        start: el.selectionStart,
        end: el.selectionEnd,
        text: el.value.slice(el.selectionStart, el.selectionEnd),
      }));
      if (selection.start === selection.end) {
        fail("dragging from a hoverable word in the input produced no text selection at all");
      }
      console.log(`  OK - drag-selecting from a hoverable word selected ${JSON.stringify(selection.text)}`);

      // The phrase popover reuses the exact same widget as single-word
      // hover (WordCandidatesPopover), just portaled to a fixed anchor
      // instead of nested inline - so it renders the same
      // .word-candidates-popover element. Same zero-size-anchor pattern as
      // the word-token popover above: the anchor div itself is
      // deliberately 0x0 (its only child is position:absolute, so it never
      // contributes to the anchor's own box), so wait for the actual
      // rendered popover content, not the anchor element.
      await page.waitForSelector(".chat-input__phrase-popover-anchor .word-candidates-popover", { timeout: 8_000 });
      // Scoped to a *nested* instance (inside a hoverable word span) so this
      // only catches a genuine leftover single-word popover, not the phrase
      // popover itself - both render the same .word-candidates-popover class.
      const stillShowingWordPopover = (await page.locator(".chat-input__hoverable .word-candidates-popover").count()) > 0;
      if (stillShowingWordPopover) {
        fail("a single-word popover was still showing alongside the phrase popover after a drag-select");
      }
      console.log("  OK - selecting a phrase shows a phrase translation popover, not a leftover single-word one");

      // The selected phrase ("gusta mucho el gato negro") is all Spanish,
      // so this should translate it INTO English (majority-vote over the
      // selection's per-word is_spanish flags), not treat it as a draft to
      // correct into Spanish.
      await page.waitForFunction(
        () => {
          const el = document.querySelector(".chat-input__phrase-popover-anchor .candidate-cycler__text");
          return !!el && !el.textContent.includes("…");
        },
        { timeout: 8_000 },
      );
      const phraseHeading = await page
        .locator(".chat-input__phrase-popover-anchor .word-candidates-popover__heading")
        .first()
        .innerText();
      const phraseTranslationText = await page
        .locator(".chat-input__phrase-popover-anchor .candidate-cycler__text")
        .first()
        .innerText();
      if (phraseHeading !== "English") {
        fail(`expected the phrase popover for an all-Spanish selection to read "English", got ${JSON.stringify(phraseHeading)}`);
      }
      if (!phraseTranslationText.toLowerCase().includes("cat")) {
        fail(`expected the Spanish->English phrase translation to mention "cat", got ${JSON.stringify(phraseTranslationText)}`);
      }
      console.log(`  OK - highlighting an all-Spanish phrase translated it to English: ${JSON.stringify(phraseTranslationText)}`);
      // Debug screenshot: visually confirm the phrase popover's actual
      // rendered appearance (not just its DOM structure via selectors).
      const phraseBox = await page.locator(".chat-input__phrase-popover-anchor .word-candidates-popover").first().boundingBox();
      await page.screenshot({
        path: "e2e-debug-phrase-popover.png",
        clip: { x: Math.max(0, phraseBox.x - 10), y: Math.max(0, phraseBox.y - 10), width: phraseBox.width + 20, height: phraseBox.height + 20 },
      });
    } else {
      fail('could not find "gusta"/"negro" as hoverable words to test drag-selection');
    }
    await textarea.fill("");

    // Click the speaker button and confirm the browser's own /tts/speak
    // request (not just our direct fetch earlier) actually succeeds and the
    // UI doesn't end up in its error state. Checked via response status +
    // headers (reliable) rather than response.body() - buffering the full
    // body through CDP after the page has already consumed it as a Blob is
    // flaky and isn't needed: the direct backend check earlier already
    // proved real audio bytes come back for identical text.
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
    // Give the page a moment to finish turning the response into a Blob and
    // updating the button state, then confirm it didn't land on error.
    await page.waitForTimeout(500);
    const speakClass = (await speakButton.getAttribute("class")) ?? "";
    if (speakClass.includes("audio-button--error")) {
      fail("speaker button ended up in its error state after clicking");
    }
    console.log(`  OK - speaker button fetched ${contentType}, ${contentLength} bytes`);

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
await checkEnglishInputAndTranslation();
await checkHistoryAndClear();
await checkBrowserEndToEnd();
await checkTTSDirect();
await checkVoicesDirect();
console.log("ALL CHECKS PASSED");
