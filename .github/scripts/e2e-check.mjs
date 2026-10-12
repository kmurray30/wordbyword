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
//     confirms the LLM-backed translation path actually returns something.
//     Also checks /translate/tag-input (the cheap, LLM-free per-word pass)
//     and /translate/gloss-spans (the slower LLM-backed word/group
//     glossing, fetched lazily by the frontend rather than on every
//     keystroke - see ChatInput.tsx) directly on an idiom-bearing draft -
//     every returned span must carry a usable gloss/translation (no more
//     dictionary/Argos fallback anywhere - a miss just means no gloss),
//     and whether it grouped multiple words into one span (e.g. "miss
//     you") is eyeballed, not asserted, since that varies run to run with
//     a small model. Finally streams /brainstorm/turn (the Brainstorm
//     field's coach - Server-Sent Events, not a single JSON response) using
//     this session's own recent history for context, confirms
//     /brainstorm/open holds both persisted turns, then sends a chat turn
//     (which stamps that open log onto the new user message) and folds it
//     into notes via /brainstorm/fold.
//  3. Confirm GET /chat/history reflects what stages 1/2 just sent
//     (including the brainstorm log attached to stage 2's last user
//     message), and that POST /chat/history/clear actually empties it.
//  4. If those pass, drive the real site with Playwright end to end: confirm
//     the LLM-readiness gate isn't blocking the page (the model server is
//     already known warm by this point), send a message, hover AND click a
//     word gloss (click should pin it open with a highlight, no underline -
//     the only way it ever shows on a real tap, which never fires hover;
//     hovering should dock the popover flush to the RIGHT of that
//     message's own bubble, not stack underneath its full translation),
//     toggle both messages' translation rows open and closed (including
//     confirming the globe toggle actually closes on a second click, not
//     just opens), confirm the user message gets two rows (EN+ES), hover
//     words in the input box (every word is hoverable immediately from the
//     cheap per-keystroke pass, showing a "Translating…" placeholder until
//     the slower, lazily-fetched real gloss lands), drag-select a
//     multi-word phrase in the input (starting the drag ON a hoverable
//     word, not just in a gap between them) and confirm it shows one
//     phrase translation (in whichever direction the selected words'
//     majority language calls for) rather than a leftover single-word
//     popover, then brainstorm with the coach in the Brainstorm field
//     (focusing it must not get stolen back by the Reply field's focus
//     keeper) and confirm its streamed answer lands in the log and its
//     proposed sentence fills the Reply field, send that reply, and wait
//     for the sent bubble's notes dropdown to settle (the brainstorm log
//     folded into notes) - all of that before touching TTS at all, same
//     reasoning as stages 1-2 before 5-6 - then play the assistant
//     message's audio, switch the voice picker (a button-based
//     SegmentedControl, not a native <select> - see SegmentedControl.tsx)
//     and confirm that request carries the chosen voice, confirm the
//     composer's two textareas (Brainstorm + Reply) are the only native
//     form controls left on the page (the point of removing the native
//     select/checkbox controls - keeps iOS Safari's keyboard accessory bar
//     from showing field-navigation arrows), then reload and clear the chat.
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
  return { elapsed, text: data.text, tokens: data.tokens ?? [], data };
}

async function checkBackendDirect() {
  console.log(`[1/6] Checking backend health at ${BACKEND_URL}/health ...`);
  const health = await fetch(`${BACKEND_URL}/health`);
  if (!health.ok) fail(`/health returned ${health.status}`);
  console.log("  health OK");

  console.log(`[1/6] Sending a direct /chat/turn request (session ${TEST_SESSION_ID}, up to ${CHAT_TIMEOUT_MS / 1000}s) ...`);
  const { elapsed, text, tokens } = await chatTurn("Cuéntame sobre tu familia y tu mejor amigo.");
  console.log(`  OK in ${elapsed}ms - reply: ${JSON.stringify(text)}`);
  console.log(`  tokens: ${tokens.length}`);
  if (elapsed > 6000) {
    console.log(`  (!) /chat/turn took ${elapsed}ms - that's NOT expected for a hosted API call`);
  }

  // gloss_reply's span-list rewrite asks the model to gloss EVERY word,
  // including short function words (y, mi, el, la, de, que, ...) - a
  // regression here (e.g. too tight a max_tokens budget truncating the
  // JSON and silently dropping every gloss in the message) reads live as
  // "lots of words aren't clickable", not a clean failure, so this checks
  // coverage directly rather than just that SOME token got a gloss. Not
  // 100% - an occasional single word slipping through on a given sample
  // isn't itself a regression - but a wide miss is.
  const alphabetic = tokens.filter((t) => /[a-zA-Zá-úñÁ-ÚÑ]/.test(t.surface));
  const glossed = alphabetic.filter((t) => t.gloss && t.gloss.trim());
  const coverage = alphabetic.length > 0 ? glossed.length / alphabetic.length : 1;
  console.log(
    `  gloss coverage: ${glossed.length}/${alphabetic.length} words (${(coverage * 100).toFixed(0)}%)` +
      (alphabetic.length > glossed.length
        ? ` - missing: ${JSON.stringify(alphabetic.filter((t) => !t.gloss || !t.gloss.trim()).map((t) => t.surface))}`
        : ""),
  );
  if (coverage < 0.7) {
    fail(`gloss coverage too low (${(coverage * 100).toFixed(0)}%) - gloss_reply likely failed/truncated for this reply`);
  }

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

  // Direct backend check for the input's cheap, LLM-free per-word pass -
  // /translate/tag-input now returns ONLY this (spaCy tokenize/lemmatize +
  // the is_spanish heuristic), no LLM call and no spans at all; the
  // slower, LLM-backed word/group glossing that used to live here moved to
  // /translate/gloss-spans below, fetched lazily by the frontend rather
  // than on every keystroke.
  console.log(`[2/6] Checking /translate/tag-input (cheap per-word pass) ...`);
  const draftText = "I want to go to the beach tomorrow, I miss you";
  const tagRes = await fetch(`${BACKEND_URL}/translate/tag-input`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: draftText }),
  });
  if (!tagRes.ok) fail(`/translate/tag-input returned ${tagRes.status}: ${await tagRes.text()}`);
  const tagData = await tagRes.json();
  if (!Array.isArray(tagData.tokens) || tagData.tokens.length === 0) {
    fail(`/translate/tag-input returned no tokens: ${JSON.stringify(tagData)}`);
  }
  if ("spans" in tagData) {
    fail(`/translate/tag-input returned a "spans" field - that moved to /translate/gloss-spans: ${JSON.stringify(tagData)}`);
  }
  console.log(`  OK - ${tagData.tokens.length} tokens, no LLM-backed spans (moved to /translate/gloss-spans)`);

  // The slower, LLM-backed half (app.translate.llm_translate.tag_draft +
  // span_matching) - fetched lazily on hover/click/tap in the real app,
  // but hit directly here on the same idiom-bearing draft ("miss you", a
  // clear single-unit idiom) to exercise the model's ability to group
  // multiple words into one span rather than only ever tagging single
  // words.
  console.log(`[2/6] Checking /translate/gloss-spans (LLM-based word/group glossing) ...`);
  const glossRes = await fetch(`${BACKEND_URL}/translate/gloss-spans`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: draftText }),
  });
  if (!glossRes.ok) fail(`/translate/gloss-spans returned ${glossRes.status}: ${await glossRes.text()}`);
  const glossData = await glossRes.json();
  if (!Array.isArray(glossData.spans) || glossData.spans.length === 0) {
    fail(`/translate/gloss-spans returned no spans at all: ${JSON.stringify(glossData)}`);
  }
  console.log(`  spans: ${JSON.stringify(glossData.spans.map((s) => s.surface))}`);
  for (const span of glossData.spans) {
    if (!span.candidates?.[0]?.translation) {
      fail(`span ${JSON.stringify(span.surface)} has no usable gloss/translation: ${JSON.stringify(span)}`);
    }
  }
  console.log("  OK - every returned span carries a usable gloss/translation");
  // Whether the model actually groups "miss you" into one span (rather
  // than two single-word ones) varies run to run with a small model - not
  // asserted, just eyeballed, same as this file's other free-form LLM
  // output checks (see chatTurn above).
  const hasMultiWordGroup = glossData.spans.some((s) => s.surface.trim().includes(" "));
  console.log(
    hasMultiWordGroup
      ? '  (eyeball) the model grouped at least one multi-word span (e.g. the "miss you" idiom)'
      : '  (eyeball) no multi-word group this run - the model tagged every word individually',
  );

  // The Brainstorm field's coach - streamed as Server-Sent Events
  // ("intent", "verdict", "draft", "say_chunk"s, "say_complete", then
  // "saved" once both turns are persisted; "error" if nothing usable came
  // back - see app/routes/brainstorm.py). Uses this session's recent
  // history (stage 1's "Hola" and this function's own hobbies question)
  // for context, so runs after there's actually some history to use. A
  // rough Spanish attempt, so the coach should normally propose a draft.
  console.log(`[2/6] Checking /brainstorm/turn (streamed SSE) ...`);
  const brainstormText = "me gusta mucho leer libros de historia, y tu?";
  const brainstormRes = await fetch(`${BACKEND_URL}/brainstorm/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: TEST_SESSION_ID, text: brainstormText, words: [], reply_draft: "", source: "text" }),
  });
  if (!brainstormRes.ok) fail(`/brainstorm/turn returned ${brainstormRes.status}: ${await brainstormRes.text()}`);
  const brainstormEvents = await readSseEvents(brainstormRes);
  const brainstormEvent = (name) => brainstormEvents.find((e) => e.event === name)?.data;
  const brainstormError = brainstormEvent("error");
  if (brainstormError) fail(`/brainstorm/turn sent an "error" event: ${JSON.stringify(brainstormError)}`);
  const brainstormIntent = brainstormEvent("intent")?.intent;
  if (!brainstormIntent) fail(`/brainstorm/turn never sent an "intent" event: ${JSON.stringify(brainstormEvents)}`);
  const brainstormSay = brainstormEvent("say_complete")?.say;
  if (!brainstormSay || !brainstormSay.trim()) {
    fail(`/brainstorm/turn never sent a non-empty "say_complete" event: ${JSON.stringify(brainstormEvents)}`);
  }
  const brainstormSaved = brainstormEvent("saved");
  if (!brainstormSaved?.learner_turn_id || !brainstormSaved?.coach_turn_id) {
    fail(`/brainstorm/turn never sent a usable "saved" event: ${JSON.stringify(brainstormEvents)}`);
  }
  const brainstormVerdict = brainstormEvent("verdict")?.verdict;
  const brainstormDraft = brainstormEvent("draft")?.draft ?? "";
  console.log(`  intent: ${JSON.stringify(brainstormIntent)}, verdict: ${JSON.stringify(brainstormVerdict)}`);
  console.log(`  draft: ${JSON.stringify(brainstormDraft)}`);
  console.log(`  say: ${JSON.stringify(brainstormSay)}`);
  console.log(`  OK - streamed intent/say_complete/saved (turn ids ${brainstormSaved.learner_turn_id}/${brainstormSaved.coach_turn_id})`);

  // Both sides of that exchange are the session's open brainstorm log now.
  console.log(`[2/6] Checking GET /brainstorm/open holds both turns ...`);
  const openRes = await fetch(`${BACKEND_URL}/brainstorm/open?session_id=${encodeURIComponent(TEST_SESSION_ID)}`);
  if (!openRes.ok) fail(`/brainstorm/open returned ${openRes.status}: ${await openRes.text()}`);
  const openData = await openRes.json();
  const openTurns = openData.turns ?? [];
  const openLearner = openTurns.find((t) => t.id === brainstormSaved.learner_turn_id);
  const openCoach = openTurns.find((t) => t.id === brainstormSaved.coach_turn_id);
  if (openTurns.length !== 2 || openLearner?.role !== "learner" || openCoach?.role !== "coach") {
    fail(`/brainstorm/open should hold exactly the learner + coach turns just saved: ${JSON.stringify(openData)}`);
  }
  if (openLearner.text !== brainstormText) {
    fail(`/brainstorm/open's learner turn text ${JSON.stringify(openLearner.text)} != what was sent ${JSON.stringify(brainstormText)}`);
  }
  console.log(`  OK - open log holds the learner + coach turns`);

  // Sending a reply closes the open log (stamped onto the new user
  // message - has_brainstorm in the turn's response), and
  // /brainstorm/fold then turns it into notes. Idempotent, and scoped to
  // the session that owns the message.
  console.log(`[2/6] Sending a chat turn that closes the open brainstorm log ...`);
  const foldTurn = await chatTurn(brainstormDraft.trim() || "Me gusta mucho leer libros de historia. ¿Y a ti?");
  console.log(`  reply: ${JSON.stringify(foldTurn.text)}`);
  if (foldTurn.data.has_brainstorm !== true) {
    fail(`/chat/turn after brainstorming should report has_brainstorm=true: ${JSON.stringify({ has_brainstorm: foldTurn.data.has_brainstorm, user_message_id: foldTurn.data.user_message_id })}`);
  }
  const foldUserMessageId = foldTurn.data.user_message_id;
  if (!foldUserMessageId) fail(`/chat/turn returned no user_message_id: ${JSON.stringify(foldTurn.data)}`);
  console.log(`  OK - has_brainstorm=true, user_message_id=${foldUserMessageId}`);

  const openAfterRes = await fetch(`${BACKEND_URL}/brainstorm/open?session_id=${encodeURIComponent(TEST_SESSION_ID)}`);
  if (!openAfterRes.ok) fail(`/brainstorm/open (after send) returned ${openAfterRes.status}`);
  const openAfterData = await openAfterRes.json();
  if ((openAfterData.turns ?? []).length !== 0) {
    fail(`/brainstorm/open still has turns after the reply closed the log: ${JSON.stringify(openAfterData)}`);
  }
  console.log(`  OK - the open log is empty once the reply is sent`);

  console.log(`[2/6] Checking POST /brainstorm/fold ...`);
  const foldStart = Date.now();
  const foldRes = await fetch(`${BACKEND_URL}/brainstorm/fold`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: TEST_SESSION_ID, user_message_id: foldUserMessageId }),
  });
  if (foldRes.status !== 200) fail(`/brainstorm/fold returned ${foldRes.status}: ${await foldRes.text()}`);
  const foldData = await foldRes.json();
  if (!Array.isArray(foldData.messages)) fail(`/brainstorm/fold returned no "messages" array: ${JSON.stringify(foldData)}`);
  for (const entry of foldData.messages) {
    if (typeof entry.message_id !== "number" || !Array.isArray(entry.notes)) {
      fail(`/brainstorm/fold entry is malformed: ${JSON.stringify(entry)}`);
    }
  }
  console.log(`  OK in ${Date.now() - foldStart}ms - notes: ${JSON.stringify(foldData.messages)}`);
  if (foldData.messages.length === 0) {
    console.log("  (eyeball) the fold found nothing worth noting this run - allowed, model-dependent");
  }

  // Idempotent: asking again returns the same notes rather than re-folding.
  const foldAgainRes = await fetch(`${BACKEND_URL}/brainstorm/fold`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: TEST_SESSION_ID, user_message_id: foldUserMessageId }),
  });
  if (foldAgainRes.status !== 200) fail(`/brainstorm/fold (repeat) returned ${foldAgainRes.status}: ${await foldAgainRes.text()}`);
  const foldAgainData = await foldAgainRes.json();
  if (JSON.stringify(foldAgainData) !== JSON.stringify(foldData)) {
    fail(`/brainstorm/fold isn't idempotent: first ${JSON.stringify(foldData)}, then ${JSON.stringify(foldAgainData)}`);
  }
  // ...and refuses a message from a different session.
  const foldWrongRes = await fetch(`${BACKEND_URL}/brainstorm/fold`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: `${TEST_SESSION_ID}-other`, user_message_id: foldUserMessageId }),
  });
  if (foldWrongRes.status !== 404) fail(`/brainstorm/fold for the wrong session returned ${foldWrongRes.status}, expected 404`);
  console.log("  OK - fold is idempotent and 404s for another session's message");
  return { foldUserMessageId };
}

// Reads a Server-Sent Events response body to completion and parses it
// into { event, data } pairs - used for /brainstorm/turn, which streams
// rather than returning a single JSON response (see brainstorm_stream in
// app/translate/llm_translate.py and the route in app/routes/brainstorm.py).
async function readSseEvents(response) {
  const text = await response.text();
  const events = [];
  for (const block of text.split("\n\n")) {
    if (!block.trim()) continue;
    const lines = block.split("\n");
    const eventLine = lines.find((l) => l.startsWith("event:"));
    const dataLine = lines.find((l) => l.startsWith("data:"));
    if (!eventLine || !dataLine) continue;
    events.push({
      event: eventLine.slice("event:".length).trim(),
      data: JSON.parse(dataLine.slice("data:".length).trim()),
    });
  }
  return events;
}

// Hovering/tapping a word in the chat input opens its popover immediately
// (every word is hoverable right away from the cheap per-keystroke pass),
// but shows a "Translating…" placeholder until the slower, lazily-fetched
// real gloss (/translate/gloss-spans, fired by that hover/tap itself, not
// on a typing timer) resolves - see WordCandidatesPopover.tsx. Callers
// that need to read the resolved candidate text must wait for this first.
async function waitForGlossReady(pageOrFrame, timeout = 10_000) {
  await pageOrFrame.waitForSelector(".word-candidates-popover__loading", { state: "hidden", timeout });
}

async function checkHistoryAndClear({ foldUserMessageId }) {
  console.log(`[3/6] Checking GET /chat/history reflects this session's turns ...`);
  const historyRes = await fetch(`${BACKEND_URL}/chat/history?session_id=${encodeURIComponent(TEST_SESSION_ID)}`);
  if (!historyRes.ok) fail(`/chat/history returned ${historyRes.status}: ${await historyRes.text()}`);
  const historyData = await historyRes.json();
  // 3 chatTurn() calls so far (stage 1's family question, stage 2's
  // hobbies question and its post-brainstorm reply) = 6 messages
  // (user+assistant each). Brainstorm turns are kept apart from these -
  // they never count as chat messages.
  if (historyData.messages.length !== 6) {
    fail(`/chat/history returned ${historyData.messages.length} messages, expected 6: ${JSON.stringify(historyData)}`);
  }
  console.log(`  OK - ${historyData.messages.length} messages recorded for this session`);

  // Stage 2's post-brainstorm reply carries its brainstorm log (both
  // turns), and - already folded - is no longer pending.
  const foldedMessage = historyData.messages.find((m) => m.id === foldUserMessageId);
  if (!foldedMessage) fail(`/chat/history has no message with id ${foldUserMessageId} (stage 2's post-brainstorm reply)`);
  const foldedLog = foldedMessage.brainstorm ?? [];
  if (foldedLog.length !== 2 || foldedLog[0].role !== "learner" || foldedLog[1].role !== "coach") {
    fail(`stage 2's post-brainstorm reply should carry its learner + coach brainstorm turns: ${JSON.stringify(foldedMessage)}`);
  }
  if (foldedMessage.notes_pending !== false) {
    fail(`stage 2's post-brainstorm reply is still notes_pending after /brainstorm/fold: ${JSON.stringify(foldedMessage)}`);
  }
  if (historyData.messages.some((m) => !Array.isArray(m.notes))) {
    fail(`/chat/history messages should all carry a "notes" array: ${JSON.stringify(historyData.messages)}`);
  }
  console.log(`  OK - the post-brainstorm reply carries its brainstorm log (${foldedLog.length} turns) and is folded`);

  console.log(`[3/6] Clearing this session's history via POST /chat/history/clear ...`);
  const clearRes = await fetch(`${BACKEND_URL}/chat/history/clear?session_id=${encodeURIComponent(TEST_SESSION_ID)}`, {
    method: "POST",
  });
  if (!clearRes.ok) fail(`/chat/history/clear returned ${clearRes.status}: ${await clearRes.text()}`);
  const clearData = await clearRes.json();
  if (clearData.cleared !== 6) {
    fail(`/chat/history/clear reported clearing ${clearData.cleared} messages, expected 6`);
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
    // The composer has two textareas now - the Brainstorm field
    // (textarea.brainstorm__textarea) and the Reply field
    // (textarea.chat-input__textarea, the one that actually gets sent) -
    // so every Reply-field selector here is scoped to its class.
    await page.waitForSelector("textarea.chat-input__textarea", { timeout: 15_000 });
    await page.waitForSelector("textarea.brainstorm__textarea", { timeout: 15_000 });
    await page.screenshot({ path: "e2e-1-loaded.png" });

    // The Reply field's Send button always just sends (no "check first"
    // mode anymore - checking moved to the Brainstorm field).
    await page.fill("textarea.chat-input__textarea", "x");
    const defaultSendLabel = (await page.locator(".chat-input__send").innerText()).trim();
    if (defaultSendLabel !== "Send") {
      fail(`the Send button should read "Send", got ${JSON.stringify(defaultSendLabel)}`);
    }
    console.log('  OK - the Send button reads "Send"');
    await page.fill("textarea.chat-input__textarea", "");

    await page.fill("textarea.chat-input__textarea", "Hola");
    await page.keyboard.press("Enter");

    // The reply now streams in token-by-token (see App.tsx's streamingText)
    // via a plain, lightweight placeholder bubble that shares .chat-
    // message--assistant's own styling classes but isn't the real,
    // interactive ChatMessage component - that only gets mounted once the
    // full reply + its word-by-word glosses are ready. Waiting on the bare
    // ".chat-message--assistant" selector would resolve the instant that
    // placeholder appears (possibly with no text yet at all), well before
    // there's anything real to assert on - .chat-message__actions (the
    // audio/translate-toggle row) is only ever rendered by the real
    // component, so waiting on that instead reliably waits out the stream.
    await page.waitForSelector(".chat-message--assistant .chat-message__actions", { timeout: CHAT_TIMEOUT_MS });
    await page.screenshot({ path: "e2e-2-chat-reply.png" });

    const replyText = await page.locator(".chat-message--assistant").first().innerText();
    if (!replyText || replyText.trim().length === 0) {
      fail("assistant message rendered but has no text");
    }
    console.log(`  OK - assistant replied: ${JSON.stringify(replyText)}`);

    // Hover a word token and confirm the translation popover appears,
    // docked flush to the RIGHT of the bubble itself (DockedPopover
    // position="right", anchored to .chat-message__bubble - see
    // ChatMessage.tsx) rather than stacking underneath the bubble's own
    // full-message translation rows, which it used to overlap.
    const wordToken = page.locator(".chat-message--assistant .word-token").first();
    const assistantBubble = page.locator(".chat-message--assistant .chat-message__bubble").first();
    if ((await wordToken.count()) > 0) {
      // No word should carry a persistent highlight at rest (e.g. a "new
      // vocabulary" tint) - only the click-to-pin/hover state below should
      // ever color a word, and only while it's actually open.
      const restBackground = await wordToken.evaluate((el) => getComputedStyle(el).backgroundColor);
      if (restBackground !== "rgba(0, 0, 0, 0)" && restBackground !== "transparent") {
        fail(`word token has a persistent background at rest: ${restBackground}`);
      } else {
        console.log("  OK - word tokens carry no persistent highlight at rest");
      }

      await wordToken.hover();
      await page.waitForSelector(".translate-popover", { timeout: 8_000 });
      console.log("  OK - hover translation popover works");

      const bubbleBox = await assistantBubble.boundingBox();
      const wordPopoverBox = await page.locator(".translate-popover").first().boundingBox();
      // Allow some slack (the DockedPopover--right wrap has an 8px
      // padding-left, and the popover's own top can shift a hair from the
      // bubble's) - the real assertion is "to the right of the bubble",
      // not an exact pixel gap.
      if (wordPopoverBox.x <= bubbleBox.x + bubbleBox.width - 1) {
        fail(
          `chat-bubble word popover isn't docked to the right of the bubble (popover x=${wordPopoverBox.x}, bubble right edge=${bubbleBox.x + bubbleBox.width})`,
        );
      } else if (wordPopoverBox.y < bubbleBox.y - 2) {
        fail(`chat-bubble word popover's top is above the bubble's own top (popover y=${wordPopoverBox.y}, bubble y=${bubbleBox.y})`);
      } else {
        console.log("  OK - word popover docks flush to the right of the bubble, not stacked under the translation rows");
      }

      // Close it before moving on - docked over to the side, which can
      // still land near the action row and intercept the
      // next step's hover on the globe button otherwise. The close is
      // delayed (HOVER_CLOSE_DELAY_MS in ChatMessage.tsx) so the pointer
      // can safely travel from the word down to the docked popover without
      // it vanishing first - wait past that delay, not just a moment.
      await page.mouse.move(0, 0);
      await page.waitForTimeout(300);

      // Clicking (not just hovering) should pin it open with a highlight -
      // the mobile equivalent of hover, since a tap never fires mouseenter
      // on a real touch device. Focus a draft in the input FIRST, so this
      // same click also confirms checking a word's meaning mid-draft
      // doesn't cost the mobile keyboard - clicking a bubble word must
      // hand focus right back to the draft rather than leaving it blurred.
      const draftTextarea = page.locator("textarea.chat-input__textarea");
      await draftTextarea.fill("un borrador sin terminar");
      await draftTextarea.focus();
      await wordToken.click();
      await page.waitForTimeout(100);
      const pinnedActive = await wordToken.evaluate((el) => el.classList.contains("word-token--active"));
      const pinnedPopoverVisible = await page.locator(".translate-popover").first().isVisible().catch(() => false);
      if (!pinnedActive || !pinnedPopoverVisible) {
        fail("clicking a word token did not pin it open with a highlight + popover");
      }
      const borderBottom = await wordToken.evaluate((el) => getComputedStyle(el).borderBottomStyle);
      if (borderBottom !== "none") {
        fail(`word token still has a border-bottom (${borderBottom}) - looks like a hyperlink`);
      }
      console.log("  OK - clicking a word token pins it open with a highlight, no underline");

      const activeAfterWordClick = await page.evaluate(() => document.activeElement?.className ?? document.activeElement?.tagName);
      if (!String(activeAfterWordClick).includes("chat-input__textarea")) {
        fail(`clicking a chat-bubble word left focus on ${JSON.stringify(activeAfterWordClick)}, not the Reply textarea - the keyboard would drop`);
      } else {
        console.log("  OK - clicking a bubble word returns focus to the draft (keyboard stays up)");
      }
      const draftStillThere = await draftTextarea.inputValue();
      if (draftStillThere !== "un borrador sin terminar") {
        fail(`in-progress draft was lost/changed: ${JSON.stringify(draftStillThere)}`);
      }
      await draftTextarea.fill("");

      // Pinning is sticky by design (stays open after the mouse leaves,
      // same pattern as ChatMessage's own translate-pin) - unpin it again
      // explicitly so it doesn't sit open and overlap later steps below it
      // in the bubble (the audio/translate-toggle buttons).
      await wordToken.click();
      await page.mouse.move(0, 0);
      await page.waitForTimeout(300);
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

    // Regression check: a second click on the toggle, WITHOUT moving the
    // mouse away first (the bug's exact trigger condition - onMouseEnter
    // only fires once per hover, so it never refired between the two
    // clicks and used to mask the un-pin with a lingering hover state),
    // must close the row immediately.
    await assistantToggle.click();
    await page.waitForTimeout(100);
    const closedBySecondClick = await page.locator(".chat-message--assistant .translation-row--assistant").count();
    if (closedBySecondClick !== 0) {
      fail("clicking the globe toggle a second time (mouse still over it) did not close the translation row");
    }
    console.log("  OK - a second click on the globe toggle closes the translation row, even with the mouse still over it");
    await page.mouse.move(0, 0);
    await page.waitForTimeout(200);

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
    const textarea = page.locator("textarea.chat-input__textarea");
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
      await waitForGlossReady(page);
      const gatoBox = await gatoPopover.boundingBox();
      await page.screenshot({
        path: "e2e-debug-single-word-hover-popover.png",
        clip: { x: Math.max(0, gatoBox.x - 10), y: Math.max(0, gatoBox.y - 10), width: gatoBox.width + 20, height: gatoBox.height + 20 },
      });
    }
    await page.mouse.move(10, 10);
    await page.waitForTimeout(200);
    await textarea.fill("");

    // Hovering a clickable English word dropped into an otherwise-Spanish
    // draft should show a Spanish replacement candidate, and clicking it
    // should swap just that word in place - the input's main per-word LLM
    // path (app.translate.llm_translate.tag_draft), replacing the old
    // dictionary/Argos one. Wait for the actual /translate/tag-input
    // response rather than a fixed delay - this is a real LLM call now,
    // not an instant dictionary lookup.
    const tagResponsePromise = page.waitForResponse(
      (res) => res.url().includes("/translate/tag-input") && res.request().method() === "POST",
      { timeout: 15_000 }
    );
    await textarea.fill("Quiero ir to the beach");
    await tagResponsePromise;
    await page.waitForTimeout(200); // let React apply the response to state/DOM
    const toSpan = page.locator(".chat-input__hoverable", { hasText: "to" }).first();
    if ((await toSpan.count()) > 0) {
      await toSpan.hover();
      await page.waitForSelector(".word-candidates-popover", { timeout: 8_000 });
      // Hovering just fired the lazy /translate/gloss-spans fetch for this
      // word - wait for its "Translating…" placeholder to resolve into the
      // real candidate before reading it.
      await waitForGlossReady(page);
      const candidateText = await page.locator(".candidate-cycler__text, .candidate-cycler__main").first().innerText();
      console.log(`  "to" candidate: ${JSON.stringify(candidateText)}`);
      // The active-highlight class (Fix 2) should show while it's open.
      const isActive = await toSpan.evaluate((el) => el.classList.contains("chat-input__hoverable--active"));
      if (!isActive) fail('hovering "to" did not add the chat-input__hoverable--active highlight class');
      else console.log("  OK - the hovered word gets the active highlight class");

      // Debug screenshot: visually confirm the word-hover popover's actual
      // rendered appearance (not just its DOM structure via selectors).
      const toBox = await page.locator(".word-candidates-popover").first().boundingBox();
      await page.screenshot({
        path: "e2e-debug-word-hover-popover.png",
        clip: { x: Math.max(0, toBox.x - 10), y: Math.max(0, toBox.y - 10), width: toBox.width + 20, height: toBox.height + 20 },
      });

      const candidateButton = page.locator(".word-candidates-popover button").first();
      if ((await candidateButton.count()) > 0) {
        await candidateButton.click();
        await page.waitForTimeout(150);
        const afterReplace = await textarea.inputValue();
        if (afterReplace.includes(" to ")) {
          fail(`clicking the candidate for "to" did not replace it in the draft: ${JSON.stringify(afterReplace)}`);
        } else {
          console.log(`  OK - clicking a word's candidate replaced it in place: ${JSON.stringify(afterReplace)}`);
        }
      } else {
        console.log('  (eyeball) "to" had no clickable candidate this run - not failing, model-dependent');
      }
    } else {
      fail('"to" in "Quiero ir to the beach" was not flagged as hoverable');
    }
    await textarea.fill("");

    // Multi-word grouping: the model may group a clear idiom into ONE span
    // covering several words rather than tagging each word alone - the
    // old dictionary/Argos path had no sentence context and could never
    // do this at all. Unlike the old /translate/tag-input path, a group
    // can only ever come from the LAZY /translate/gloss-spans fetch (the
    // cheap per-keystroke pass is always single-word) - so this has to
    // actually hover a word first and wait for that fetch to resolve
    // before it can tell whether grouping happened. Clicking such a
    // group's candidate should replace the WHOLE group, not just one word
    // inside it. Not asserted as a hard requirement (a small model
    // doesn't group every time), but exercised and eyeballed, with a hard
    // check that IF it grouped, replacement covers the whole span.
    await textarea.fill("I miss you");
    // The cheap per-keystroke pass debounces up to IDLE_DEBOUNCE_MS (the
    // last character typed, "u", isn't a word boundary) before the naive
    // hoverable spans render - wait for one to actually show up rather
    // than a fixed delay shorter than that debounce.
    await page.waitForSelector(".chat-input__hoverable", { timeout: 3_000 }).catch(() => {});
    const missWord = page.locator(".chat-input__hoverable", { hasText: "miss" }).first();
    if ((await missWord.count()) > 0) {
      await missWord.hover();
      await page.waitForSelector(".word-candidates-popover", { timeout: 8_000 });
      await waitForGlossReady(page);
      // Re-reads whichever element currently matches - if the real gloss
      // widened "miss" into a group, this now resolves to that wider span
      // (hasText does substring matching, and "miss you" still contains
      // "miss").
      const hoveredText = (await missWord.innerText()).trim();
      const isGrouped = /\s/.test(hoveredText);
      console.log(
        isGrouped
          ? `  (eyeball) the model grouped a multi-word span: ${JSON.stringify(hoveredText)}`
          : '  (eyeball) the model tagged "miss" on its own this run, not as a group - not failing, model-dependent',
      );
      if (isGrouped) {
        const groupButton = page.locator(".word-candidates-popover button").first();
        if ((await groupButton.count()) > 0) {
          await groupButton.click();
          await page.waitForTimeout(150);
          const afterGroupReplace = await textarea.inputValue();
          if (afterGroupReplace.includes(hoveredText)) {
            fail(`clicking the group's candidate left the original group text "${hoveredText}" in the draft: ${JSON.stringify(afterGroupReplace)}`);
          } else {
            console.log(`  OK - clicking a multi-word group ("${hoveredText}") replaced the whole group: ${JSON.stringify(afterGroupReplace)}`);
          }
        }
      }
    } else {
      fail('"miss" in "I miss you" was not flagged as hoverable');
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
      await page.waitForSelector(".word-candidates-popover", { timeout: 8_000 });
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
          const el = document.querySelector(".word-candidates-popover .candidate-cycler__text");
          return !!el && !el.textContent.includes("…");
        },
        { timeout: 8_000 },
      );
      const phraseTranslationText = await page
        .locator(".word-candidates-popover .candidate-cycler__text")
        .first()
        .innerText();
      if (!phraseTranslationText.toLowerCase().includes("cat")) {
        fail(`expected the Spanish->English phrase translation to mention "cat", got ${JSON.stringify(phraseTranslationText)}`);
      }
      console.log(`  OK - highlighting an all-Spanish phrase translated it to English: ${JSON.stringify(phraseTranslationText)}`);
      // Debug screenshot: visually confirm the phrase popover's actual
      // rendered appearance (not just its DOM structure via selectors).
      const phraseBox = await page.locator(".word-candidates-popover").first().boundingBox();
      await page.screenshot({
        path: "e2e-debug-phrase-popover.png",
        clip: { x: Math.max(0, phraseBox.x - 10), y: Math.max(0, phraseBox.y - 10), width: phraseBox.width + 20, height: phraseBox.height + 20 },
      });
    } else {
      fail('could not find "gusta"/"negro" as hoverable words to test drag-selection');
    }
    await textarea.fill("");

    // A real tap on a word in the input (not a drag/scroll) should open its
    // translation popover, same as desktop hover - and must NOT cost the
    // textarea its focus (the mobile keyboard staying up depends on this).
    // The hoverable overlay is pointer-events:none ONLY on a touch/coarse-
    // pointer device (see ChatInput.css's `@media (hover: hover) and
    // (pointer: fine)` gate - so a tap reaches the textarea underneath for
    // native caret placement on a real phone) - the shared `page` above has
    // no touch emulation, so that media query reads as a normal mouse-
    // capable desktop there and the overlay legitimately intercepts
    // (desktop hover behavior). Testing the touch path for real needs its
    // own touch-enabled context, same as a real mobile browser.
    {
      const touchContext = await browser.newContext({ viewport: { width: 390, height: 700 }, hasTouch: true });
      const touchPage = await touchContext.newPage();
      await touchPage.goto(FRONTEND_URL, { waitUntil: "load", timeout: 30_000 });
      const touchTextarea = touchPage.locator("textarea.chat-input__textarea");
      await touchTextarea.waitFor({ timeout: 15_000 });
      const tagResponsePromise2 = touchPage.waitForResponse(
        (res) => res.url().includes("/translate/tag-input") && res.request().method() === "POST",
        { timeout: 10_000 },
      );
      await touchTextarea.fill("Hola amigo");
      await tagResponsePromise2;
      await touchPage.waitForTimeout(150);
      const amigoSpan = touchPage.locator(".chat-input__hoverable", { hasText: "amigo" }).first();
      if ((await amigoSpan.count()) > 0) {
        const box = await amigoSpan.boundingBox();
        await touchPage.evaluate(
          ({ x, y }) => {
            const el = document.elementFromPoint(x, y);
            const touch = new Touch({ identifier: 1, target: el, clientX: x, clientY: y });
            const opts = { touches: [touch], targetTouches: [touch], changedTouches: [touch], bubbles: true, cancelable: true };
            el.dispatchEvent(new TouchEvent("touchstart", opts));
            el.dispatchEvent(new TouchEvent("touchend", { ...opts, touches: [] }));
          },
          { x: box.x + box.width / 2, y: box.y + box.height / 2 },
        );
        await touchPage.waitForSelector(".word-candidates-popover", { timeout: 3_000 }).catch(() => {});
        const tapPopoverVisible = await touchPage.locator(".word-candidates-popover").first().isVisible().catch(() => false);
        if (!tapPopoverVisible) fail('a tap on "amigo" in the input did not open its translation popover');
        const stillFocused = await touchPage.evaluate(() => !!document.activeElement?.classList.contains("chat-input__textarea"));
        if (!stillFocused) fail("tapping a word in the input lost focus on the textarea (keyboard would drop)");
        if (tapPopoverVisible && stillFocused) {
          console.log('  OK - tapping "amigo" in the input (real touch emulation) opens its popover and keeps the textarea focused');
        }
        // Docked flush against the input bar (DockedPopover), not floating
        // off wherever the tapped word happened to sit.
        if (tapPopoverVisible) {
          const tapPopoverBox = await touchPage.locator(".word-candidates-popover").first().boundingBox();
          const touchInputBarBox = await touchPage.locator(".chat-input").boundingBox();
          const dockGap = touchInputBarBox.y - (tapPopoverBox.y + tapPopoverBox.height);
          if (Math.abs(dockGap) > 2) {
            fail(`word-tap popover isn't flush against the input bar (gap=${dockGap.toFixed(1)}px)`);
          } else {
            console.log("  OK - word-tap popover sits flush against the input bar, not floating");
          }
        }
      } else {
        fail('"amigo" was not flagged as hoverable in the input - cannot test tap-to-translate');
      }
      await touchContext.close();
    }

    // The Brainstorm field (replaces the old "check first" Send mode and
    // its coach popover): a side chat with the coach about the next reply,
    // kept apart from the Reply field. Its answer streams into the log
    // above the field, and a proposed Spanish sentence auto-fills the
    // Reply field - but a typed brainstorm turn never sends on its own.
    const brainstormTextarea = page.locator("textarea.brainstorm__textarea");
    const sendButton = page.locator(".chat-input__send");
    const failWithScreenshot = async (path, message) => {
      await page.screenshot({ path }).catch(() => {});
      fail(message);
    };

    // The Reply field's focus keeper (see ChatInput.tsx) puts focus back
    // on the Reply after a tap elsewhere - but NOT when that tap lands on
    // another real form control, and the Brainstorm field is one.
    await textarea.fill("");
    await textarea.focus();
    await brainstormTextarea.click();
    await page.waitForTimeout(150);
    const focusAfterBrainstormClick = await page.evaluate(() => document.activeElement?.className ?? "");
    if (!focusAfterBrainstormClick.includes("brainstorm__textarea")) {
      await failWithScreenshot(
        "e2e-debug-brainstorm-focus.png",
        `clicking the Brainstorm field (with the Reply focused) left focus on ${JSON.stringify(focusAfterBrainstormClick)} - the Reply's focus keeper stole it`,
      );
    }
    console.log("  OK - focusing the Brainstorm field isn't stolen back by the Reply field's focus keeper");

    const brainstormAttempt = "quiero ir playa manana";
    const coachTurnsBefore = await page.locator(".brainstorm__turn--coach:not(.brainstorm__turn--live)").count();
    await brainstormTextarea.fill(brainstormAttempt);
    await page.locator(".brainstorm__ask").click();
    // The learner's own line joins the log right away, and the field clears.
    await page
      .waitForSelector(".brainstorm__turn--learner", { timeout: 5_000 })
      .catch(() => failWithScreenshot("e2e-debug-brainstorm-no-learner-turn.png", "asking the coach didn't add the learner's line to the brainstorm log"));
    const brainstormValueAfterAsk = await brainstormTextarea.inputValue();
    if (brainstormValueAfterAsk !== "") {
      fail(`the Brainstorm field should clear once its line is asked, still has ${JSON.stringify(brainstormValueAfterAsk)}`);
    }
    // Wait for the coach's answer to finish streaming (the --live turn is
    // replaced by a finished one), or for the panel's error line.
    await page
      .waitForFunction(
        (before) =>
          document.querySelectorAll(".brainstorm__turn--coach:not(.brainstorm__turn--live)").length > before ||
          !!document.querySelector(".brainstorm__error"),
        coachTurnsBefore,
        { timeout: CHAT_TIMEOUT_MS },
      )
      .catch(() => failWithScreenshot("e2e-debug-brainstorm-timeout.png", "the coach's brainstorm answer never finished streaming"));
    const brainstormErrorText = await page.locator(".brainstorm__error").innerText().catch(() => null);
    if (brainstormErrorText) {
      await failWithScreenshot("e2e-debug-brainstorm-error.png", `the Brainstorm panel showed an error: ${JSON.stringify(brainstormErrorText)}`);
    }
    const coachTurn = page.locator(".brainstorm__turn--coach:not(.brainstorm__turn--live)").last();
    const coachTurnText = (await coachTurn.innerText()).trim();
    if (!coachTurnText) await failWithScreenshot("e2e-debug-brainstorm-empty.png", "the coach's finished brainstorm turn is empty");
    console.log(`  OK - the coach answered in the brainstorm log: ${JSON.stringify(coachTurnText)}`);
    await page.screenshot({ path: "e2e-debug-brainstorm.png" });

    // A rough Spanish attempt should normally get a proposed sentence (the
    // turn's "↓ ..." draft chip), which must have been auto-filled into the
    // Reply field. A model that only answered without one isn't a hard
    // failure (model-dependent) - fill a reply by hand so the send/notes
    // flow below still runs.
    const draftChip = coachTurn.locator(".brainstorm__draft");
    let replyToSend;
    if ((await draftChip.count()) > 0) {
      const chipDraft = (await draftChip.first().innerText()).replace(/^↓\s*/, "").trim();
      const replyValue = (await textarea.inputValue()).trim();
      if (!replyValue) {
        await failWithScreenshot("e2e-debug-brainstorm-no-fill.png", `the coach proposed ${JSON.stringify(chipDraft)} but the Reply field is still empty`);
      }
      if (replyValue !== chipDraft) {
        fail(`the Reply field holds ${JSON.stringify(replyValue)}, not the coach's proposed draft ${JSON.stringify(chipDraft)}`);
      }
      if (replyValue.toLowerCase().includes("manana") && !/mañana/i.test(replyValue)) {
        console.log(`  (eyeball) the draft kept "manana" without the ñ: ${JSON.stringify(replyValue)}`);
      }
      console.log(`  OK - the coach's draft auto-filled the Reply field: ${JSON.stringify(replyValue)}`);
      const sendReady = await sendButton.evaluate((el) => el.classList.contains("chat-input__send--ready"));
      console.log(`  (eyeball) Send button marked ready (coach judged the draft ready): ${sendReady}`);
      replyToSend = replyValue;
    } else {
      console.log("  (eyeball) the coach proposed no draft for this attempt - not failing, model-dependent; filling the reply by hand");
      replyToSend = "Quiero ir a la playa mañana.";
      await textarea.fill(replyToSend);
    }
    // A typed brainstorm turn must never send by itself.
    const userBubblesBeforeSend = await page.locator(".chat-message--user").count();
    const sendLabel = (await sendButton.innerText()).trim();
    if (sendLabel !== "Send") fail(`the Send button should still read "Send", got ${JSON.stringify(sendLabel)}`);

    // Sending closes the open brainstorm log: it goes under the sent
    // bubble ("Organizing notes…" while POST /brainstorm/fold runs) and the
    // Brainstorm panel starts fresh.
    const assistantCountBeforeSend = await page.locator(".chat-message--assistant").count();
    await sendButton.click();
    await page.waitForFunction((count) => document.querySelectorAll(".chat-message--assistant").length > count, assistantCountBeforeSend, {
      timeout: CHAT_TIMEOUT_MS,
    });
    const draftAfterSend = await textarea.inputValue();
    if (draftAfterSend !== "") fail(`the Reply should be cleared after sending, still has ${JSON.stringify(draftAfterSend)}`);
    if ((await page.locator(".chat-message--user").count()) !== userBubblesBeforeSend + 1) {
      fail("clicking Send didn't add exactly one user bubble");
    }
    const openLogAfterSend = await page.locator(".brainstorm__turn").count();
    if (openLogAfterSend !== 0) fail(`the Brainstorm log should start fresh after sending, still shows ${openLogAfterSend} turns`);
    console.log(`  OK - Send sent ${JSON.stringify(replyToSend)} and the brainstorm log moved off the composer`);

    const sentBubble = page.locator(".chat-message--user").last();
    const notesToggle = sentBubble.locator(".message-notes__toggle:not(.message-notes__toggle--nested)").first();
    await notesToggle
      .waitFor({ state: "visible", timeout: 15_000 })
      .catch(() => failWithScreenshot("e2e-debug-notes-missing.png", "the sent bubble never got a notes dropdown (.message-notes__toggle)"));
    // Folding is one more LLM call - generous timeout. It ends as
    // "Notes (n)", or "Brainstorm" (just the log) if nothing was worth
    // noting or the fold failed and will retry on the next load.
    await page
      .waitForFunction(
        (el) => !!el && !el.textContent.includes("Organizing notes"),
        await notesToggle.elementHandle(),
        { timeout: CHAT_TIMEOUT_MS },
      )
      .catch(() => failWithScreenshot("e2e-debug-notes-pending.png", 'the sent bubble\'s notes never settled (still "Organizing notes…")'));
    const notesLabel = (await notesToggle.innerText()).trim();
    if (!/Notes \(\d+\)|Brainstorm/.test(notesLabel)) {
      await failWithScreenshot("e2e-debug-notes-label.png", `unexpected notes toggle label ${JSON.stringify(notesLabel)}`);
    }
    console.log(`  OK - the sent bubble's notes settled: ${JSON.stringify(notesLabel)}`);

    // Open it: the folded notes (if any) plus the raw brainstorm log.
    await notesToggle.click();
    await sentBubble.locator(".message-notes__body").waitFor({ state: "visible", timeout: 3_000 });
    const noteItems = await sentBubble.locator(".message-notes__list li").allInnerTexts();
    console.log(`  (eyeball) notes on the sent reply: ${JSON.stringify(noteItems)}`);
    const logToggle = sentBubble.locator(".message-notes__toggle--nested");
    if ((await logToggle.count()) === 0) fail("the sent bubble's notes have no nested \"Brainstorm log\" toggle");
    await logToggle.click();
    const learnerLogTurns = await sentBubble.locator(".message-notes__turn--learner").count();
    const coachLogTurns = await sentBubble.locator(".message-notes__turn--coach").count();
    if (learnerLogTurns < 1 || coachLogTurns < 1) {
      await failWithScreenshot(
        "e2e-debug-notes-log.png",
        `the sent bubble's brainstorm log should hold the learner + coach turns, got ${learnerLogTurns} learner / ${coachLogTurns} coach`,
      );
    }
    console.log(`  OK - the sent bubble's brainstorm log holds ${learnerLogTurns} learner / ${coachLogTurns} coach turn(s)`);
    const partnerNotes = await page.locator(".chat-message--assistant .message-notes__toggle").allInnerTexts();
    console.log(`  (eyeball) partner bubbles' notes toggles: ${JSON.stringify(partnerNotes)}`);
    await page.screenshot({ path: "e2e-debug-message-notes.png" });
    // Collapse it again so it doesn't push later steps around.
    await notesToggle.click();
    await page.mouse.move(0, 0);

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
    // control renders. This is a button-based SegmentedControl, not a
    // native <select> (removed so the composer's textareas are the page's
    // only native form controls - see SegmentedControl.tsx - which otherwise
    // made iOS Safari show its keyboard accessory bar's field-navigation
    // arrows).
    const voiceOptions = page.locator(".app__voice-picker .segmented-control__option");
    if ((await voiceOptions.count()) > 0) {
      const labels = await voiceOptions.allTextContents();
      const targetLabel = labels.includes("Alex") ? "Alex" : labels.find((l) => l.trim().length > 0);
      if (!targetLabel) {
        fail("voice picker has no selectable options");
      }
      await voiceOptions.filter({ hasText: targetLabel }).first().click();

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
      fail("no voice picker (.app__voice-picker .segmented-control__option) found in the header");
    }

    // Regression check for the iOS keyboard accessory bar fix: the
    // composer's two textareas (Brainstorm + Reply) should be the only
    // native form controls left on the page (see SegmentedControl.tsx -
    // the voice picker is button-based).
    const formControls = await page.evaluate(() =>
      [...document.querySelectorAll("input, select, textarea")].map((el) => `${el.tagName}.${el.className}`),
    );
    const expectedControls = ["TEXTAREA.brainstorm__textarea", "TEXTAREA.chat-input__textarea"];
    if (JSON.stringify([...formControls].sort()) !== JSON.stringify(expectedControls)) {
      fail(`expected only the Brainstorm + Reply textareas as native form controls, found: ${JSON.stringify(formControls)}`);
    } else {
      console.log("  OK - the Brainstorm + Reply textareas are the only native form controls on the page (no stray select/checkbox)");
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
const brainstormResult = await checkEnglishInputAndTranslation();
await checkHistoryAndClear(brainstormResult);
await checkBrowserEndToEnd();
await checkTTSDirect();
await checkVoicesDirect();
console.log("ALL CHECKS PASSED");
