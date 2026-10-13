// On-demand end-to-end check of VOICE MODE against the deployed Railway
// services, with the real providers behind them (Deepgram STT, ElevenLabs
// coach TTS, DeepInfra/Kokoro partner TTS, OpenAI coach LLM). Runs on
// GitHub Actions - see voice-e2e-check.yml for why not the dev sandbox.
//
// Stages, cheapest/most-isolating first so a provider problem fails on
// its own stage rather than masking the rest:
//  1. /tts/speak role=coach (ElevenLabs) and role=partner (Kokoro)
//     directly - both must return real audio. The clips are uploaded as
//     artifacts so a human can listen to the coach's mixed English/Spanish
//     pronunciation, which no script can judge.
//  2. /translate/coach directly (OpenAI): cold Spanglish must get a coach
//     reply with a [[Spanish | English]] suggestion, and a question ("what
//     does correr mean?") a plain answer. Both printed for eyeballing.
//  3. WS /voice/listen from a browser context (Deepgram handshake): send
//     the "start" frame plus a second of silence, assert no "error" event.
//  4. The real thing: the partner clip from stage 1 (Kokoro saying
//     "Hola, ¿cómo estás? ¿Te gusta correr?") is decoded to a WAV and fed
//     to Chromium as its fake microphone against the deployed frontend.
//     Dictate into the message box (words stream in, a pause submits
//     nothing), finish -> the coach checks it in "Your next message" and
//     reads its one suggestion aloud (ElevenLabs) -> Use it / Send -> the
//     partner's reply (read aloud, Kokoro). Then a question asked in their
//     message's own 💬 conversation is answered there, and working out the
//     next reply stays in the panel. Screenshots at each step.
import { chromium } from "playwright";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const BACKEND_URL = process.env.BACKEND_URL;
const FRONTEND_URL = process.env.FRONTEND_URL;
const TEST_SESSION_ID = `voice-e2e-${crypto.randomUUID()}`;
const SPANISH_SENTENCE = "Hola, ¿cómo estás? ¿Te gusta correr?";
// Problems with a third-party provider that shouldn't stop the rest of the
// check from running - reported (and failed on) at the very end instead.
const deferredFailures = [];

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

async function post(path, body, timeoutMs = 90_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${BACKEND_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    fail(`${path} failed/timed out: ${err}`);
  } finally {
    clearTimeout(timer);
  }
}

async function readSseEvents(response) {
  const text = await response.text();
  const events = [];
  for (const block of text.split("\n\n")) {
    if (!block.trim() || block.startsWith(":")) continue;
    const lines = block.split("\n");
    const name = lines.find((l) => l.startsWith("event:"))?.slice(6).trim();
    const data = lines.find((l) => l.startsWith("data:"))?.slice(5).trim();
    if (name && data) events.push({ name, data: JSON.parse(data) });
  }
  return events;
}

// ---- stage 1: TTS for both roles ------------------------------------------
async function checkTtsDirect() {
  console.log("\n=== Stage 1: /tts/speak coach (ElevenLabs) + partner (Kokoro) ===");
  // Wake a sleeping backend first so the TTS timing below is about the
  // provider, not Railway's cold start.
  await fetch(`${BACKEND_URL}/health`).catch(() => {});
  const clips = {};
  for (const [role, body] of [
    ["coach", { text: "Nice one. In Spanish, running is correr. Say: ¿Te gusta correr?", role: "coach" }],
    ["partner", { text: SPANISH_SENTENCE, language: "es", role: "partner" }],
  ]) {
    const start = Date.now();
    const res = await post("/tts/speak", body, 90_000);
    const elapsed = Date.now() - start;
    if (!res.ok && role === "partner") {
      // The partner voice (DeepInfra) is down - everything else can still
      // be checked, with the coach voice saying the sentence as the mic.
      const detail = `/tts/speak role=partner returned ${res.status} after ${elapsed}ms: ${await res.text()}`;
      console.log(`PARTNER VOICE DOWN - ${detail}`);
      deferredFailures.push(detail);
      const fallback = await post("/tts/speak", { text: SPANISH_SENTENCE, role: "coach" }, 90_000);
      if (!fallback.ok) fail(`fallback mic clip via the coach voice also failed: ${fallback.status}`);
      clips.mic = Buffer.from(await fallback.arrayBuffer());
      writeFileSync("voice-mic-fallback.mp3", clips.mic);
      console.log(`using the coach voice saying ${JSON.stringify(SPANISH_SENTENCE)} as the mic (${clips.mic.length} bytes)`);
      continue;
    }
    if (!res.ok) fail(`/tts/speak role=${role} returned ${res.status} after ${elapsed}ms: ${await res.text()}`);
    const buf = Buffer.from(await res.arrayBuffer());
    const type = res.headers.get("content-type");
    console.log(`role=${role}: ${res.status} ${type} ${buf.length} bytes in ${elapsed}ms`);
    if (!type?.includes("audio") || buf.length < 1000) fail(`role=${role}: not a real audio response`);
    writeFileSync(`voice-${role}.mp3`, buf);
    clips[role] = buf;
    if (role === "partner") clips.mic = buf;
  }
  return clips;
}

// ---- stage 2: the spoken coach, directly ----------------------------------
async function coachStream(text) {
  const start = Date.now();
  const res = await post("/translate/coach", { text, session_id: TEST_SESSION_ID });
  if (!res.ok) fail(`/translate/coach returned ${res.status}: ${await res.text()}`);
  const events = await readSseEvents(res);
  const get = (name) => events.find((e) => e.name === name)?.data;
  const error = get("error");
  if (error) fail(`/translate/coach error event: ${error.message}`);
  return { events, get, elapsed: Date.now() - start };
}

async function checkCoachRoutingDirect() {
  console.log("\n=== Stage 2: /translate/coach (OpenAI) - one conversational coach ===");
  const attempt = await coachStream("hola, te gusta running?");
  const reply = attempt.get("reply_complete")?.reply;
  console.log(`cold Spanglish (${attempt.elapsed}ms): about=${JSON.stringify(attempt.get("about"))} ready=${JSON.stringify(attempt.get("ready"))}`);
  console.log(`  coach: ${JSON.stringify(reply)}`);
  if (!reply) fail("no coach reply");
  if (!/\[\[[^\]|]+\|[^\]]+\]\]/.test(reply)) fail("the coach should suggest a fix as a [[Spanish | English]] block for Spanglish");
  if (attempt.get("ready")?.ready) fail("Spanglish shouldn't be ready to send");

  const question = await coachStream("what does 'correr' mean?");
  const answer = question.get("reply_complete")?.reply;
  console.log(`question (${question.elapsed}ms): about=${JSON.stringify(question.get("about"))}`);
  console.log(`  coach: ${JSON.stringify(answer)}`);
  if (!answer || /^(ABOUT|READY|REPLY|ANSWER)\b/i.test(answer)) fail("coach answer missing or starts with an echoed label");
}

async function checkListenHandshake(browser) {
  console.log("\n=== Stage 3: WS /voice/listen handshake (Deepgram) ===");
  const page = await browser.newPage();
  await page.goto(FRONTEND_URL);
  const result = await page.evaluate(
    ({ url }) =>
      new Promise((resolve) => {
        const events = [];
        const ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";
        const done = (why) => {
          try {
            ws.close();
          } catch {}
          resolve({ why, events });
        };
        ws.onerror = () => done("ws error");
        ws.onclose = (e) => done(`closed ${e.code}`);
        ws.onmessage = (m) => {
          events.push(String(m.data).slice(0, 200));
          if (String(m.data).includes('"error"')) done("error event");
        };
        ws.onopen = () => {
          ws.send(JSON.stringify({ type: "start", session_id: "voice-e2e-handshake" }));
          const silence = new ArrayBuffer(3200);
          let n = 0;
          const tick = setInterval(() => {
            ws.send(silence);
            if (++n >= 10) {
              clearInterval(tick);
              ws.send(JSON.stringify({ type: "stop" }));
              setTimeout(() => done("stopped"), 3000);
            }
          }, 100);
        };
        setTimeout(() => done("timeout"), 20000);
      }),
    { url: BACKEND_URL.replace(/^http/, "ws") + "/voice/listen" },
  );
  console.log(`handshake: ${result.why}; events: ${JSON.stringify(result.events)}`);
  if (result.why === "error event" || result.why === "ws error") fail("Deepgram handshake failed");
  await page.close();
}

// ---- stage 4: the real conversation, fake mic fed with the partner clip ---
async function decodeToWav(browser, mp3) {
  // Chromium decodes the mp3 (no ffmpeg needed); 1s of leading silence so
  // the mic is definitely open before the speech starts, 3s trailing so
  // Deepgram's utterance_end fires.
  const page = await browser.newPage();
  const base64 = mp3.toString("base64");
  const wavBase64 = await page.evaluate(async (b64) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const probe = new OfflineAudioContext(1, 1, 48000);
    const decoded = await probe.decodeAudioData(bytes.buffer.slice(0));
    const rate = 48000;
    const lead = rate, tail = rate * 3;
    const ctx = new OfflineAudioContext(1, lead + Math.ceil(decoded.duration * rate) + tail, rate);
    const src = ctx.createBufferSource();
    src.buffer = decoded;
    src.connect(ctx.destination);
    src.start(1);
    const rendered = await ctx.startRendering();
    const samples = rendered.getChannelData(0);
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
    v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data");
    v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    let out = "";
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < u8.length; i += 0x8000) out += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(out);
  }, base64);
  await page.close();
  const path = resolve("voice-mic-input.wav");
  writeFileSync(path, Buffer.from(wavBase64, "base64"));
  return path;
}

function wavStats(path) {
  const buf = readFileSync(path);
  const rate = buf.readUInt32LE(24);
  const n = (buf.length - 44) / 2;
  let sum = 0, peak = 0;
  for (let i = 0; i < n; i++) {
    const s = buf.readInt16LE(44 + i * 2) / 32768;
    sum += s * s;
    if (Math.abs(s) > peak) peak = Math.abs(s);
  }
  return { seconds: (n / rate).toFixed(2), rms: Math.sqrt(sum / n).toFixed(4), peak: peak.toFixed(3) };
}

async function checkVoiceConversation(wavPath) {
  console.log("\n=== Stage 4: full voice conversation on the deployed site (mic = Kokoro clip) ===");
  console.log(`mic clip: ${JSON.stringify(wavStats(wavPath))}`);
  const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"] });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));

  // The "microphone" is generated in-page: getUserMedia is replaced with a
  // MediaStream from a MediaStreamAudioDestinationNode playing the decoded
  // clip. Deterministic and independent of Chromium's fake-capture-file
  // flag, whose format requirements are strict and undocumented (a first
  // attempt with it produced silence at Deepgram).
  await page.addInitScript(({ wavBase64 }) => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      if (!constraints || !constraints.audio) return original(constraints);
      const ctx = new AudioContext({ sampleRate: 48000 });
      await ctx.resume();
      const bytes = Uint8Array.from(atob(wavBase64), (c) => c.charCodeAt(0));
      const buffer = await ctx.decodeAudioData(bytes.buffer);
      const dest = ctx.createMediaStreamDestination();
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(dest);
      src.start(ctx.currentTime + 0.5);
      window.__fakeMic = { duration: buffer.duration, ctx };
      return dest.stream;
    };
  }, { wavBase64: readFileSync(wavPath).toString("base64") });

  // Observe (not mock) the /voice/listen socket: how much audio the page
  // actually sends, how loud it is, and every event Deepgram sends back.
  const wsStats = { frames: 0, bytes: 0, loudFrames: 0, events: [] };
  await page.routeWebSocket("**/voice/listen", (ws) => {
    const server = ws.connectToServer();
    ws.onMessage((message) => {
      if (typeof message !== "string") {
        wsStats.frames += 1;
        wsStats.bytes += message.length;
        const view = new Int16Array(message.buffer, message.byteOffset, Math.floor(message.length / 2));
        let peak = 0;
        for (let i = 0; i < view.length; i += 8) peak = Math.max(peak, Math.abs(view[i]));
        if (peak > 500) wsStats.loudFrames += 1;
      }
      server.send(message);
    });
    server.onMessage((message) => {
      if (typeof message === "string") {
        wsStats.events.push(message.slice(0, 160));
        if (!message.includes('"interim"')) console.log(`  deepgram -> ${message.slice(0, 200)}`);
      }
      ws.send(message);
    });
  });
  const tts = [];
  const coachCalls = [];
  page.on("response", async (res) => {
    const url = res.url();
    if (url.includes("/tts/speak")) {
      let role = "?", text = "?";
      try { const body = res.request().postDataJSON(); role = body.role; text = body.text; } catch {}
      tts.push({ role, status: res.status(), text });
      let detail = "";
      if (res.status() !== 200) detail = ` body=${JSON.stringify(await res.text().catch(() => "?")).slice(0, 300)}`;
      console.log(`  tts role=${role} status=${res.status()} text=${JSON.stringify(text)}${detail}`);
    }
    if (url.includes("/translate/coach") || url.includes("/chat/turn/stream")) {
      let text = "";
      try { text = res.request().postDataJSON().text ?? res.request().postDataJSON().message ?? ""; } catch {}
      coachCalls.push(url.includes("/translate/coach") ? text : null);
      console.log(`  ${res.status()} ${url.replace(BACKEND_URL, "")} ${JSON.stringify(text)}`);
    }
  });

  try {
    await page.goto(FRONTEND_URL);
    await page.waitForSelector(".llm-gate", { state: "hidden", timeout: 60_000 }).catch(() => {});
    // Two inputs: the message box (what you'll send) and "Your next
    // message" (where you work it out with the coach).
    const TALK = ".chat-input .talk-button";
    const DRAFT = ".chat-input__textarea";
    const SEND = ".chat-input__send";
    const BRAIN = ".next-message__textarea";
    const draftValue = () => page.inputValue(DRAFT).catch(() => "");

    // 1. Dictate the message: the words stream into the message box; a
    //    pause submits nothing; tapping the talking bubble finishes and
    //    asks the coach to check it (the check lands in the panel).
    await page.waitForSelector(TALK, { timeout: 30_000 });
    await page.click(TALK);
    await page.waitForSelector(".chat-input .talk-button--talking", { timeout: 20_000 });
    await page.screenshot({ path: "voice-e2e-1-listening.png" });
    console.log("dictation started (talking bubble shown)");
    let streamed = "";
    const heardBy = Date.now() + 60_000;
    while (Date.now() < heardBy) {
      streamed = (await draftValue()).trim();
      if (/correr/i.test(streamed)) break;
      await page.waitForTimeout(300);
    }
    if (!streamed) {
      await page.screenshot({ path: "voice-e2e-timeout.png" });
      const mic = await page.evaluate(() => window.__fakeMic ? { duration: window.__fakeMic.duration, ctxState: window.__fakeMic.ctx.state } : null);
      console.log(`nothing streamed into the box within 60s. fakeMic=${JSON.stringify(mic)}`);
      console.log(`audio sent: ${wsStats.frames} frames / ${wsStats.bytes} bytes, ${wsStats.loudFrames} with speech-level peaks; deepgram events: ${JSON.stringify(wsStats.events.slice(0, 10))}`);
      console.log(`console/page errors: ${JSON.stringify(errors)}`);
      fail("dictation never reached the message box");
    }
    console.log(`streamed into the box: ${JSON.stringify(streamed)} (${wsStats.frames} frames sent, ${wsStats.loudFrames} speech-level)`);
    await page.waitForTimeout(3000);
    if (coachCalls.length) fail("a pause in speech submitted the text - only finishing should");
    await page.screenshot({ path: "voice-e2e-2-dictated.png" });

    await page.click(TALK);
    await page.waitForSelector(".chat-input .talk-button--talking", { state: "detached", timeout: 10_000 });
    const submitted = (await draftValue()).trim();
    console.log(`finished -> message box keeps: ${JSON.stringify(submitted)}`);
    if (!/gusta|correr|hola|cómo|como/i.test(submitted)) fail(`final text doesn't look like the spoken sentence: ${submitted}`);
    const lastCoach = ".next-message .talk-line--coach:last-child";
    await page.waitForSelector(`${lastCoach}[data-status="done"], ${lastCoach}[data-status="error"]`, { timeout: 120_000 });
    const card = (await page.locator(lastCoach).textContent()).trim();
    console.log(`coach (in Your next message): ${JSON.stringify(card.slice(0, 300))}`);
    if ((await page.locator(`${lastCoach}[data-status="error"]`).count())) fail("coach check failed");
    await page.screenshot({ path: "voice-e2e-3-checked.png" });
    await page.waitForTimeout(1500);
    const coachClips = tts.filter((t) => t.role === "coach");
    if (!coachClips.some((t) => t.status === 200)) fail("the coach's result was not read aloud");
    if (coachClips.some((t) => t.status !== 200)) fail(`coach TTS failures: ${JSON.stringify(coachClips.filter((t) => t.status !== 200))}`);

    // 2. Send (taking the suggestion first if the coach made one).
    if (!(await page.locator(".chat-input__send--ready").count())) {
      const use = page.locator(`${lastCoach} .coach-text__use:not([disabled])`).first();
      if (await use.count()) await use.click();
      else fail("coach neither approved the message nor suggested one to use");
    }
    await page.waitForSelector(".chat-input__send--ready", { timeout: 10_000 });
    const bubblesBefore = await page.locator(".chat-message__bubble").count();
    await page.click(SEND);
    // Wait for the partner's reply to be a real, saved message (it has an
    // actions row; the bubble shown while it streams doesn't).
    await page.waitForFunction((n) => document.querySelectorAll(".chat-message--assistant .chat-message__actions").length >= 1 && document.querySelectorAll(".chat-message__bubble").length >= n + 2, bubblesBefore, { timeout: 120_000 });
    const bubbles = await page.$$eval(".chat-message__bubble", (els) => els.map((e) => e.textContent.trim()).slice(-2));
    console.log(`chat: ${JSON.stringify(bubbles)}`);
    if (await page.locator(".next-message__log").count()) fail("Your next message should be empty after sending");
    if (!(await page.locator(".chat-message--user .chat-message__thread-toggle").count())) fail("the sent message has no thread with its check");
    // The partner voice (DeepInfra) can take ~10s per clip - wait for it.
    const partnerBy = Date.now() + 90_000;
    while (Date.now() < partnerBy && !tts.some((t) => t.role === "partner")) await page.waitForTimeout(500);
    if (!tts.some((t) => t.role === "partner" && t.status === 200)) {
      if (deferredFailures.length) console.log("partner reply not read aloud - expected while the partner voice is down");
      else fail("the partner's reply was not read aloud");
    }

    // 3. A question asked in THEIR message's own 💬 conversation (an
    //    overlay sheet over the chat) is answered there.
    const coachClipsBefore = tts.filter((t) => t.role === "coach").length;
    await page.locator(".chat-message--assistant .chat-message__thread-toggle").last().click();
    const THREAD_BOX = ".thread-overlay .message-thread__textarea";
    await page.waitForSelector(THREAD_BOX);
    if (!/their message/i.test(await page.textContent(".thread-overlay__label"))) fail("the 💬 overlay isn't about their message");
    await page.fill(THREAD_BOX, "what does 'correr' mean?");
    await page.press(THREAD_BOX, "Enter");
    await page.waitForSelector(".thread-overlay .message-thread .talk-line--coach", { timeout: 120_000 });
    await page.waitForFunction(
      () => {
        const t = document.querySelector(".thread-overlay .message-thread");
        if (!t || t.querySelector('[data-status="error"]')) return !!t;
        return !t.querySelector('[data-status="streaming"]') && !!t.querySelector(".talk-line--coach");
      },
      null,
      { timeout: 120_000 },
    );
    await page.waitForTimeout(2500);
    const thread = (await page.textContent(".thread-overlay .message-thread")).trim();
    console.log(`their-message thread: ${JSON.stringify(thread.slice(0, 400))}`);
    await page.screenshot({ path: "voice-e2e-4-coach.png" });
    if (await page.locator('.message-thread [data-status="error"]').count()) fail("coach answer failed");
    if (/tutor/i.test(thread)) fail("'Tutor' is still shown - it's the coach now");
    if ((await page.locator(".chat-message__bubble").count()) !== bubblesBefore + 2) fail("the question was sent to the partner instead of the coach");
    if (await page.inputValue(THREAD_BOX)) fail("the message's coach box should clear after asking");
    if (tts.filter((t) => t.role === "coach").length <= coachClipsBefore) fail("the coach's answer was not read aloud");
    await page.click(".thread-overlay__close");

    // 4. Working out the NEXT reply happens in the panel.
    // The coach box lives inside the "Discuss your next message" dropdown.
    if (!(await page.locator(BRAIN).isVisible().catch(() => false))) await page.click(".next-message__header");
    await page.fill(BRAIN, "I want to say I like swimming too");
    await page.press(BRAIN, "Enter");
    await page.waitForSelector(`${lastCoach}[data-status="done"], ${lastCoach}[data-status="error"]`, { timeout: 120_000 });
    const panel = (await page.textContent(".next-message__log")).trim();
    console.log(`Your next message: ${JSON.stringify(panel.slice(0, 300))}`);
    if ((await page.locator(".chat-message__bubble").count()) !== bubblesBefore + 2) fail("working out a reply sent something to the partner");
    console.log(`TTS: ${tts.filter((t) => t.role === "coach").length} coach clips, ${tts.filter((t) => t.role === "partner").length} partner clips, statuses ${[...new Set(tts.map((t) => t.status))].join(",")}`);

    // While the partner voice is down its failed /tts/speak calls show up as
    // console resource errors - expected, not a page bug.
    const pageErrors = deferredFailures.length ? errors.filter((e) => !/status of 50[23]/.test(e)) : errors;
    console.log(`console/page errors: ${pageErrors.length ? JSON.stringify(pageErrors) : "none"}`);
    if (pageErrors.length) fail("page errors during the voice conversation");
  } finally {
    // Clean up whatever this browser session chatted (its own session id,
    // kept in localStorage by lib/session.ts).
    try {
      const sid = await page.evaluate(() => Object.entries(localStorage).find(([k, v]) => /session/i.test(k) && /^[0-9a-f-]{36}$/i.test(v))?.[1] ?? null);
      if (sid) await post(`/chat/history/clear?session_id=${encodeURIComponent(sid)}`, {});
    } catch {}
    await browser.close();
  }
}

if (!BACKEND_URL || !FRONTEND_URL) fail("BACKEND_URL and FRONTEND_URL must be set");
const clips = await checkTtsDirect();
await checkCoachRoutingDirect();
const helper = await chromium.launch();
await checkListenHandshake(helper);
const wavPath = await decodeToWav(helper, clips.mic);
await helper.close();
console.log(`fake-mic wav written: ${wavPath}`);
await checkVoiceConversation(wavPath);
await post(`/chat/history/clear?session_id=${encodeURIComponent(TEST_SESSION_ID)}`, {}).catch(() => {});
if (deferredFailures.length) {
  console.log("\nEverything else passed, but a provider was down:");
  for (const f of deferredFailures) console.log(`  - ${f}`);
  fail("partner voice (DeepInfra) unavailable");
}
console.log("\nALL VOICE STAGES PASSED");
