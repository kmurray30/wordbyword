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
//  2. /translate/coach directly (OpenAI): a Spanglish attempt must come
//     back as a coach verdict (not routed to the tutor), and a question
//     ("what does correr mean?") must be routed to the tutor ("ask") with
//     a streamed answer. Both printed for eyeballing.
//  3. WS /voice/listen from a browser context (Deepgram handshake): send
//     the "start" frame plus a second of silence, assert no "error" event.
//  4. The real thing: the partner clip from stage 1 (Kokoro saying
//     "Hola, ¿cómo estás? ¿Te gusta correr?") is decoded to a WAV and fed
//     to Chromium as its fake microphone against the deployed frontend.
//     Voice input is dictation: tap the mic, the words stream into the
//     chat box (a pause submits nothing), tap Done -> the same check as
//     typed text, read aloud by the coach (ElevenLabs) -> Send -> the
//     partner's reply (read aloud, Kokoro). Then a typed question goes to
//     the tutor and its answer appears (and is read aloud) in the thread
//     under the partner's message. Screenshots at each step.
import { chromium } from "playwright";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const BACKEND_URL = process.env.BACKEND_URL;
const FRONTEND_URL = process.env.FRONTEND_URL;
const TEST_SESSION_ID = `voice-e2e-${crypto.randomUUID()}`;
const SPANISH_SENTENCE = "Hola, ¿cómo estás? ¿Te gusta correr?";

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
    if (!res.ok) fail(`/tts/speak role=${role} returned ${res.status} after ${elapsed}ms: ${await res.text()}`);
    const buf = Buffer.from(await res.arrayBuffer());
    const type = res.headers.get("content-type");
    console.log(`role=${role}: ${res.status} ${type} ${buf.length} bytes in ${elapsed}ms`);
    if (!type?.includes("audio") || buf.length < 1000) fail(`role=${role}: not a real audio response`);
    writeFileSync(`voice-${role}.mp3`, buf);
    clips[role] = buf;
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
  console.log("\n=== Stage 2: /translate/coach (OpenAI) - attempt vs question routing ===");
  const attempt = await coachStream("hola, te gusta running?");
  const verdict = attempt.get("verdict")?.verdict;
  console.log(`attempt (${attempt.elapsed}ms): verdict=${verdict} meaning=${JSON.stringify(attempt.get("meaning_complete")?.meaning)}`);
  for (const e of attempt.events.filter((e) => e.name === "option_complete" || e.name === "suggestion_complete")) {
    console.log(`  ${e.name}: ${JSON.stringify(e.data)}`);
  }
  if (!["clean", "minor", "fix"].includes(verdict)) fail(`a Spanglish attempt should get a coach verdict, got ${verdict}`);
  if (!attempt.get("meaning_complete")?.meaning) fail("attempt had no meaning");

  const question = await coachStream("what does 'correr' mean?");
  const qVerdict = question.get("verdict")?.verdict;
  const answer = question.get("answer_complete")?.answer;
  console.log(`question (${question.elapsed}ms): verdict=${qVerdict}`);
  console.log(`  tutor: ${JSON.stringify(answer)}`);
  if (qVerdict !== "ask") fail(`a question should be routed to the tutor ("ask"), got ${qVerdict}`);
  if (!answer || /^(ABOUT|ANSWER)\b/i.test(answer)) fail("tutor answer missing or starts with an echoed label");
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
    const MIC = ".chat-input__mic";
    const DRAFT = ".chat-input__textarea";
    const SEND = ".chat-input__send";
    const draftValue = () => page.inputValue(DRAFT).catch(() => "");

    // 1. Dictation: the words stream into the box; nothing is submitted
    //    until Done.
    await page.waitForSelector(MIC, { timeout: 30_000 });
    await page.click(MIC);
    await page.waitForSelector(`${MIC}.chat-input__mic--done`, { timeout: 20_000 });
    await page.screenshot({ path: "voice-e2e-1-listening.png" });
    console.log("dictation started (mic became Done)");
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
      fail("dictation never reached the chat box");
    }
    console.log(`streamed into the box: ${JSON.stringify(streamed)} (${wsStats.frames} frames sent, ${wsStats.loudFrames} speech-level)`);
    // The clip is over and Deepgram has seen the pause - still nothing may
    // be submitted.
    await page.waitForTimeout(3000);
    if (coachCalls.length) fail("a pause in speech submitted the text - only Done should");
    await page.screenshot({ path: "voice-e2e-2-dictated.png" });

    // 2. Done -> the same check typed text gets, read aloud.
    await page.click(MIC);
    await page.waitForSelector(`${MIC}.chat-input__mic--done`, { state: "detached", timeout: 10_000 });
    const submitted = (await draftValue()).trim();
    console.log(`Done -> box: ${JSON.stringify(submitted)}`);
    if (!/gusta|correr|hola|cómo|como/i.test(submitted)) fail(`final text doesn't look like the spoken sentence: ${submitted}`);
    await page.waitForSelector(".coach-popover .coach-popover__row", { timeout: 120_000 });
    await page.waitForFunction(() => !document.querySelector(".chat-input__send")?.textContent.includes("Checking"), null, { timeout: 120_000 });
    const coachText = (await page.textContent(".coach-popover")).trim();
    console.log(`coach: ${JSON.stringify(coachText.slice(0, 300))}`);
    await page.screenshot({ path: "voice-e2e-3-checked.png" });
    await page.waitForTimeout(1500);
    const coachClips = tts.filter((t) => t.role === "coach");
    if (!coachClips.some((t) => t.status === 200)) fail("the coach's result was not read aloud");
    if (coachClips.some((t) => t.status !== 200)) fail(`coach TTS failures: ${JSON.stringify(coachClips.filter((t) => t.status !== 200))}`);

    // 3. Send (picking the natural option first if the coach wants a fix).
    if (!(await page.locator(".chat-input__send--ready").count())) {
      await page.click(".coach-popover__option-main >> nth=0");
      await page.waitForSelector(".chat-input__send--ready", { timeout: 10_000 });
    }
    const bubblesBefore = await page.locator(".chat-message__bubble").count();
    await page.click(SEND);
    await page.waitForFunction((n) => document.querySelectorAll(".chat-message--assistant .chat-message__bubble").length >= 1 && document.querySelectorAll(".chat-message__bubble").length >= n + 2, bubblesBefore, { timeout: 120_000 });
    const bubbles = await page.$$eval(".chat-message__bubble", (els) => els.map((e) => e.textContent.trim()).slice(-2));
    console.log(`chat: ${JSON.stringify(bubbles)}`);
    await page.waitForTimeout(2000);
    if (!tts.some((t) => t.role === "partner" && t.status === 200)) fail("the partner's reply was not read aloud");
    if (!(await page.locator(".chat-message--user .chat-message__thread-toggle").count())) fail("the sent message has no thread with its check");

    // 4. A question goes to the tutor, answered under the partner's message.
    const coachClipsBefore = tts.filter((t) => t.role === "coach").length;
    await page.fill(DRAFT, "what does 'correr' mean?");
    await page.press(DRAFT, "Enter");
    await page.waitForSelector(".chat-message--assistant .message-thread", { timeout: 120_000 });
    await page.waitForFunction(
      () => {
        const lines = [...document.querySelectorAll(".chat-message--assistant .message-thread .message-thread__line")];
        const last = lines.at(-1)?.textContent ?? "";
        return /tutor/i.test(last) && last.replace(/^tutor/i, "").trim().length > 10 && !document.querySelector(".message-thread__error");
      },
      null,
      { timeout: 120_000 },
    );
    await page.waitForTimeout(2500);
    const thread = (await page.textContent(".chat-message--assistant .message-thread")).trim();
    console.log(`tutor thread: ${JSON.stringify(thread.slice(0, 400))}`);
    await page.screenshot({ path: "voice-e2e-4-tutor.png" });
    if (await page.locator(".message-thread__error").count()) fail("tutor answer failed");
    if ((await page.locator(".chat-message__bubble").count()) !== bubblesBefore + 2) fail("the question was sent to the partner instead of the tutor");
    if (tts.filter((t) => t.role === "coach").length <= coachClipsBefore) fail("the tutor's answer was not read aloud");
    console.log(`TTS: ${tts.filter((t) => t.role === "coach").length} coach/tutor clips, ${tts.filter((t) => t.role === "partner").length} partner clips, statuses ${[...new Set(tts.map((t) => t.status))].join(",")}`);

    console.log(`console/page errors: ${errors.length ? JSON.stringify(errors) : "none"}`);
    if (errors.length) fail("page errors during the voice conversation");
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
const wavPath = await decodeToWav(helper, clips.partner);
await helper.close();
console.log(`fake-mic wav written: ${wavPath}`);
await checkVoiceConversation(wavPath);
await post(`/chat/history/clear?session_id=${encodeURIComponent(TEST_SESSION_ID)}`, {}).catch(() => {});
console.log("\nALL VOICE STAGES PASSED");
