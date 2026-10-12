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
//  2. /voice/coach directly (OpenAI) with a Spanglish transcript and one
//     deliberately low-confidence word - prints the intent / verdict /
//     draft / spoken reply for eyeballing, asserts all four arrived.
//  3. WS /voice/listen from a browser context (Deepgram handshake): send
//     the "start" frame plus a second of silence, assert no "error" event.
//  4. The real thing: the partner clip from stage 1 (Kokoro saying
//     "Hola, ¿cómo estás? ¿Te gusta correr?") is decoded to a WAV and fed
//     to Chromium as its fake microphone against the deployed frontend.
//     Start voice chat -> Deepgram transcribes the clip -> the coach
//     answers (ElevenLabs audio) -> if it judged the sentence ready, it's
//     sent and the partner replies (Kokoro audio). Transcript, coach text
//     and every TTS request/response are printed; screenshots at each step.
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
async function checkVoiceCoachDirect() {
  console.log("\n=== Stage 2: /voice/coach (OpenAI) on a Spanglish transcript ===");
  const start = Date.now();
  const res = await post("/voice/coach", {
    transcript: "hola, te gusta running?",
    words: [
      { word: "hola,", confidence: 0.98 },
      { word: "te", confidence: 0.95 },
      { word: "gusta", confidence: 0.58 },
      { word: "running?", confidence: 0.97 },
    ],
    coach_dialog: [],
    session_id: TEST_SESSION_ID,
  });
  if (!res.ok) fail(`/voice/coach returned ${res.status}: ${await res.text()}`);
  const events = await readSseEvents(res);
  const elapsed = Date.now() - start;
  const get = (name) => events.find((e) => e.name === name)?.data;
  const error = get("error");
  if (error) fail(`/voice/coach error event: ${error.message}`);
  const intent = get("intent")?.intent;
  const verdict = get("verdict")?.verdict;
  const draft = get("draft")?.draft;
  const say = get("say_complete")?.say;
  console.log(`(${elapsed}ms) intent=${intent} verdict=${verdict}`);
  console.log(`draft: ${JSON.stringify(draft)}`);
  console.log(`say:   ${JSON.stringify(say)}`);
  if (!intent || !verdict || draft === undefined || !say) fail("missing one of intent/verdict/draft/say");
  if (!["attempt", "question", "clarify", "confirm", "pick", "cancel", "translate_last"].includes(intent)) fail(`bad intent ${intent}`);
  if (!["ready", "revise", "none"].includes(verdict)) fail(`bad verdict ${verdict}`);
}

// ---- stage 3: Deepgram handshake over WS /voice/listen --------------------
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
    if (url.includes("/voice/coach") || url.includes("/chat/turn/stream")) console.log(`  ${res.status()} ${url.replace(BACKEND_URL, "")}`);
  });

  try {
    await page.goto(FRONTEND_URL);
    await page.waitForSelector(".llm-gate", { state: "hidden", timeout: 60_000 }).catch(() => {});
    // Voice mode lives inside the chat input: the mic button starts it, the
    // spoken words land in the draft textarea, the coach's spoken reply is
    // the docked coach popover, and sent/partner messages are the normal
    // chat bubbles.
    const MIC = ".chat-input__mic";
    const DRAFT = ".chat-input__textarea";
    const VOICE_STATE = ".coach-popover--voice .coach-popover__voice-state";
    const VOICE_SAY = ".coach-popover--voice .coach-popover__voice-say";
    const VOICE_ERR = ".coach-popover--voice .coach-popover__status--error";
    await page.waitForSelector(MIC, { timeout: 30_000 });
    await page.click(MIC);
    await page.waitForSelector(`${MIC}.chat-input__mic--listening`, { timeout: 20_000 });
    await page.screenshot({ path: "voice-e2e-1-listening.png" });
    console.log("voice chat started: listening (mic button active)");

    const draftValue = () => page.inputValue(DRAFT).catch(() => "");
    const voiceState = () => page.locator(VOICE_STATE).textContent({ timeout: 300 }).catch(() => null);
    const voiceError = () => page.locator(VOICE_ERR).textContent({ timeout: 300 }).catch(() => null);
    const micActive = () => page.locator(`${MIC}.chat-input__mic--active`).count().then((n) => n > 0);

    // Everything below has to be captured while the turn is live: a ready
    // verdict sends the draft (clearing the textarea) and the coach dialog
    // resets after a send, so polling is the only way to see what was
    // heard and said.
    let transcript = "";
    const coachTurns = [];
    let sendButtonWentGreen = false;
    const observe = async () => {
      const v = await draftValue();
      if (v.trim().length > transcript.length) transcript = v.trim();
      const says = await page.$$eval(VOICE_SAY, (els) => els.map((e) => e.textContent.replace(/^Coach/, "").trim())).catch(() => []);
      for (const t of says) {
        if (!t) continue;
        const i = coachTurns.findIndex((prev) => t.startsWith(prev) || prev.startsWith(t));
        if (i === -1) coachTurns.push(t);
        else if (t.length > coachTurns[i].length) coachTurns[i] = t;
      }
      if (!sendButtonWentGreen && (await page.locator(".chat-input__send--ready").count()) > 0) sendButtonWentGreen = true;
    };

    const heardBy = Date.now() + 60_000;
    while (Date.now() < heardBy && !transcript) {
      await observe();
      if (!transcript) await page.waitForTimeout(300);
    }
    if (!transcript) {
      await page.screenshot({ path: "voice-e2e-timeout.png" });
      const state = await voiceState();
      const interim = await page.locator(".chat-input__mic-pill").textContent({ timeout: 300 }).catch(() => null);
      const panelErr = await voiceError();
      const mic = await page.evaluate(() => window.__fakeMic ? { duration: window.__fakeMic.duration, ctxState: window.__fakeMic.ctx.state } : null);
      console.log(`no transcript within 60s. state=${state} interim=${JSON.stringify(interim)} error=${JSON.stringify(panelErr)} fakeMic=${JSON.stringify(mic)}`);
      console.log(`audio sent: ${wsStats.frames} frames / ${wsStats.bytes} bytes, ${wsStats.loudFrames} with speech-level peaks; deepgram events: ${JSON.stringify(wsStats.events.slice(0, 10))}`);
      console.log(`console/page errors: ${JSON.stringify(errors)}`);
      fail("the spoken clip never showed up in the draft box");
    }
    console.log(`audio sent so far: ${wsStats.frames} frames (${wsStats.loudFrames} speech-level)`);
    await page.screenshot({ path: "voice-e2e-2-heard.png" });

    // Let the turn play out: coach reply (+ possible send and partner
    // reply), then back to plain listening - the voice popover hides again
    // once there's nothing to show, or stays with "Listening…".
    const settled = Date.now() + 150_000;
    let sawCoach = false;
    let lastState = null;
    while (Date.now() < settled) {
      await observe();
      if (coachTurns.length) sawCoach = true;
      const s = await voiceState();
      if (s) lastState = s;
      if (sawCoach && (s === null || s === "Listening…" || s === "Problem")) break;
      await page.waitForTimeout(500);
    }
    console.log(`Deepgram heard: ${JSON.stringify(transcript)}`);
    if (!/gusta|correr|hola|cómo|como/i.test(transcript)) fail(`transcript doesn't look like the spoken sentence: ${transcript}`);
    console.log(`coach said: ${JSON.stringify(coachTurns)}`);
    const err = await voiceError();
    const stillActive = await micActive();
    console.log(`after the turn: mic active=${stillActive}, voice state=${lastState}, draft=${JSON.stringify(await draftValue())}${err ? ` (error: ${err})` : ""}`);
    const bubbles = await page.$$eval(".chat-message__bubble", (els) => els.map((e) => e.textContent.trim()).slice(-2));
    console.log(`last chat bubbles: ${JSON.stringify(bubbles)}`);
    await page.screenshot({ path: "voice-e2e-3-after-turn.png" });

    if (!coachTurns.length || !coachTurns[0]) fail("coach produced no spoken text");
    if (err) fail(`voice session error: ${err}`);
    if (!stillActive) fail("voice chat ended by itself during the turn");
    if (/^(INTENT|VERDICT|DRAFT|SAY)\b/i.test(coachTurns[0])) fail(`coach text starts with an echoed field label: ${coachTurns[0]}`);
    // The clip is correct Spanish, so the coach normally judges it ready and
    // says "Sending:" - in which case the sentence MUST actually have been
    // sent and answered (draft cleared, Send went green, partner spoken).
    if (/sending/i.test(coachTurns[0])) {
      if (!bubbles.some((b) => /gusta correr/i.test(b))) fail(`coach said "Sending" but no user bubble with the sentence appeared: ${JSON.stringify(bubbles)}`);
      if (bubbles.length < 2) fail(`coach said "Sending" but no partner reply bubble appeared: ${JSON.stringify(bubbles)}`);
      if (!tts.some((t) => t.role === "partner" && t.status === 200)) fail("coach said \"Sending\" but the partner reply was never spoken");
      if ((await draftValue()).trim()) fail("sent, but the draft box still holds the transcript");
      console.log(`auto-send after a ready verdict: OK (send button went green: ${sendButtonWentGreen})`);
    } else {
      console.log("coach asked for a revision rather than sending - acceptable, nothing to send-check");
    }
    const coachOk = tts.filter((t) => t.role === "coach" && t.status === 200).length;
    const coachBad = tts.filter((t) => t.role === "coach" && t.status !== 200);
    console.log(`coach TTS clips: ${coachOk} ok, ${coachBad.length} failed; partner clips: ${tts.filter((t) => t.role === "partner").map((t) => t.status).join(",") || "none"}`);
    if (!coachOk) fail("no successful coach TTS clip was fetched");
    if (coachBad.length) fail(`coach TTS failures: ${JSON.stringify(coachBad)}`);

    await page.click(MIC);
    await page.waitForSelector(`${MIC}.chat-input__mic--active`, { state: "detached", timeout: 10_000 });
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
await checkVoiceCoachDirect();
const helper = await chromium.launch();
await checkListenHandshake(helper);
const wavPath = await decodeToWav(helper, clips.partner);
await helper.close();
console.log(`fake-mic wav written: ${wavPath}`);
await checkVoiceConversation(wavPath);
await post(`/chat/history/clear?session_id=${encodeURIComponent(TEST_SESSION_ID)}`, {}).catch(() => {});
console.log("\nALL VOICE STAGES PASSED");
