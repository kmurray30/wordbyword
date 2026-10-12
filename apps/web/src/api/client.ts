import type { components, paths } from "./schema";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type ChatTurnRequest =
  paths["/chat/turn"]["post"]["requestBody"]["content"]["application/json"];
type ChatTurnResponse =
  paths["/chat/turn"]["post"]["responses"][200]["content"]["application/json"];
type ChatHistoryResponse =
  paths["/chat/history"]["get"]["responses"][200]["content"]["application/json"];
type ClearHistoryResponse =
  paths["/chat/history/clear"]["post"]["responses"][200]["content"]["application/json"];

type TranslateTextRequest =
  paths["/translate/text"]["post"]["requestBody"]["content"]["application/json"];
type TranslateTextResponse =
  paths["/translate/text"]["post"]["responses"][200]["content"]["application/json"];

type TagInputRequest =
  paths["/translate/tag-input"]["post"]["requestBody"]["content"]["application/json"];
type TagInputResponse =
  paths["/translate/tag-input"]["post"]["responses"][200]["content"]["application/json"];

type GlossSpansRequest =
  paths["/translate/gloss-spans"]["post"]["requestBody"]["content"]["application/json"];
type GlossSpansResponse =
  paths["/translate/gloss-spans"]["post"]["responses"][200]["content"]["application/json"];

type InterpretInputRequest =
  paths["/translate/interpret"]["post"]["requestBody"]["content"]["application/json"];
type InterpretInputResponse =
  paths["/translate/interpret"]["post"]["responses"][200]["content"]["application/json"];

type SaveUserInterpretationRequest =
  paths["/chat/message/interpretation"]["post"]["requestBody"]["content"]["application/json"];
type SaveUserInterpretationResponse =
  paths["/chat/message/interpretation"]["post"]["responses"][200]["content"]["application/json"];

// Reads any `event: <name>\ndata: <json>\n\n`-framed SSE response body
// incrementally, yielding each raw (name, data-string) pair as soon as its
// block completes - shared by every streaming endpoint below (brainstormStream,
// chatTurnStream) so the \n\n-delimited buffering/decoding logic exists in
// exactly one place. Each caller parses `data` into its own event union and
// filters `event` to the names it knows about.
async function* readServerSentEvents(response: Response): AsyncGenerator<{ event: string; data: string }> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let eventName = "";
      let dataLine = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice("event:".length).trim();
        else if (line.startsWith("data:")) dataLine = line.slice("data:".length).trim();
      }
      if (dataLine) yield { event: eventName, data: dataLine };
    }
  }
}

// /chat/turn/stream streams Server-Sent Events rather than a single JSON
// response (see app/routes/chat.py's take_turn_stream) - these mirror app/
// schemas.py's ChatTurnChunkEvent/ChatTurnErrorEvent by hand, same as the
// brainstorm types below; the "done" event reuses ChatTurnResponse unchanged,
// since its shape is already exactly what /chat/turn itself returns.
export interface ChatTurnChunkEvent {
  delta: string;
}

export interface ChatTurnErrorEvent {
  message: string;
}

// WS /voice/listen's server -> client messages (see app/routes/voice.py and
// app/stt/deepgram_client.py's TranscriptEvent). Hand-written like the
// other streaming shapes here - no OpenAPI type exists for a WebSocket.
export interface VoiceTranscriptWord {
  word: string;
  confidence: number;
}
export type VoiceTranscriptEvent =
  | { type: "interim"; text: string; words: VoiceTranscriptWord[] }
  | { type: "final"; text: string; words: VoiceTranscriptWord[] }
  | { type: "utterance_end"; text: string; words: VoiceTranscriptWord[] }
  | { type: "speech_started"; text: string; words: VoiceTranscriptWord[] }
  | { type: "error"; message: string };

// The backend's /voice/listen WebSocket address, derived from the same
// BASE_URL every HTTP call uses (http -> ws, https -> wss).
export function voiceListenUrl(): string {
  return `${BASE_URL.replace(/^http/, "ws")}/voice/listen`;
}

// POST /brainstorm/turn - one typed or spoken line in the Brainstorm field
// and the coach's streamed answer (see app/routes/brainstorm.py and
// app.translate.llm_translate.brainstorm_stream). SSE, so the event shapes
// are hand-written like the others here.
export type BrainstormTurnRequest =
  paths["/brainstorm/turn"]["post"]["requestBody"]["content"]["application/json"];
export type BrainstormLogTurn = components["schemas"]["BrainstormLogTurn"];
type BrainstormOpenResponse =
  paths["/brainstorm/open"]["get"]["responses"][200]["content"]["application/json"];
type FoldBrainstormRequest =
  paths["/brainstorm/fold"]["post"]["requestBody"]["content"]["application/json"];
export type FoldBrainstormResponse =
  paths["/brainstorm/fold"]["post"]["responses"][200]["content"]["application/json"];
export type BrainstormStreamEvent =
  | { type: "intent"; data: { intent: string } }
  | { type: "verdict"; data: { verdict: string } }
  | { type: "draft"; data: { draft: string } }
  | { type: "say_chunk"; data: { delta: string } }
  | { type: "say_complete"; data: { say: string } }
  | { type: "saved"; data: { learner_turn_id: number; coach_turn_id: number } }
  | { type: "error"; data: { message: string } };

const BRAINSTORM_EVENT_NAMES = new Set(["intent", "verdict", "draft", "say_chunk", "say_complete", "saved", "error"]);

async function* brainstormStream(body: BrainstormTurnRequest, signal?: AbortSignal): AsyncGenerator<BrainstormStreamEvent> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/brainstorm/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const detail = response.body ? await response.text() : "";
    throw new Error(`/brainstorm/turn failed (${response.status}): ${detail}`);
  }
  for await (const { event, data } of readServerSentEvents(response)) {
    if (BRAINSTORM_EVENT_NAMES.has(event)) {
      yield { type: event, data: JSON.parse(data) } as BrainstormStreamEvent;
    }
  }
}

async function brainstormOpen(sessionId: string): Promise<BrainstormOpenResponse> {
  const response = await fetchWithColdStartRetry(
    `${BASE_URL}/brainstorm/open?session_id=${encodeURIComponent(sessionId)}`,
  );
  if (!response.ok) throw new Error(`/brainstorm/open failed (${response.status})`);
  return response.json() as Promise<BrainstormOpenResponse>;
}

export type ChatTurnStreamEvent =
  | { type: "chunk"; data: ChatTurnChunkEvent }
  | { type: "done"; data: ChatTurnResponse }
  | { type: "error"; data: ChatTurnErrorEvent };

// Same turn as api.chatTurn, but the reply streams in as it's generated -
// "chunk" events carry each text delta for a live typing-style display;
// once generation finishes, one "done" event carries the exact same
// ChatTurnResponse shape the non-streaming call returns (glosses, word-bank
// updates and persistence all already happened server-side by then).
async function* chatTurnStream(body: ChatTurnRequest, signal?: AbortSignal): AsyncGenerator<ChatTurnStreamEvent> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/chat/turn/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const detail = response.body ? await response.text() : "";
    throw new Error(`/chat/turn/stream failed (${response.status}): ${detail}`);
  }
  for await (const { event, data } of readServerSentEvents(response)) {
    if (event === "chunk" || event === "done" || event === "error") {
      yield { type: event, data: JSON.parse(data) } as ChatTurnStreamEvent;
    }
  }
}

type RewardEventRequest =
  paths["/events/reward"]["post"]["requestBody"]["content"]["application/json"];
type RewardEventResponse =
  paths["/events/reward"]["post"]["responses"][200]["content"]["application/json"];

type TTSRequest =
  paths["/tts/speak"]["post"]["requestBody"]["content"]["application/json"];
type TTSVoicesResponse =
  paths["/tts/voices"]["get"]["responses"][200]["content"]["application/json"];

type LlmHealthResponse =
  paths["/health/llm"]["get"]["responses"][200]["content"]["application/json"];

// The backend runs with Railway's Serverless mode (sleeps after ~5-10min
// idle, wakes on the next request) - the documented signature of that
// wake-up window is a 502/503, or the request failing to connect at all,
// on the very first request after a quiet spell, not a real outage. Worth
// one short retry before treating it as a real failure - chatHistory and
// listVoices in particular fire automatically on page load, so without
// this, opening the app after it's been idle a while would show a broken
// error instead of just loading a beat slower.
async function fetchWithColdStartRetry(input: string, init?: RequestInit): Promise<Response> {
  for (let attempt = 0; ; attempt++) {
    try {
      const response = await fetch(input, init);
      if ((response.status === 502 || response.status === 503) && attempt === 0) {
        await new Promise((resolve) => setTimeout(resolve, 3000));
        continue;
      }
      return response;
    } catch (err) {
      if (attempt === 0) {
        await new Promise((resolve) => setTimeout(resolve, 3000));
        continue;
      }
      throw err;
    }
  }
}

async function post<Req, Res>(path: string, body: Req): Promise<Res> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${path} failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<Res>;
}

async function chatHistory(sessionId: string): Promise<ChatHistoryResponse> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/chat/history?session_id=${encodeURIComponent(sessionId)}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`/chat/history failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<ChatHistoryResponse>;
}

async function clearChatHistory(sessionId: string): Promise<ClearHistoryResponse> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/chat/history/clear?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`/chat/history/clear failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<ClearHistoryResponse>;
}

async function listVoices(language: string): Promise<TTSVoicesResponse> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/tts/voices?language=${encodeURIComponent(language)}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`/tts/voices failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<TTSVoicesResponse>;
}

async function speak(body: TTSRequest): Promise<Blob> {
  // Observed live: DeepInfra itself hung for the backend's full timeout on
  // every request during a rough patch, leaving the speaker button stuck
  // on its loading spinner with no feedback. A plain fetch() has no
  // built-in ceiling - abort client-side a bit past the backend's own
  // worst case (3 attempts * 15s timeout each, see deepinfra_client.py) so
  // this call always settles into a real error the UI can show, rather
  // than spinning indefinitely if something between the browser and the
  // backend hangs too, and - the actual live bug - rather than aborting
  // before the backend's own retries had a chance to succeed.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 50_000);
  try {
    const response = await fetchWithColdStartRetry(`${BASE_URL}/tts/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`/tts/speak failed (${response.status}): ${detail}`);
    }
    return await response.blob();
  } finally {
    clearTimeout(timeoutId);
  }
}

// Whether the model server can actually serve a chat completion right now -
// polled on page load to gate the UI (see App.tsx's LlmGate) rather than
// let the first real interaction be the one to hit a cold/sleeping model
// server. A non-2xx response (the backend itself still waking up, same
// Railway serverless-sleep situation) is just as much "not ready yet" as a
// { ready: false } body, not a reason to throw and stop polling.
async function healthLlm(): Promise<LlmHealthResponse> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/health/llm`);
  if (!response.ok) {
    return { ready: false };
  }
  return response.json() as Promise<LlmHealthResponse>;
}

export const api = {
  chatTurn: (body: ChatTurnRequest) => post<ChatTurnRequest, ChatTurnResponse>("/chat/turn", body),
  translateText: (body: TranslateTextRequest) =>
    post<TranslateTextRequest, TranslateTextResponse>("/translate/text", body),
  tagInput: (body: TagInputRequest) => post<TagInputRequest, TagInputResponse>("/translate/tag-input", body),
  glossSpans: (body: GlossSpansRequest) => post<GlossSpansRequest, GlossSpansResponse>("/translate/gloss-spans", body),
  interpretInput: (body: InterpretInputRequest) =>
    post<InterpretInputRequest, InterpretInputResponse>("/translate/interpret", body),
  saveUserInterpretation: (body: SaveUserInterpretationRequest) =>
    post<SaveUserInterpretationRequest, SaveUserInterpretationResponse>("/chat/message/interpretation", body),
  brainstormStream,
  brainstormOpen,
  foldBrainstorm: (body: FoldBrainstormRequest) =>
    post<FoldBrainstormRequest, FoldBrainstormResponse>("/brainstorm/fold", body),
  chatTurnStream,
  rewardEvent: (body: RewardEventRequest) =>
    post<RewardEventRequest, RewardEventResponse>("/events/reward", body),
  speak,
  listVoices,
  chatHistory,
  clearChatHistory,
  healthLlm,
};

export type TokenAnnotation = ChatTurnResponse["tokens"][number];
export type DraftToken = TagInputResponse["tokens"][number];
export type DraftSpan = GlossSpansResponse["spans"][number];
export type TranslateCandidate = components["schemas"]["TranslateCandidate"];
export type ChatHistoryMessage = ChatHistoryResponse["messages"][number];

export type {
  ChatTurnResponse,
  TranslateTextResponse,
  TagInputResponse,
  GlossSpansResponse,
  InterpretInputResponse,
  RewardEventRequest,
  TTSVoicesResponse,
  ChatHistoryResponse,
};
