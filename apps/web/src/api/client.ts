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

type CoachDraftRequest =
  paths["/translate/coach"]["post"]["requestBody"]["content"]["application/json"];

// /translate/coach streams Server-Sent Events rather than a single JSON
// response (see app/routes/translate.py's coach_draft), so there's no
// OpenAPI-generated response type for it - these mirror app/schemas.py's
// CoachOption/CoachCoreEvent/CoachTranslationsEvent/CoachErrorEvent by
// hand instead.
export interface CoachOption {
  formality: string;
  spanish: string;
  // Both empty until the "translations" event arrives - see
  // CoachStreamEvent below.
  english: string;
  spans: DraftSpan[];
}

export interface CoachCoreEvent {
  meaning: string;
  feedback: string;
  options: CoachOption[];
}

export interface CoachTranslationsEvent {
  // Parallel to the core event's options by list position.
  options: CoachOption[];
}

export interface CoachErrorEvent {
  message: string;
}

export type CoachStreamEvent =
  | { type: "core"; data: CoachCoreEvent }
  | { type: "translations"; data: CoachTranslationsEvent }
  | { type: "error"; data: CoachErrorEvent };

function parseSseBlock(block: string): CoachStreamEvent | null {
  let eventName = "";
  let dataLine = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) dataLine = line.slice("data:".length).trim();
  }
  if (!dataLine) return null;
  if (eventName === "core" || eventName === "translations" || eventName === "error") {
    return { type: eventName, data: JSON.parse(dataLine) } as CoachStreamEvent;
  }
  return null;
}

// Reads the coaching response's SSE stream incrementally, yielding each
// event as it completes - "core" (the feedback + options themselves,
// usable right away) typically arrives well before "translations" (each
// option's own English translation + word-by-word breakdown, streamed
// from the SAME underlying LLM call - see coach_draft_stream's docstring
// in app/translate/llm_translate.py). `signal` lets the caller abort mid-
// stream (e.g. the draft changed before coaching finished).
async function* coachDraftStream(body: CoachDraftRequest, signal?: AbortSignal): AsyncGenerator<CoachStreamEvent> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/translate/coach`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const detail = response.body ? await response.text() : "";
    throw new Error(`/translate/coach failed (${response.status}): ${detail}`);
  }
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
      const event = parseSseBlock(block);
      if (event) yield event;
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

type SettingsResponse =
  paths["/settings"]["get"]["responses"][200]["content"]["application/json"];
type UpdateSettingsRequest =
  paths["/settings"]["put"]["requestBody"]["content"]["application/json"];

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

async function getSettings(): Promise<SettingsResponse> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/settings`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`/settings failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<SettingsResponse>;
}

async function updateSettings(body: UpdateSettingsRequest): Promise<SettingsResponse> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`/settings failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<SettingsResponse>;
}

export const api = {
  chatTurn: (body: ChatTurnRequest) => post<ChatTurnRequest, ChatTurnResponse>("/chat/turn", body),
  translateText: (body: TranslateTextRequest) =>
    post<TranslateTextRequest, TranslateTextResponse>("/translate/text", body),
  tagInput: (body: TagInputRequest) => post<TagInputRequest, TagInputResponse>("/translate/tag-input", body),
  glossSpans: (body: GlossSpansRequest) => post<GlossSpansRequest, GlossSpansResponse>("/translate/gloss-spans", body),
  interpretInput: (body: InterpretInputRequest) =>
    post<InterpretInputRequest, InterpretInputResponse>("/translate/interpret", body),
  coachDraftStream,
  rewardEvent: (body: RewardEventRequest) =>
    post<RewardEventRequest, RewardEventResponse>("/events/reward", body),
  speak,
  listVoices,
  chatHistory,
  clearChatHistory,
  healthLlm,
  getSettings,
  updateSettings,
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
  SettingsResponse,
};
