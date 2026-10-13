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
type MessageGlossesRequest =
  paths["/chat/message/glosses"]["post"]["requestBody"]["content"]["application/json"];
export type MessageGlosses = components["schemas"]["MessageGlosses"];

type SaveUserInterpretationResponse =
  paths["/chat/message/interpretation"]["post"]["responses"][200]["content"]["application/json"];

type CoachDraftRequest =
  paths["/translate/coach"]["post"]["requestBody"]["content"]["application/json"];

// /translate/coach streams Server-Sent Events (see app/routes/translate.py's
// coach_draft and app.translate.llm_translate.coach_reply_stream): which
// partner message the reply is about (null = the learner's next message),
// whether the learner's own Spanish is ready to send, then the coach's
// reply text - suggestions are written inside it as [[Spanish | English]].
export type ThreadEntry = components["schemas"]["ThreadEntryOut"];
export type ThreadEntryIn = components["schemas"]["ThreadEntryIn"];

export type CoachStreamEvent =
  | { type: "about"; data: { message_id: number | null } }
  | { type: "ready"; data: { ready: boolean } }
  | { type: "reply_chunk"; data: { delta: string } }
  | { type: "reply_complete"; data: { reply: string; entries: ThreadEntry[] } }
  | { type: "error"; data: { message: string } };

// Reads any `event: <name>\ndata: <json>\n\n`-framed SSE response body
// incrementally, yielding each raw (name, data-string) pair as soon as its
// block completes - shared by every streaming endpoint below (coachDraftStream,
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

// Reads the coaching response's SSE stream incrementally, yielding each
// event as it completes - see CoachStreamEvent's own comment for the full
// field-by-field sequence. `signal` lets the caller abort mid-stream (e.g.
// the draft changed before coaching finished).
const COACH_EVENT_NAMES = new Set(["about", "ready", "reply_chunk", "reply_complete", "error"]);

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
  for await (const { event, data } of readServerSentEvents(response)) {
    if (COACH_EVENT_NAMES.has(event)) {
      yield { type: event, data: JSON.parse(data) } as CoachStreamEvent;
    }
  }
}

// /chat/turn/stream streams Server-Sent Events rather than a single JSON
// response (see app/routes/chat.py's take_turn_stream) - these mirror app/
// schemas.py's ChatTurnChunkEvent/ChatTurnErrorEvent by hand, same as the
// coach types above; the "done" event reuses ChatTurnResponse unchanged,
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

export type SavedWord = components["schemas"]["SavedWordOut"];
type SavedWordIn = components["schemas"]["SavedWordIn"];

// The learner's saved words (tap a word twice in a chat message).
async function listSavedWords(sessionId: string): Promise<SavedWord[]> {
  const response = await fetchWithColdStartRetry(`${BASE_URL}/words?session_id=${encodeURIComponent(sessionId)}`);
  if (!response.ok) throw new Error(`/words failed (${response.status})`);
  return ((await response.json()) as { words: SavedWord[] }).words;
}

async function removeSavedWord(sessionId: string, word: string): Promise<void> {
  const response = await fetchWithColdStartRetry(
    `${BASE_URL}/words?session_id=${encodeURIComponent(sessionId)}&word=${encodeURIComponent(word)}`,
    { method: "DELETE" },
  );
  if (!response.ok) throw new Error(`/words delete failed (${response.status})`);
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
  // Observed live: a TTS provider hung for the backend's full timeout on
  // every request during a rough patch, leaving the speaker button stuck
  // on its loading spinner with no feedback. A plain fetch() has no
  // built-in ceiling - abort client-side a bit past the backend's own
  // worst case (3 attempts * 20s timeout each, see elevenlabs_client.py) so
  // this call always settles into a real error the UI can show, rather
  // than spinning indefinitely if something between the browser and the
  // backend hangs too, and - the actual live bug - rather than aborting
  // before the backend's own retries had a chance to succeed.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 65_000);
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
  // A message's word/phrase glosses, made and saved on first request.
  messageGlosses: (body: MessageGlossesRequest) =>
    post<MessageGlossesRequest, MessageGlosses>("/chat/message/glosses", body),
  coachDraftStream,
  chatTurnStream,
  rewardEvent: (body: RewardEventRequest) =>
    post<RewardEventRequest, RewardEventResponse>("/events/reward", body),
  speak,
  listVoices,
  chatHistory,
  clearChatHistory,
  listSavedWords,
  saveWord: (body: SavedWordIn) => post<SavedWordIn, SavedWord>("/words", body),
  removeSavedWord,
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
