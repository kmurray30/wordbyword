import type { paths } from "./schema";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type ChatTurnRequest =
  paths["/chat/turn"]["post"]["requestBody"]["content"]["application/json"];
type ChatTurnResponse =
  paths["/chat/turn"]["post"]["responses"][200]["content"]["application/json"];

type TranslateWordRequest =
  paths["/translate/word"]["post"]["requestBody"]["content"]["application/json"];
type TranslateWordResponse =
  paths["/translate/word"]["post"]["responses"][200]["content"]["application/json"];

type TranslateTextRequest =
  paths["/translate/text"]["post"]["requestBody"]["content"]["application/json"];
type TranslateTextResponse =
  paths["/translate/text"]["post"]["responses"][200]["content"]["application/json"];

type TagInputRequest =
  paths["/translate/tag-input"]["post"]["requestBody"]["content"]["application/json"];
type TagInputResponse =
  paths["/translate/tag-input"]["post"]["responses"][200]["content"]["application/json"];

type RewardEventRequest =
  paths["/events/reward"]["post"]["requestBody"]["content"]["application/json"];
type RewardEventResponse =
  paths["/events/reward"]["post"]["responses"][200]["content"]["application/json"];

type TTSRequest =
  paths["/tts/speak"]["post"]["requestBody"]["content"]["application/json"];

async function post<Req, Res>(path: string, body: Req): Promise<Res> {
  const response = await fetch(`${BASE_URL}${path}`, {
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

async function speak(body: TTSRequest): Promise<Blob> {
  const response = await fetch(`${BASE_URL}/tts/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`/tts/speak failed (${response.status}): ${detail}`);
  }
  return response.blob();
}

export const api = {
  chatTurn: (body: ChatTurnRequest) => post<ChatTurnRequest, ChatTurnResponse>("/chat/turn", body),
  translateWord: (body: TranslateWordRequest) =>
    post<TranslateWordRequest, TranslateWordResponse>("/translate/word", body),
  translateText: (body: TranslateTextRequest) =>
    post<TranslateTextRequest, TranslateTextResponse>("/translate/text", body),
  tagInput: (body: TagInputRequest) => post<TagInputRequest, TagInputResponse>("/translate/tag-input", body),
  rewardEvent: (body: RewardEventRequest) =>
    post<RewardEventRequest, RewardEventResponse>("/events/reward", body),
  speak,
};

export type TokenAnnotation = ChatTurnResponse["tokens"][number];
export type InputTokenAnnotation = TagInputResponse["tokens"][number];
export type TranslateCandidate = TranslateWordResponse["candidates"][number];

export type {
  ChatTurnResponse,
  TranslateWordResponse,
  TranslateTextResponse,
  TagInputResponse,
  RewardEventRequest,
};
