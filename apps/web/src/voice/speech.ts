import { api } from "../api/client";
import { AudioQueue } from "./audioQueue";
import { takeCompleteSentences } from "./sentences";

// Everything the app reads aloud goes through ONE queue, so the coach, the
// tutor and the partner never talk over each other: the coach's verdict
// and options and the tutor's answers in the coach voice (ElevenLabs,
// English with Spanish inside it), the partner's replies in their Spanish
// voice (Kokoro). Gated by the "Read aloud" setting in the menu.

const STORAGE_KEY = "wordbyword.readAloud";
const queue = new AudioQueue();
const listeners = new Set<() => void>();

let readAloud = (() => {
  try {
    return localStorage.getItem(STORAGE_KEY) !== "off";
  } catch {
    return true;
  }
})();

export function isReadAloudOn(): boolean {
  return readAloud;
}

export function setReadAloud(on: boolean): void {
  readAloud = on;
  try {
    localStorage.setItem(STORAGE_KEY, on ? "on" : "off");
  } catch {
    // per-device convenience only
  }
  if (!on) queue.stop();
  listeners.forEach((l) => l());
}

export function subscribeReadAloud(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

// iOS only plays audio that starts inside a tap: call this synchronously
// from every tap that may lead to speech (Send/Check, Done, the mic).
export function unlockSpeech(): void {
  queue.unlock();
}

export function stopSpeech(): void {
  queue.stop();
}

export function speakCoach(text: string): void {
  const clean = text.trim();
  if (!readAloud || !clean) return;
  void queue.enqueue(api.speak({ text: clean, language: "en", role: "coach" })).catch(() => {
    // One failed clip shouldn't stop the rest - the text is on screen.
  });
}

export function speakPartner(text: string, voice?: string): void {
  const clean = text.trim();
  if (!readAloud || !clean) return;
  void queue.enqueue(api.speak({ text: clean, language: "es", voice, role: "partner" })).catch(() => {});
}

// Speaks streamed text (the tutor's answer) sentence by sentence as it
// arrives, so the first sentence plays before the rest has been written.
export class SentenceSpeaker {
  private buffer = "";

  push(delta: string): void {
    this.buffer += delta;
    const { sentences, rest } = takeCompleteSentences(this.buffer);
    this.buffer = rest;
    sentences.forEach(speakCoach);
  }

  flush(): void {
    speakCoach(this.buffer);
    this.buffer = "";
  }
}

// --- the coach's result, as speech -----------------------------------------
// Built from the same streamed fields the popover shows, field by field as
// each completes, so the spoken version starts as soon as the meaning is
// known: "Did you mean: … ? If so, try: «…», which means «…». Or, more
// casually: …".

const OPTION_INTRO = ["Try", "Or, more casually", "Or, more formally"];

function sentence(text: string): string {
  const t = text.trim();
  return /[.!?…]["'”’»]?$/.test(t) ? t : `${t}.`;
}

function quoted(text: string): string {
  // Quotes keep the voice reading the Spanish as one phrase.
  return `“${text.trim().replace(/[.!]+$/, "")}”`;
}

export function coachMeaningSpeech(verdict: string, meaning: string): string {
  const m = meaning.trim().replace(/[.?!]+$/, "");
  if (!m) return "";
  if (verdict === "fix") return `Did you mean: ${m}? If so,`;
  if (verdict === "minor") return `Almost. You mean: ${m}.`;
  return `Sounds good. ${sentence(m)}`;
}

export function coachSuggestionSpeech(suggestion: string): string {
  return suggestion.trim() ? `Just a small fix: ${quoted(suggestion)}.` : "";
}

export function coachOptionSpeech(index: number, spanish: string, english: string): string {
  if (!spanish.trim()) return "";
  const intro = OPTION_INTRO[index] ?? "Or";
  const meaning = english.trim() ? `, which means ${quoted(english)}` : "";
  return `${intro}: ${quoted(spanish)}${meaning}.`;
}
