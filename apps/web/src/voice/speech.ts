import { api } from "../api/client";
import { AudioQueue } from "./audioQueue";
import { takeCompleteSentences } from "./sentences";
import { speakableCoachText } from "../lib/coachText";

// Everything the app reads aloud goes through ONE queue, so the coach and
// the partner never talk over each other: the coach's replies in the coach
// voice (ElevenLabs, English with Spanish inside it), the partner's replies
// in their Spanish voice (Kokoro). Gated by the "Read aloud" setting in the
// menu.

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

// Speaks a streamed coach reply as it arrives: plain sentences as soon as
// they're complete, each [[Spanish | English]] suggestion as “Spanish”,
// which means “English” once its block has closed.
export class CoachSpeaker {
  private raw = "";
  private spokenUpTo = 0;

  push(delta: string): void {
    this.raw += delta;
    const { speakable } = speakableCoachText(this.raw);
    const pending = speakable.slice(this.spokenUpTo);
    const { sentences, rest } = takeCompleteSentences(pending);
    sentences.forEach(speakCoach);
    this.spokenUpTo = speakable.length - rest.length;
  }

  flush(): void {
    const { speakable } = speakableCoachText(this.raw);
    speakCoach(speakable.slice(this.spokenUpTo));
    this.spokenUpTo = speakable.length;
  }
}
