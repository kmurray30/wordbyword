import type { LanguageCode } from "./language";

// Spanish keeps the key it had before there were other languages, so an
// existing conversation carries on.
const sessionKey = (lang: LanguageCode) => (lang === "es" ? "wordbyword.sessionId" : `wordbyword.sessionId.${lang}`);

// In-memory fallback for when localStorage throws (private browsing,
// blocked storage, ...) - memoized so at least this page load is stable,
// even though it won't survive a refresh without persistence.
const inMemorySessionIds = new Map<LanguageCode, string>();

// Not auth - just enough to give each browser its own conversation thread
// instead of one global chat history shared by every visitor (see
// README's "single user, no auth" note). One per language: each language
// is its own conversation, with its own saved words and coach notes.
export function getSessionId(lang: LanguageCode): string {
  try {
    const existing = localStorage.getItem(sessionKey(lang));
    if (existing) return existing;
    const fresh = crypto.randomUUID();
    localStorage.setItem(sessionKey(lang), fresh);
    return fresh;
  } catch {
    if (!inMemorySessionIds.has(lang)) inMemorySessionIds.set(lang, crypto.randomUUID());
    return inMemorySessionIds.get(lang)!;
  }
}
