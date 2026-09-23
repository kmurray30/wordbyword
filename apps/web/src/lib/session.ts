const SESSION_ID_KEY = "wordbyword.sessionId";

// In-memory fallback for when localStorage throws (private browsing,
// blocked storage, ...) - memoized so at least this page load is stable,
// even though it won't survive a refresh without persistence.
let inMemorySessionId: string | null = null;

// Not auth - just enough to give each browser its own conversation thread
// instead of one global chat history shared by every visitor (see
// README's "single user, no auth" note; the word bank itself stays
// global/shared on purpose, only the chat log is split up this way).
export function getSessionId(): string {
  try {
    const existing = localStorage.getItem(SESSION_ID_KEY);
    if (existing) return existing;
    const fresh = crypto.randomUUID();
    localStorage.setItem(SESSION_ID_KEY, fresh);
    return fresh;
  } catch {
    if (!inMemorySessionId) inMemorySessionId = crypto.randomUUID();
    return inMemorySessionId;
  }
}
