import { useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import type { DisplayMessage } from "./components/ChatMessage";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import { SegmentedControl } from "./components/SegmentedControl";
import { getSessionId } from "./lib/session";
import "./App.css";

let nextLocalId = -1;

const VOICE_STORAGE_KEY = "wordbyword.voice";

// "ef_dora" -> "Dora". Kokoro voice ids are "<lang><gender>_<name>".
function voiceLabel(voiceId: string): string {
  const name = voiceId.split("_")[1] ?? voiceId;
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function App() {
  const [sessionId] = useState(getSessionId);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // null = no turn in flight; "" (or more) while the assistant's reply is
  // streaming in - rendered as its own lightweight bubble below, OUTSIDE
  // ChatMessage, until the "done" event's complete message is ready to
  // push as one real DisplayMessage with a stable id. Keeping it out of
  // ChatMessage/messages until then avoids both remounting that component
  // under a changing key and misfiring its id-keyed eager-translation
  // effect against text that's still incomplete.
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  // Clear chat + the voice picker used to sit directly in the header,
  // taking up a full extra row under the title - folded into this
  // dropdown instead to keep the header to one line.
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [voices, setVoices] = useState<string[]>([]);
  const [voice, setVoice] = useState<string | null>(() => {
    try {
      return localStorage.getItem(VOICE_STORAGE_KEY);
    } catch {
      return null;
    }
  });

  useEffect(() => {
    api
      .listVoices("es")
      .then((res) => {
        setVoices(res.voices);
        setVoice((current) => (current && res.voices.includes(current) ? current : res.default));
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    api
      .chatHistory(sessionId)
      .then((res) => {
        const historyMessages: DisplayMessage[] = res.messages.map((m) => ({
          id: m.id,
          role: m.role as "user" | "assistant",
          text: m.text,
          tokens: m.tokens,
          fromHistory: true,
        }));
        // This fetch can resolve after the user has already sent a message
        // (a cold-started backend makes it slow enough for that to happen) -
        // replacing the array outright would silently erase whatever they
        // just sent. Keep anything already in state that isn't part of the
        // history response instead of overwriting it.
        setMessages((prev) => {
          const historyIds = new Set(historyMessages.map((m) => m.id));
          const notInHistory = prev.filter((m) => !historyIds.has(m.id));
          return [...historyMessages, ...notInHistory];
        });
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setHistoryLoaded(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  useEffect(() => {
    if (!menuOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [menuOpen]);

  const handleClear = () => {
    if (clearing) return;
    setClearing(true);
    api
      .clearChatHistory(sessionId)
      .then(() => setMessages([]))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setClearing(false));
    setMenuOpen(false);
  };

  const handleVoiceChange = (next: string) => {
    setVoice(next);
    try {
      localStorage.setItem(VOICE_STORAGE_KEY, next);
    } catch {
      // per-viewer convenience only - fine if it can't persist
    }
  };

  const handleSend = async (text: string) => {
    setError(null);
    setMessages((prev) => [...prev, { id: nextLocalId--, role: "user", text }]);
    setStreamingText("");

    try {
      for await (const event of api.chatTurnStream({ message: text, session_id: sessionId })) {
        if (event.type === "chunk") {
          setStreamingText((prev) => (prev ?? "") + event.data.delta);
        } else if (event.type === "error") {
          throw new Error(event.data.message);
        } else {
          const res = event.data;
          setMessages((prev) => [
            ...prev,
            { id: res.message_id, role: "assistant", text: res.text, tokens: res.tokens, translation: res.translation },
          ]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStreamingText(null);
    }
  };

  return (
    <div className="app">
      <header className="app__header">
        <h1>wordbyword</h1>
        <p className="app__tagline">Chat in Spanish, click any word to learn it</p>
        <div className="app__menu" ref={menuRef}>
          <button
            type="button"
            className="app__menu-toggle"
            onClick={() => setMenuOpen((open) => !open)}
            aria-label="Menu"
            aria-expanded={menuOpen}
            aria-haspopup="true"
          >
            ☰
          </button>
          {menuOpen && (
            <div className="app__menu-dropdown">
              {voices.length > 0 && voice && (
                <div className="app__voice-picker">
                  Voice:{" "}
                  <SegmentedControl
                    value={voice}
                    onChange={handleVoiceChange}
                    options={voices.map((v) => ({ value: v, label: voiceLabel(v) }))}
                  />
                </div>
              )}
              {messages.length > 0 && (
                <button type="button" className="app__clear-chat" onClick={handleClear} disabled={clearing}>
                  {clearing ? "Clearing…" : "Clear chat"}
                </button>
              )}
            </div>
          )}
        </div>
      </header>

      <main className="app__chat">
        {!historyLoaded && <p className="app__empty">Loading…</p>}
        {historyLoaded && messages.length === 0 && (
          <p className="app__empty">Say hello to start a conversation - ¡Hola!</p>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} message={m} voice={voice ?? undefined} />
        ))}
        {streamingText !== null && (
          <div className="chat-message chat-message--assistant">
            <div className="chat-message__column">
              <div className="chat-message__bubble">{streamingText || "…"}</div>
            </div>
          </div>
        )}
        {error && <p className="app__error">{error}. Is the backend running and configured correctly?</p>}
      </main>

      <ChatInput onSend={handleSend} sessionId={sessionId} />
    </div>
  );
}

export default App;
