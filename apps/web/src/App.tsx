import { useEffect, useState } from "react";
import { api } from "./api/client";
import type { DisplayMessage } from "./components/ChatMessage";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import "./App.css";

let nextLocalId = -1;

const VOICE_STORAGE_KEY = "wordbyword.voice";

// "ef_dora" -> "Dora". Kokoro voice ids are "<lang><gender>_<name>".
function voiceLabel(voiceId: string): string {
  const name = voiceId.split("_")[1] ?? voiceId;
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function App() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
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

  const handleVoiceChange = (next: string) => {
    setVoice(next);
    try {
      localStorage.setItem(VOICE_STORAGE_KEY, next);
    } catch {
      // per-viewer convenience only - fine if it can't persist
    }
  };

  const handleSend = (text: string) => {
    setError(null);
    setMessages((prev) => [...prev, { id: nextLocalId--, role: "user", text }]);
    setSending(true);

    api
      .chatTurn({ message: text })
      .then((res) => {
        setMessages((prev) => [
          ...prev,
          { id: res.message_id, role: "assistant", text: res.text, tokens: res.tokens },
        ]);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setSending(false));
  };

  return (
    <div className="app">
      <header className="app__header">
        <h1>wordbyword</h1>
        <p>Chat in Spanish. Hover any word for a translation.</p>
        {voices.length > 0 && voice && (
          <label className="app__voice-picker">
            Voice:{" "}
            <select value={voice} onChange={(e) => handleVoiceChange(e.target.value)}>
              {voices.map((v) => (
                <option key={v} value={v}>
                  {voiceLabel(v)}
                </option>
              ))}
            </select>
          </label>
        )}
      </header>

      <main className="app__chat">
        {messages.length === 0 && (
          <p className="app__empty">Say hello to start a conversation - ¡Hola!</p>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} message={m} voice={voice ?? undefined} />
        ))}
        {sending && <p className="app__typing">…</p>}
        {error && (
          <p className="app__error">
            {error}. Is the backend running, and is Ollama up at the configured URL?
          </p>
        )}
      </main>

      <ChatInput onSend={handleSend} />
    </div>
  );
}

export default App;
