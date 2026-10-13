import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { api } from "./api/client";
import type { ThreadEntry, ThreadEntryIn } from "./api/client";
import type { DisplayMessage } from "./components/ChatMessage";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import type { ChatInputControl } from "./components/ChatInput";
import { NextMessagePanel } from "./components/NextMessagePanel";
import type { CoachEvent, NextMessageControl } from "./components/NextMessagePanel";
import type { LiveCoachReply } from "./components/MessageThread";
import { SegmentedControl } from "./components/SegmentedControl";
import { getSessionId } from "./lib/session";
import { useAppUpdate } from "./lib/useAppUpdate";
import { isReadAloudOn, setReadAloud, speakPartner, subscribeReadAloud } from "./voice/speech";
import "./App.css";

let nextLocalId = -1;

// The backend's id for a message (a freshly sent user message only learns
// it from the turn's "done" event - see handleSend).
function serverIdOf(m: DisplayMessage): number | null {
  if (m.serverId !== undefined) return m.serverId;
  return m.id > 0 ? m.id : null;
}

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
  const readAloud = useSyncExternalStore(subscribeReadAloud, isReadAloudOn);
  // Which messages have their side conversation expanded (by display id).
  const [openThreads, setOpenThreads] = useState<Set<number>>(new Set());
  // The thread the learner is currently in: follow-up questions continue
  // it. Cleared when they say something to the partner.
  const [activeThreadId, setActiveThreadId] = useState<number | null>(null);
  // The coach's reply while it streams, and which message it's about
  // (null = no message to anchor to yet, shown at the end of the chat).
  const [liveCoach, setLiveCoach] = useState<(LiveCoachReply & { messageId: number }) | null>(null);
  const inputControlRef = useRef<ChatInputControl | null>(null);
  const nextMessageRef = useRef<NextMessageControl | null>(null);
  // Text approved to send (the coach said it's fine, or it came from a
  // suggestion the learner chose) - the Send button goes green for it.
  const [readyText, setReadyText] = useState<string | null>(null);
  // What's in the message box right now (the coach panel marks a used
  // suggestion only while the box still holds it).
  const [messageText, setMessageText] = useState("");
  const threadAnchorRef = useRef<HTMLDivElement | null>(null);
  // A newer build is deployed: update straight away unless that would lose
  // something typed or still arriving.
  const { updateReady, reload } = useAppUpdate(
    () =>
      !messageText.trim() &&
      streamingText === null &&
      liveCoach === null &&
      !(nextMessageRef.current?.busy() ?? false),
  );
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
          native: m.native,
          target: m.target,
          thread: m.thread,
          glosses: m.glosses,
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
      .then(() => {
        setMessages([]);
        nextMessageRef.current?.clear();
        setActiveThreadId(null);
        setOpenThreads(new Set());
      })
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

  // Resolves with the partner's completed reply, or null if the turn
  // failed. `thread` = the coach checks behind this message.
  const handleSend = async (text: string): Promise<{ text: string } | null> => {
    setError(null);
    setActiveThreadId(null);
    setReadyText(null);
    // Everything worked out in "Your next message" becomes this message's
    // thread.
    const thread: ThreadEntryIn[] = nextMessageRef.current?.takeThread() ?? [];
    const userLocalId = nextLocalId--;
    const localThread: ThreadEntry[] = thread.map((e) => ({ ...e, id: nextLocalId-- }));
    setMessages((prev) => [...prev, { id: userLocalId, role: "user", text, thread: localThread }]);
    setStreamingText("");
    let reply: { text: string } | null = null;

    try {
      for await (const event of api.chatTurnStream({ message: text, session_id: sessionId, thread })) {
        if (event.type === "chunk") {
          setStreamingText((prev) => (prev ?? "") + event.data.delta);
        } else if (event.type === "error") {
          throw new Error(event.data.message);
        } else {
          const res = event.data;
          reply = { text: res.text };
          speakPartner(res.text, voice ?? undefined);
          setMessages((prev) => [
            // Patches the real backend id onto the user message just sent
            // (see ChatMessage.tsx's persist effect) without touching its
            // `id` - that's still used as this list's React key, and
            // swapping it for the server's id here would remount that
            // bubble instead of just updating a prop.
            ...prev.map((m) => (m.id === userLocalId ? { ...m, serverId: res.user_message_id } : m)),
            {
              id: res.message_id,
              role: "assistant",
              text: res.text,
              tokens: res.tokens,
              translation: res.translation,
              glosses: res.glosses,
            },
          ]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStreamingText(null);
    }
    return reply;
  };

  const openThreadFor = (messageId: number | null) => {
    if (messageId === null) return;
    const target = messages.find((m) => serverIdOf(m) === messageId);
    if (target) setOpenThreads((prev) => new Set(prev).add(target.id));
  };

  // Mirrors liveCoach synchronously: events arrive faster than renders, and
  // "done" needs the question/anchor that "start" set.
  const liveCoachRef = useRef<(LiveCoachReply & { messageId: number }) | null>(null);
  const handleCoachEvent = (event: CoachEvent) => {
    if (event.type === "start") {
      liveCoachRef.current = { messageId: event.messageId, question: event.question, reply: "", error: null };
      setLiveCoach(liveCoachRef.current);
      setActiveThreadId(event.messageId);
      openThreadFor(event.messageId);
    } else if (event.type === "chunk") {
      setLiveCoach((prev) => (prev ? { ...prev, reply: prev.reply + event.delta } : prev));
    } else if (event.type === "done") {
      const live = liveCoachRef.current;
      liveCoachRef.current = null;
      setLiveCoach(null);
      if (!live) return;
      const entries: ThreadEntry[] = event.entries.length
        ? event.entries
        : [
            { id: nextLocalId--, role: "learner", kind: "question", text: live.question, data: "" },
            { id: nextLocalId--, role: "coach", kind: "answer", text: event.reply, data: "" },
          ];
      setMessages((prev) =>
        prev.map((m) => (serverIdOf(m) === live.messageId ? { ...m, thread: [...(m.thread ?? []), ...entries] } : m)),
      );
    } else if (event.type === "error") {
      setLiveCoach((prev) => (prev ? { ...prev, error: event.message } : prev));
    }
  };

  const toggleThread = (m: DisplayMessage) => {
    const open = !openThreads.has(m.id);
    setOpenThreads((prev) => {
      const next = new Set(prev);
      if (open) next.add(m.id);
      else next.delete(m.id);
      return next;
    });
    // An open thread is the one follow-up questions continue.
    setActiveThreadId(open ? serverIdOf(m) : null);
  };

  // The newest reply's popups open upward (see ChatMessage's `latest`).
  const latestAssistantId = [...messages].reverse().find((m) => m.role === "assistant")?.id;

  // Keep the streaming answer in view.
  useEffect(() => {
    if (liveCoach) threadAnchorRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [liveCoach]);
  return (
    <div className="app">
      <header className="app__header">
        <h1>wordbyword</h1>
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
              <div className="app__voice-picker">
                Read aloud:{" "}
                <SegmentedControl
                  value={readAloud ? "on" : "off"}
                  onChange={(v) => setReadAloud(v === "on")}
                  options={[
                    { value: "on", label: "On" },
                    { value: "off", label: "Off" },
                  ]}
                />
              </div>
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

      {updateReady && (
        <button type="button" className="app__update" onClick={reload}>
          A new version of the app is ready - tap to update
        </button>
      )}
      <main className="app__chat">
        {!historyLoaded && <p className="app__empty">Loading…</p>}
        {historyLoaded && messages.length === 0 && (
          <p className="app__empty">Say hello to start a conversation - ¡Hola!</p>
        )}
        {messages.map((m) => {
          const live = liveCoach && liveCoach.messageId === serverIdOf(m) ? liveCoach : null;
          const latest = m.role === "assistant" && m.id === latestAssistantId && streamingText === null;
          return (
            <div key={m.id} ref={live ? threadAnchorRef : undefined}>
              <ChatMessage
                message={m}
                voice={voice ?? undefined}
                threadOpen={openThreads.has(m.id) || !!live}
                onToggleThread={() => toggleThread(m)}
                liveCoach={live}
                latest={latest}
              />
            </div>
          );
        })}
        {streamingText !== null && (
          <div className="chat-message chat-message--assistant">
            <div className="chat-message__column">
              <div className="chat-message__bubble">{streamingText || "…"}</div>
            </div>
          </div>
        )}
        {error && <p className="app__error">{error}. Is the backend running and configured correctly?</p>}
      </main>

      <NextMessagePanel
        sessionId={sessionId}
        activeThreadId={activeThreadId}
        messageText={messageText}
        onCoachEvent={handleCoachEvent}
        onUse={(text) => inputControlRef.current?.setText(text)}
        onReady={setReadyText}
        controlRef={nextMessageRef}
      />
      <ChatInput
        onSend={(text) => void handleSend(text)}
        onCheck={(text) => nextMessageRef.current?.check(text)}
        sessionId={sessionId}
        readyText={readyText}
        onValueChange={setMessageText}
        controlRef={inputControlRef}
      />
    </div>
  );
}

export default App;
