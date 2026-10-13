import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";
import { api } from "./api/client";
import type { ThreadEntry, ThreadEntryIn } from "./api/client";
import type { DisplayMessage } from "./components/ChatMessage";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import type { ChatInputControl } from "./components/ChatInput";
import { NextMessagePanel } from "./components/NextMessagePanel";
import type { NextMessageControl } from "./components/NextMessagePanel";
import type { LiveCoachReply } from "./components/MessageThread";
import { SegmentedControl } from "./components/SegmentedControl";
import { getSessionId } from "./lib/session";
import { useAppUpdate } from "./lib/useAppUpdate";
import { useChatAutoScroll } from "./lib/useChatAutoScroll";
import { syncVisibleViewport, useKeyboardUp } from "./lib/visibleViewport";
import { CoachSpeaker, isReadAloudOn, setReadAloud, speakPartner, stopSpeech, subscribeReadAloud } from "./voice/speech";
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
  // The coach's reply while it streams, and which message it's about.
  const [liveCoach, setLiveCoach] = useState<(LiveCoachReply & { messageId: number }) | null>(null);
  const inputControlRef = useRef<ChatInputControl | null>(null);
  const nextMessageRef = useRef<NextMessageControl | null>(null);
  // Text approved to send (the coach said it's fine, or it came from a
  // suggestion the learner chose) - the Send button goes green for it.
  const [readyText, setReadyText] = useState<string | null>(null);
  // What's in the message box right now (the coach panel marks a used
  // suggestion only while the box still holds it).
  const [messageText, setMessageText] = useState("");
  const chatRef = useRef<HTMLElement | null>(null);
  const chatContentRef = useRef<HTMLDivElement | null>(null);
  const chatScroll = useChatAutoScroll(chatRef, chatContentRef);

  // Saved words (lowercased) - see ChatMessage's add-word flow.
  const [savedWords, setSavedWords] = useState<Set<string>>(new Set());
  useEffect(() => {
    api
      .listSavedWords(sessionId)
      .then((words) => setSavedWords(new Set(words.map((w) => w.word.toLowerCase()))))
      .catch(() => {});
  }, [sessionId]);
  const saveWord = (entry: { word: string; gloss: string; grammar: string; sentence: string; messageId?: number }) => {
    setSavedWords((prev) => new Set(prev).add(entry.word.toLowerCase()));
    api
      .saveWord({
        session_id: sessionId,
        word: entry.word,
        gloss: entry.gloss,
        grammar: entry.grammar,
        sentence: entry.sentence,
        message_id: entry.messageId ?? null,
      })
      .catch(() => {});
  };
  const removeWord = (word: string) => {
    setSavedWords((prev) => {
      const next = new Set(prev);
      next.delete(word.toLowerCase());
      return next;
    });
    api.removeSavedWord(sessionId, word).catch(() => {});
  };
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

  // A conversation with the coach about one message (opened from its 💬):
  // the reply streams into that message's thread, is read aloud, and is
  // saved there by the server.
  const askCoachAbout = async (m: DisplayMessage, question: string) => {
    const messageId = serverIdOf(m);
    if (messageId === null) return;
    stopSpeech();
    const speaker = new CoachSpeaker();
    setLiveCoach({ messageId, question, reply: "", error: null });
    let reply = "";
    try {
      for await (const event of api.coachDraftStream({
        text: question,
        session_id: sessionId,
        thread_message_id: messageId,
        message_draft: messageText,
        draft_thread: [],
        check: false,
      })) {
        if (event.type === "reply_chunk") {
          reply += event.data.delta;
          speaker.push(event.data.delta);
          setLiveCoach((prev) => (prev ? { ...prev, reply: prev.reply + event.data.delta } : prev));
        } else if (event.type === "reply_complete") {
          speaker.flush();
          reply = event.data.reply;
          const entries: ThreadEntry[] = event.data.entries.length
            ? event.data.entries
            : [
                { id: nextLocalId--, role: "learner", kind: "question", text: question, data: "" },
                { id: nextLocalId--, role: "coach", kind: "answer", text: reply, data: "" },
              ];
          setLiveCoach(null);
          setMessages((prev) =>
            prev.map((x) => (serverIdOf(x) === messageId ? { ...x, thread: [...(x.thread ?? []), ...entries] } : x)),
          );
          requestAnimationFrame(() => revealConversation(m.id, false));
        } else if (event.type === "error") {
          throw new Error(event.data.message);
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setLiveCoach((prev) => (prev ? { ...prev, error: message } : prev));
    }
  };

  // Scroll position each conversation was opened from: closing it glides
  // back there, so the message returns to where it was.
  const openedFromRef = useRef(new Map<number, number>());

  const toggleThread = (m: DisplayMessage) => {
    // The bubble stays put and its conversation opens below it (see
    // revealConversation), instead of the chat re-pinning to the bottom.
    chatScroll.hold();
    const opening = !openThreads.has(m.id);
    // Where the message sits on screen before it opens (see placeOpened).
    const before = opening ? messageBox(m.id) : null;
    const scroller = chatRef.current;
    if (opening && scroller) openedFromRef.current.set(m.id, scroller.scrollTop);
    if (!opening) {
      const from = openedFromRef.current.get(m.id);
      openedFromRef.current.delete(m.id);
      if (typingInThread === m.id) setTypingInThread(null);
      if (placingRef.current?.id === m.id) setPlacing(null);
      if (from !== undefined && scroller) requestAnimationFrame(() => scroller.scrollTo({ top: from, behavior: "smooth" }));
    }
    setOpenThreads((prev) => {
      const next = new Set(prev);
      if (next.has(m.id)) next.delete(m.id);
      else next.add(m.id);
      return next;
    });
    if (opening && before) {
      const hasConversation = (m.thread ?? []).length > 0;
      requestAnimationFrame(() => (hasConversation ? placeOpened(m.id, before) : revealConversation(m.id)));
    }
  };

  // The newest reply's popups open upward (see ChatMessage's `latest`).
  const latestAssistantId = [...messages].reverse().find((m) => m.role === "assistant")?.id;

  // A message's coach conversation opens downward under it; if that runs
  // past the bottom of the screen, scroll just enough to bring its bottom
  // into view (pushing the bubble up - and if it's all taller than the
  // screen, the top of the bubble is what gets cut off). Same while the
  // coach's answer streams in.
  const revealConversation = (displayId: number, smooth = true) => {
    // Typing in its box: the typing placement (below) is in charge.
    if (placingRef.current?.id === displayId) return;
    const scroller = chatRef.current;
    const el = scroller?.querySelector(`[data-message-id="${displayId}"]`);
    if (!scroller || !el) return;
    const overflow = el.getBoundingClientRect().bottom + 8 - scroller.getBoundingClientRect().bottom;
    if (overflow > 0) scroller.scrollBy({ top: overflow, behavior: smooth ? "smooth" : "auto" });
  };
  // A message's top and height relative to the visible part of the chat.
  const messageBox = (displayId: number) => {
    const scroller = chatRef.current;
    const el = scroller?.querySelector(`[data-message-id="${displayId}"]`);
    if (!scroller || !el) return null;
    const view = scroller.getBoundingClientRect();
    const rect = el.getBoundingClientRect();
    return { top: rect.top - view.top, height: rect.height, view: view.height };
  };

  // Opening a conversation that has something in it: keep the message (bubble
  // + conversation) at the same RELATIVE spot on screen. Where it sat in the
  // free space before - 0 at the top, 1 at the bottom - is where it sits in
  // the (smaller) free space after. At the top nothing moves; at the bottom
  // the conversation's bottom lands on the bottom of the screen; in between
  // it ends up proportionally, roughly centered. Taller than the screen: its
  // bottom stays visible and the top of the bubble is cut off.
  const placeOpened = (displayId: number, before: { top: number; height: number; view: number }) => {
    const scroller = chatRef.current;
    const after = messageBox(displayId);
    if (!scroller || !after) return;
    const freeBefore = before.view - before.height;
    const fraction = freeBefore > 0 ? Math.min(1, Math.max(0, before.top / freeBefore)) : 1;
    const freeAfter = after.view - after.height;
    const targetTop = freeAfter >= 0 ? fraction * freeAfter : freeAfter;
    const delta = after.top - targetTop;
    if (Math.abs(delta) > 1) scroller.scrollBy({ top: delta, behavior: "smooth" });
  };

  // Typing in a message's coach box: the other two boxes step aside, and
  // the message (bubble + conversation) keeps the same RELATIVE spot on
  // screen as the keyboard takes its share - the placeOpened rule, applied
  // to the smaller space above the keyboard. Where it sat when you tapped
  // into the box (0 = top, 1 = bottom of the free space) is where it stays,
  // re-placed every frame while the keyboard slides and whenever the
  // conversation or the screen changes size, so it follows instead of
  // jumping. Taller than the space: its bottom (your box) sits right above
  // the keyboard. Leaving the box: the same rule follows the keyboard back
  // down, then lets go.
  const [typingInThread, setTypingInThread] = useState<number | null>(null);
  const [placing, setPlacing] = useState<{ id: number; fraction: number } | null>(null);
  const placingRef = useRef(placing);
  placingRef.current = placing;
  const placingStopRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The bars step aside together with the keyboard coming up (and come back
  // as it goes down), so the conversation moves once, not twice.
  const keyboardUp = useKeyboardUp();
  const barsAside = keyboardUp && (typingInThread !== null || placing !== null);

  const handleThreadTyping = (displayId: number, typing: boolean) => {
    if (placingStopRef.current) clearTimeout(placingStopRef.current);
    placingStopRef.current = null;
    if (typing) {
      // Measured before anything moves (the bars, the keyboard).
      const box = messageBox(displayId);
      const free = box ? box.view - box.height : 0;
      const fraction = box && free > 0 ? Math.min(1, Math.max(0, box.top / free)) : 1;
      setPlacing((prev) => (prev?.id === displayId ? prev : { id: displayId, fraction }));
      setTypingInThread(displayId);
    } else {
      setTypingInThread(null);
      placingStopRef.current = setTimeout(() => setPlacing(null), 900);
    }
  };

  useEffect(() => {
    if (!placing) return;
    const scroller = chatRef.current;
    const el = scroller?.querySelector(`[data-message-id="${placing.id}"]`);
    if (!scroller || !el) return;
    chatScroll.pause(true);
    const place = () => {
      syncVisibleViewport();
      const box = messageBox(placing.id);
      if (!box) return;
      const free = box.view - box.height;
      const target = free >= 0 ? placing.fraction * free : free;
      const delta = box.top - target;
      if (Math.abs(delta) > 1) scroller.scrollTop += delta;
    };
    // Follow every frame for a while after anything moves (the keyboard
    // animates over ~0.3-0.5s, and iOS only reports where it ended up).
    let frame = 0;
    let until = 0;
    const tick = () => {
      place();
      frame = performance.now() < until ? requestAnimationFrame(tick) : 0;
    };
    const follow = () => {
      until = performance.now() + 700;
      if (!frame) frame = requestAnimationFrame(tick);
    };
    follow();
    const vv = window.visualViewport;
    vv?.addEventListener("resize", follow);
    vv?.addEventListener("scroll", follow);
    const observer = new ResizeObserver(place);
    observer.observe(el);
    observer.observe(scroller);
    return () => {
      cancelAnimationFrame(frame);
      vv?.removeEventListener("resize", follow);
      vv?.removeEventListener("scroll", follow);
      observer.disconnect();
      chatScroll.pause(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [placing]);
  useEffect(
    () => () => {
      if (placingStopRef.current) clearTimeout(placingStopRef.current);
    },
    [],
  );

  const liveDisplayId = liveCoach ? messages.find((m) => serverIdOf(m) === liveCoach.messageId)?.id : undefined;
  useLayoutEffect(() => {
    if (liveDisplayId !== undefined) revealConversation(liveDisplayId, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveCoach, liveDisplayId]);
  return (
    <div className={`app${barsAside ? " app--thread-typing" : ""}`}>
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
      <main className="app__chat" ref={chatRef}>
        <div className="app__chat-content" ref={chatContentRef}>
        {!historyLoaded && <p className="app__empty">Loading…</p>}
        {historyLoaded && messages.length === 0 && (
          <p className="app__empty">Say hello to start a conversation - ¡Hola!</p>
        )}
        {messages.map((m) => {
          const live = liveCoach && liveCoach.messageId === serverIdOf(m) ? liveCoach : null;
          const latest = m.role === "assistant" && m.id === latestAssistantId && streamingText === null;
          return (
            <div key={m.id} data-message-id={m.id}>
              <ChatMessage
                message={m}
                voice={voice ?? undefined}
                threadOpen={openThreads.has(m.id) || !!live}
                onToggleThread={() => toggleThread(m)}
                onAskCoach={(question) => void askCoachAbout(m, question)}
                onThreadTyping={(typing) => handleThreadTyping(m.id, typing)}
                canAskCoach={serverIdOf(m) !== null && !liveCoach}
                sessionId={sessionId}
                liveCoach={live}
                latest={latest}
                savedWords={savedWords}
                onSaveWord={saveWord}
                onRemoveWord={removeWord}
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
        </div>
      </main>

      <ChatInput
        onSend={(text) => {
          chatScroll.toBottom();
          void handleSend(text);
        }}
        onCheck={(text) => nextMessageRef.current?.check(text)}
        sessionId={sessionId}
        readyText={readyText}
        onValueChange={setMessageText}
        controlRef={inputControlRef}
      />
      <NextMessagePanel
        sessionId={sessionId}
        messageText={messageText}
        onUse={(text) => inputControlRef.current?.setText(text)}
        onReady={setReadyText}
        controlRef={nextMessageRef}
      />
    </div>
  );
}

export default App;
