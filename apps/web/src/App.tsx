import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { api } from "./api/client";
import type { SavedWord, ThreadEntry, ThreadEntryIn } from "./api/client";
import type { DisplayMessage } from "./components/ChatMessage";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import type { ChatInputControl } from "./components/ChatInput";
import { NextMessagePanel } from "./components/NextMessagePanel";
import type { NextMessageControl } from "./components/NextMessagePanel";
import type { LiveCoachReply } from "./components/MessageThread";
import { ThreadOverlay } from "./components/ThreadOverlay";
import { SegmentedControl } from "./components/SegmentedControl";
import { LanguageMenu } from "./components/LanguageMenu";
import { WordListPanel } from "./components/WordListPanel";
import { LANGUAGES, apiLanguage, htmlLang, setReadingSettings, useReadingSettings } from "./lib/language";
import type { LanguageCode, ReadingSettings } from "./lib/language";
import { getSessionId } from "./lib/session";
import { useAppUpdate } from "./lib/useAppUpdate";
import { useChatAutoScroll } from "./lib/useChatAutoScroll";
import { useKeyboardUp } from "./lib/visibleViewport";
import { CoachSpeaker, isReadAloudOn, setReadAloud, speakPartner, stopSpeech, subscribeReadAloud } from "./voice/speech";
import "./App.css";

let nextLocalId = -1;

// The backend's id for a message (a freshly sent user message only learns
// it from the turn's "done" event - see handleSend).
function serverIdOf(m: DisplayMessage): number | null {
  if (m.serverId !== undefined) return m.serverId;
  return m.id > 0 ? m.id : null;
}

// The partner's voice, per language (Spanish keeps its original key).
const voiceStorageKey = (lang: LanguageCode) => (lang === "es" ? "wordbyword.voice" : `wordbyword.voice.${lang}`);

function App({ lang, onLanguageChange }: { lang: LanguageCode; onLanguageChange: (code: LanguageCode) => void }) {
  const config = LANGUAGES[lang];
  const readingSettings = useReadingSettings();
  const VOICE_STORAGE_KEY = voiceStorageKey(lang);
  const [sessionId] = useState(() => getSessionId(lang));
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
  // The partner's voices (ElevenLabs, from the server) - id + name.
  const [voices, setVoices] = useState<{ id: string; name: string }[]>([]);
  const readAloud = useSyncExternalStore(subscribeReadAloud, isReadAloudOn);
  // Which messages have their side conversation expanded (by display id).
  // The message whose coach conversation is open (as an overlay), if any.
  const [threadFor, setThreadFor] = useState<number | null>(null);
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

  // The saved word list - see ChatMessage's add-word flow and the ☰ menu's
  // Word list. `savedWords` (lowercased) is what the chat bubbles check.
  const [savedList, setSavedList] = useState<SavedWord[]>([]);
  const savedWords = useMemo(() => new Set(savedList.map((w) => w.word.toLowerCase())), [savedList]);
  const [wordListOpen, setWordListOpen] = useState(false);
  // Debug view of the vocabulary search next to each reply. Deliberately NOT
  // remembered: it's off every time the app opens.
  const [debug, setDebug] = useState(false);
  useEffect(() => {
    api
      .listSavedWords(sessionId)
      .then(setSavedList)
      .catch(() => {});
  }, [sessionId]);
  const saveWord = (entry: { word: string; gloss: string; grammar: string; sentence: string; messageId?: number }) => {
    setSavedList((prev) =>
      prev.some((w) => w.word.toLowerCase() === entry.word.toLowerCase())
        ? prev
        : [
            {
              id: nextLocalId--,
              word: entry.word,
              gloss: entry.gloss,
              grammar: entry.grammar,
              sentence: entry.sentence,
              message_id: entry.messageId ?? null,
              created_at: new Date().toISOString(),
            },
            ...prev,
          ],
    );
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
    setSavedList((prev) => prev.filter((w) => w.word.toLowerCase() !== word.toLowerCase()));
    api.removeSavedWord(sessionId, word).catch(() => {});
  };
  // Adding a word by hand: look up its meaning first (best effort - a word
  // with no meaning is still saved), then save it like a tapped one.
  const addWordByHand = async (word: string) => {
    if (savedWords.has(word.toLowerCase())) return;
    let gloss = "";
    try {
      gloss = (await api.translateText({ text: word, source_lang: apiLanguage(lang), target_lang: "en" })).translation ?? "";
    } catch {
      // no meaning available right now
    }
    saveWord({ word, gloss, grammar: "", sentence: "" });
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
      .listVoices(lang)
      .then((res) => {
        setVoices(res.voices);
        // A voice saved before (or from the old provider) that isn't
        // offered any more falls back to the default.
        setVoice((current) => (current && res.voices.some((v) => v.id === current) ? current : res.default));
      })
      .catch(() => {});
  }, [lang]);

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
          vocabDebug: m.vocab_debug,
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
        setThreadFor(null);
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
              vocabDebug: res.vocab_debug,
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
        } else if (event.type === "error") {
          throw new Error(event.data.message);
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setLiveCoach((prev) => (prev ? { ...prev, error: message } : prev));
    }
  };

  // A word's/phrase's "?": open that message's coach and ask it to explain
  // the text there (one coach answer at a time - while one is still coming,
  // it just opens).
  const explainIn = (m: DisplayMessage, text: string) => {
    setThreadFor(m.id);
    if (!liveCoach && serverIdOf(m) !== null) void askCoachAbout(m, `Explain "${text}" here`);
  };

  // The newest reply's popups open upward (see ChatMessage's `latest`).
  const latestAssistantId = [...messages].reverse().find((m) => m.role === "assistant")?.id;
  // Keyboard up: the bottom bars sit on it, not on the screen's rounded
  // corners (see App.css --bottom-round).
  const keyboardUp = useKeyboardUp();
  const threadMessage = threadFor === null ? undefined : messages.find((m) => m.id === threadFor);
  return (
    <div className={`app${keyboardUp ? " app--keyboard" : ""}`}>
      <header className="app__header">
        <div className="app__brand">
          <h1>wordbyword</h1>
          <LanguageMenu current={lang} onSelect={onLanguageChange} />
        </div>
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
              {config.readings && (
                <ReadingSettingsPicker lang={lang} settings={readingSettings} />
              )}
              <button
                type="button"
                className="app__menu-item"
                onClick={() => {
                  setWordListOpen(true);
                  setMenuOpen(false);
                }}
              >
                Word list ({savedList.length})
              </button>
              <div className="app__voice-picker">
                Debug:{" "}
                <SegmentedControl
                  value={debug ? "on" : "off"}
                  onChange={(v) => setDebug(v === "on")}
                  options={[
                    { value: "on", label: "On" },
                    { value: "off", label: "Off" },
                  ]}
                />
              </div>
              {voices.length > 0 && voice && (
                <div className="app__voice-picker app__voice-picker--voices">
                  Partner's voice:
                  <SegmentedControl
                    value={voice}
                    onChange={handleVoiceChange}
                    options={voices.map((v) => ({ value: v.id, label: v.name }))}
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
          <p className="app__empty">
            Say hello to start a conversation in {config.name} - <span lang={lang}>{config.greeting}</span>
          </p>
        )}
        {messages.map((m) => {
          const live = liveCoach && liveCoach.messageId === serverIdOf(m) ? liveCoach : null;
          const latest = m.role === "assistant" && m.id === latestAssistantId && streamingText === null;
          return (
            <div key={m.id} data-message-id={m.id}>
              <ChatMessage
                message={m}
                voice={voice ?? undefined}
                threadOpen={threadFor === m.id}
                onExplain={(text) => explainIn(m, text)}
                onToggleThread={() => setThreadFor(m.id)}
                liveCoach={live}
                latest={latest}
                savedWords={savedWords}
                onSaveWord={saveWord}
                onRemoveWord={removeWord}
                debug={debug}
              />
            </div>
          );
        })}
        {streamingText !== null && (
          <div className="chat-message chat-message--assistant">
            <div className="chat-message__column">
              <div className="chat-message__bubble" lang={htmlLang(lang)}>
                {streamingText || "…"}
              </div>
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
      {wordListOpen && (
        <WordListPanel
          words={savedList}
          onAdd={addWordByHand}
          onRemove={removeWord}
          onClose={() => setWordListOpen(false)}
        />
      )}
      {threadMessage && (
        <ThreadOverlay
          key={threadMessage.id}
          messageId={threadMessage.id}
          message={threadMessage}
          live={liveCoach && liveCoach.messageId === serverIdOf(threadMessage) ? liveCoach : null}
          sessionId={sessionId}
          canAsk={serverIdOf(threadMessage) !== null && !liveCoach}
          onAsk={(question) => void askCoachAbout(threadMessage, question)}
          onClose={() => setThreadFor(null)}
          onExplain={(text) => explainIn(threadMessage, text)}
          voice={voice ?? undefined}
          savedWords={savedWords}
          onSaveWord={saveWord}
          onRemoveWord={removeWord}
        />
      )}
    </div>
  );
}

// How Japanese, Korean and Chinese are shown (see ReadingText): just the
// romanization, the script with readings over it, or the script alone -
// plus, for Japanese, which readings (furigana, romaji) and whether kanji
// show as kanji; for Mandarin, Simplified or Traditional characters.
function ReadingSettingsPicker({ lang, settings }: { lang: LanguageCode; settings: ReadingSettings }) {
  const readings = LANGUAGES[lang].readings!;
  const set = (patch: Partial<ReadingSettings>) => setReadingSettings(patch, lang);
  const toggle = (key: "furigana" | "romaji", label: string) => (
    <button
      type="button"
      aria-pressed={settings[key]}
      className={`segmented-control__option${settings[key] ? " segmented-control__option--active" : ""}`}
      onClick={() => set({ [key]: !settings[key] })}
    >
      {label}
    </button>
  );
  return (
    <>
      <div className="app__voice-picker app__voice-picker--block">
        Show {LANGUAGES[lang].name} as:
        <SegmentedControl
          value={settings.mode}
          onChange={(mode) => set({ mode })}
          options={[
            { value: "roman", label: readings.roman },
            { value: "both", label: "Both" },
            { value: "script", label: readings.script },
          ]}
        />
      </div>
      {lang === "ja" && settings.mode === "both" && (
        <div className="app__voice-picker app__voice-picker--block">
          Reading aids:
          <div className="segmented-control">
            {toggle("furigana", "Furigana")}
            {toggle("romaji", "Romaji")}
          </div>
        </div>
      )}
      {lang === "ja" && settings.mode !== "roman" && (
        <div className="app__voice-picker app__voice-picker--block">
          Kanji:
          <SegmentedControl
            value={settings.kanji ? "kanji" : "kana"}
            onChange={(v) => set({ kanji: v === "kanji" })}
            options={[
              { value: "kanji", label: "漢字 Kanji" },
              { value: "kana", label: "かな Kana only" },
            ]}
          />
        </div>
      )}
      {lang === "zh" && (
        <div className="app__voice-picker app__voice-picker--block">
          Characters:
          <SegmentedControl
            value={settings.chineseScript}
            onChange={(chineseScript) => set({ chineseScript })}
            options={[
              { value: "simplified", label: "简体 Simplified" },
              { value: "traditional", label: "繁體 Traditional" },
            ]}
          />
        </div>
      )}
    </>
  );
}

export default App;
