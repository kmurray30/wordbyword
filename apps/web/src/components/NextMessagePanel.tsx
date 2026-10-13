import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { api } from "../api/client";
import type { ThreadEntryIn } from "../api/client";
import { CoachText } from "./CoachText";
import { TalkButton } from "./TalkButton";
import { CoachSpeaker, stopSpeech, unlockSpeech } from "../voice/speech";
import { useDictation } from "../voice/useDictation";
import { useStickToBottom } from "../lib/useStickToBottom";
import "./NextMessagePanel.css";

// "Your next message": a conversation with the coach about what you're
// about to say. You talk it through in its own box (type or 🎤); the coach
// answers, explains, or suggests (suggestions come with "Use this"). When
// you send, the conversation becomes the thread of the message you sent.
// (Talking about a message already in the chat happens under that message,
// from its 💬 - see MessageThread.)

type Status = "streaming" | "done" | "error";

export type DraftTurn =
  | { id: number; role: "you"; text: string; source: "panel" | "box" }
  | { id: number; role: "coach"; text: string; status: Status };

export interface NextMessageControl {
  // The exact message in the message box: the coach looks it over.
  check: (text: string) => void;
  // On send: this conversation, as the sent message's thread; cleared.
  takeThread: () => ThreadEntryIn[];
  clear: () => void;
  // Something typed in the panel's own box, or a reply still coming in -
  // i.e. reloading the app right now would lose something.
  busy: () => boolean;
}

let nextId = 1;
const storageKey = (sessionId: string) => `wordbyword.nextMessage.v2.${sessionId}`;

// The earlier version of this panel kept its log under another key, as
// checks (meaning + one suggestion) and questions/answers. Carry it over
// once, as plain turns, so updating the app doesn't lose the conversation.
type OldEntry =
  | { kind: "check"; text: string; meaning?: string; suggestion?: string; english?: string; verdict?: string | null; status?: string }
  | { kind: "question"; text: string }
  | { kind: "answer"; text: string; status?: string }
  | { kind: "note"; text: string };

function migrateOldLog(sessionId: string): DraftTurn[] | null {
  const oldKey = `wordbyword.nextMessage.${sessionId}`;
  const raw = localStorage.getItem(oldKey);
  if (raw === null) return null;
  localStorage.removeItem(oldKey);
  const parsed = JSON.parse(raw) as OldEntry[];
  if (!Array.isArray(parsed)) return null;
  const turns: DraftTurn[] = [];
  let id = 1;
  for (const e of parsed) {
    if (!e?.text?.trim()) continue;
    if (e.kind === "check") {
      turns.push({ id: id++, role: "you", text: e.text, source: "panel" });
      if (e.status !== "done") continue;
      const parts = [
        e.meaning?.trim() ? `You mean: ${e.meaning.trim()}` : "",
        e.suggestion?.trim()
          ? `[[${e.suggestion.trim()} | ${e.english?.trim() ?? ""}]]`
          : e.verdict === "clean"
            ? "That works as it is."
            : "",
      ].filter(Boolean);
      if (parts.length) turns.push({ id: id++, role: "coach", text: parts.join(" "), status: "done" });
    } else if (e.kind === "question") {
      turns.push({ id: id++, role: "you", text: e.text, source: "panel" });
    } else if (e.kind === "answer" && e.status === "done") {
      turns.push({ id: id++, role: "coach", text: e.text, status: "done" });
    }
  }
  return turns;
}

function loadTurns(sessionId: string): DraftTurn[] {
  try {
    if (localStorage.getItem(storageKey(sessionId)) === null) {
      const migrated = migrateOldLog(sessionId);
      if (migrated) {
        localStorage.setItem(storageKey(sessionId), JSON.stringify(migrated));
      }
    }
    const raw = localStorage.getItem(storageKey(sessionId));
    const parsed = raw ? (JSON.parse(raw) as DraftTurn[]) : [];
    const turns = Array.isArray(parsed) ? parsed.filter((t) => t.role === "you" || t.status === "done") : [];
    nextId = Math.max(nextId, ...turns.map((t) => t.id + 1), 1);
    return turns;
  } catch {
    return [];
  }
}

export function NextMessagePanel({
  sessionId,
  messageText,
  onUse,
  onReady,
  controlRef,
}: {
  sessionId: string;
  // What's in the message box right now ("✓ In message" only while it
  // still holds the suggestion that was used).
  messageText: string;
  onUse: (text: string) => void;
  // This exact text is fine to send (Send turns green).
  onReady: (text: string) => void;
  controlRef?: RefObject<NextMessageControl | null>;
}) {
  const [turns, setTurns] = useState<DraftTurn[]>(() => loadTurns(sessionId));
  const turnsRef = useRef(turns);
  const [collapsed, setCollapsed] = useState(false);
  const [value, setValue] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const propsRef = useRef({ onUse, onReady, messageText });
  useEffect(() => {
    propsRef.current = { onUse, onReady, messageText };
  });

  const update = (fn: (prev: DraftTurn[]) => DraftTurn[]) => {
    turnsRef.current = fn(turnsRef.current);
    setTurns(turnsRef.current);
  };
  const patch = (id: number, fields: Partial<Extract<DraftTurn, { role: "coach" }>>) =>
    update((prev) => prev.map((t) => (t.id === id && t.role === "coach" ? { ...t, ...fields } : t)));

  useEffect(() => {
    try {
      localStorage.setItem(storageKey(sessionId), JSON.stringify(turns));
    } catch {
      // per-device convenience only
    }
  }, [turns, sessionId]);

  useStickToBottom(logRef, [turns, collapsed]);

  // The box grows with its text (typed or dictated), up to its max height.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + 2}px`;
  }, [value]);

  function ask(text: string, source: "panel" | "box") {
    const clean = text.trim();
    if (!clean) return;
    stopSpeech();
    setCollapsed(false);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const history = turnsRef.current
      .filter((t) => t.role === "you" || t.status === "done")
      .map((t) => ({ role: t.role === "you" ? "learner" : "coach", text: t.text }));
    const youId = nextId++;
    update((prev) => [...prev, { id: youId, role: "you", text: clean, source }]);

    let replyId: number | null = null;
    let reply = "";
    let ready = false;
    const speaker = new CoachSpeaker();

    const fail = (message: string) => {
      if (replyId !== null) patch(replyId, { status: "error", text: reply || message });
      else update((prev) => [...prev, { id: nextId++, role: "coach", text: message, status: "error" }]);
    };

    (async () => {
      try {
        for await (const event of api.coachDraftStream(
          {
            text: clean,
            session_id: sessionId,
            thread_message_id: null,
            // What's in the message box, so "is that right?" can be about it.
            message_draft: propsRef.current.messageText,
            draft_thread: history,
            check: source === "box",
          },
          controller.signal,
        )) {
          if (controller.signal.aborted) return;
          switch (event.type) {
            case "about": {
              replyId = nextId++;
              const id = replyId;
              update((prev) => [...prev, { id, role: "coach", text: "", status: "streaming" }]);
              break;
            }
            case "ready":
              ready = event.data.ready;
              break;
            case "reply_chunk":
              reply += event.data.delta;
              speaker.push(event.data.delta);
              if (replyId !== null) patch(replyId, { text: reply });
              break;
            case "reply_complete":
              reply = event.data.reply;
              speaker.flush();
              if (replyId !== null) patch(replyId, { text: reply, status: "done" });
              // The message box's text is already fine as written.
              if (ready && source === "box") propsRef.current.onReady(clean);
              break;
            case "error":
              fail(event.data.message);
              return;
          }
        }
      } catch (err) {
        if (!controller.signal.aborted) fail(err instanceof Error ? err.message : String(err));
      }
    })();
  }

  const handleUse = (spanish: string) => {
    propsRef.current.onUse(spanish);
    propsRef.current.onReady(spanish);
  };

  // Talking it through has its own box and mic.
  const dictation = useDictation({ sessionId, onText: (text) => setValue(text) });
  const submitBox = (text: string) => {
    if (!text.trim()) return;
    setValue("");
    ask(text, "panel");
  };
  const handleTalk = () => {
    unlockSpeech();
    if (dictation.active) {
      void dictation.done().then(submitBox);
      return;
    }
    textareaRef.current?.blur();
    void dictation.start(value);
  };
  const handleAsk = () => {
    unlockSpeech();
    if (dictation.active) {
      void dictation.done().then(submitBox);
      return;
    }
    submitBox(value);
  };

  useEffect(() => {
    if (!controlRef) return;
    controlRef.current = {
      check: (text) => ask(text, "box"),
      takeThread: () => {
        abortRef.current?.abort();
        const used = messageText.trim();
        const thread: ThreadEntryIn[] = turnsRef.current
          .filter((t) => t.role === "you" || t.status === "done")
          .map((t) =>
            t.role === "you"
              ? { role: "learner", kind: "question", text: t.text, data: "" }
              : { role: "coach", kind: "answer", text: t.text, data: JSON.stringify({ used }) },
          );
        update(() => []);
        return thread;
      },
      clear: () => {
        abortRef.current?.abort();
        update(() => []);
      },
      busy: () =>
        value.trim() !== "" || dictation.active || turnsRef.current.some((t) => t.role === "coach" && t.status === "streaming"),
    };
  });

  useEffect(() => () => abortRef.current?.abort(), []);

  const count = turns.filter((t) => t.role === "you").length;

  return (
    <section className="next-message" aria-label="Your next message">
      {turns.length > 0 && (
        <button
          type="button"
          className={`next-message__header${collapsed ? " next-message__header--collapsed" : ""}`}
          onClick={() => setCollapsed((c) => !c)}
          aria-expanded={!collapsed}
        >
          <span className="next-message__chevron" aria-hidden="true">
            ▾
          </span>
          <span className="next-message__title">💬 Your next message</span>
          <span className="next-message__count">{count}</span>
          <span className="next-message__toggle-label">{collapsed ? "Show" : "Hide"}</span>
        </button>
      )}
      {turns.length > 0 && !collapsed && (
        <div className="next-message__log" ref={logRef}>
          {turns.map((t) =>
            t.role === "you" ? (
              <div key={t.id} className="talk-line talk-line--you">
                <span className="talk-line__who">You</span>
                <span className="talk-line__text">{t.text}</span>
              </div>
            ) : (
              <div key={t.id} className="talk-line talk-line--coach" data-status={t.status}>
                <span className="talk-line__who">Coach</span>
                {t.status === "error" ? (
                  <span className="talk-line__error">Couldn't reach the coach - try again</span>
                ) : (
                  <CoachText text={t.text} streaming={t.status === "streaming"} onUse={handleUse} usedText={messageText} />
                )}
              </div>
            ),
          )}
        </div>
      )}
      <div className="next-message__input">
        <textarea
          ref={textareaRef}
          className="next-message__textarea"
          rows={1}
          value={value}
          placeholder={dictation.active ? "Listening - tap the bubble when you're done" : "Talk it through with your coach"}
          autoCorrect="off"
          spellCheck={false}
          enterKeyHint="send"
          onChange={(e) => {
            if (!dictation.active) setValue(e.target.value);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleAsk();
            }
          }}
        />
        <TalkButton state={dictation.state} onClick={handleTalk} finishLabel="Finish - send to your coach" />
        <button
          type="button"
          className="next-message__ask"
          hidden={dictation.active}
          onClick={handleAsk}
          disabled={!value.trim()}
          aria-label="Send to your coach"
          title="Send to your coach"
        >
          ↑
        </button>
      </div>
      {dictation.error && !dictation.active && <div className="talk-line__error">{dictation.error}</div>}
    </section>
  );
}
