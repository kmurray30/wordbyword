import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { api } from "../api/client";
import type { ThreadEntry, ThreadEntryIn } from "../api/client";
import { TalkButton } from "./TalkButton";
import {
  SentenceSpeaker,
  coachMeaningSpeech,
  coachOptionSpeech,
  coachSuggestionSpeech,
  speakCoach,
  stopSpeech,
  unlockSpeech,
} from "../voice/speech";
import { useDictation } from "../voice/useDictation";
import "./NextMessagePanel.css";

// "Your next message": a little side conversation about what you're about
// to say, exactly like the thread under one of their messages but for the
// message you haven't sent yet. You talk it through in its own box (type
// or 🎤), the coach answers here with one suggestion you can "Use", the
// tutor answers questions about your reply here too, and when you send,
// this whole log becomes the thread of the message you sent. Questions
// about THEIR message are answered under that message instead - a short
// note here says so.

type Verdict = "clean" | "minor" | "fix";
type Force = "attempt" | "ask";
type Status = "streaming" | "done" | "error";

export type DraftEntry =
  | {
      id: number;
      kind: "check";
      // Where the text came from: talked through here, or the message box
      // (💡 / finishing a dictation there).
      source: "brainstorm" | "final";
      text: string;
      verdict: Verdict | null;
      meaning: string;
      suggestion: string;
      english: string;
      status: Status;
      used: boolean;
    }
  | { id: number; kind: "question"; text: string }
  | { id: number; kind: "answer"; text: string; status: Status }
  | { id: number; kind: "note"; text: string };

// What happens to a question the tutor answers under one of THEIR messages
// (App renders that thread).
export type TutorEvent =
  | { type: "start"; messageId: number; question: string }
  | { type: "chunk"; delta: string }
  | { type: "done"; answer: string; entries: ThreadEntry[] }
  | { type: "error"; message: string };

export interface NextMessageControl {
  // Check the text in the message box (it IS the message, so always an
  // attempt - never re-routed to the tutor).
  check: (text: string) => void;
  // Talk-it-through text, as if typed in this panel's own box.
  submit: (text: string, force?: Force) => void;
  // On send: this log, as the sent message's thread; the log is cleared.
  takeThread: () => ThreadEntryIn[];
  clear: () => void;
}

let nextId = 1;
const storageKey = (sessionId: string) => `wordbyword.nextMessage.${sessionId}`;

function loadEntries(sessionId: string): DraftEntry[] {
  try {
    const raw = localStorage.getItem(storageKey(sessionId));
    const parsed = raw ? (JSON.parse(raw) as DraftEntry[]) : [];
    const done = Array.isArray(parsed) ? parsed.filter((e) => !("status" in e) || e.status !== "streaming") : [];
    nextId = Math.max(nextId, ...done.map((e) => e.id + 1), 1);
    return done;
  } catch {
    return [];
  }
}

// The discussion so far, as context for the coach and the tutor.
function draftTurns(entries: DraftEntry[]): { role: string; text: string }[] {
  const turns: { role: string; text: string }[] = [];
  for (const e of entries) {
    if (e.kind === "check") {
      turns.push({ role: "learner", text: e.text });
      if (e.status === "done" && e.verdict) {
        const parts = [e.meaning && `They mean: ${e.meaning}`];
        if (e.suggestion) parts.push(`Suggested: ${e.suggestion}${e.english ? ` (${e.english})` : ""}`);
        if (e.verdict === "clean") parts.push("Looks good as is.");
        if (e.used) parts.push("They chose this.");
        turns.push({ role: "coach", text: parts.filter(Boolean).join(" ") });
      }
    } else if (e.kind === "question") {
      turns.push({ role: "learner", text: e.text });
    } else if (e.kind === "answer" && e.status === "done") {
      turns.push({ role: "coach", text: e.text });
    }
  }
  return turns;
}

function toThread(entries: DraftEntry[]): ThreadEntryIn[] {
  const out: ThreadEntryIn[] = [];
  for (const e of entries) {
    if (e.kind === "check" && e.status === "done" && e.verdict) {
      out.push({
        role: "coach",
        kind: "check",
        text: e.text,
        data: JSON.stringify({
          verdict: e.verdict,
          meaning: e.meaning,
          suggestion: e.verdict === "minor" ? e.suggestion : "",
          options: e.verdict === "fix" && e.suggestion ? [{ formality: "neutral", spanish: e.suggestion, english: e.english }] : [],
          picked: e.used && e.verdict === "fix" ? 0 : null,
          source: e.source,
        }),
      });
    } else if (e.kind === "question") {
      out.push({ role: "learner", kind: "question", text: e.text, data: "" });
    } else if (e.kind === "answer" && e.status === "done") {
      out.push({ role: "tutor", kind: "answer", text: e.text, data: "" });
    }
  }
  return out;
}

export function NextMessagePanel({
  sessionId,
  activeThreadId,
  onTutorEvent,
  onUse,
  onReady,
  controlRef,
}: {
  sessionId: string;
  // The thread open under one of their messages, if any - a follow-up
  // question continues it.
  activeThreadId: number | null;
  onTutorEvent: (event: TutorEvent) => void;
  // "Use this": put the text in the message box.
  onUse: (text: string) => void;
  // This exact text is approved to send (Send turns green).
  onReady: (text: string) => void;
  controlRef?: RefObject<NextMessageControl | null>;
}) {
  const [entries, setEntries] = useState<DraftEntry[]>(() => loadEntries(sessionId));
  const entriesRef = useRef(entries);
  const [collapsed, setCollapsed] = useState(false);
  const [value, setValue] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const propsRef = useRef({ activeThreadId, onTutorEvent, onUse, onReady });
  useEffect(() => {
    propsRef.current = { activeThreadId, onTutorEvent, onUse, onReady };
  });

  const update = (fn: (prev: DraftEntry[]) => DraftEntry[]) => {
    entriesRef.current = fn(entriesRef.current);
    setEntries(entriesRef.current);
  };
  const patch = (id: number, fields: Record<string, unknown>) =>
    update((prev) => prev.map((e) => (e.id === id ? ({ ...e, ...fields } as DraftEntry) : e)));

  useEffect(() => {
    try {
      localStorage.setItem(storageKey(sessionId), JSON.stringify(entries.filter((e) => e.kind !== "note")));
    } catch {
      // per-device convenience only
    }
  }, [entries, sessionId]);

  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [entries, collapsed]);

  function run(text: string, source: "brainstorm" | "final", force?: Force) {
    const clean = text.trim();
    if (!clean) return;
    stopSpeech();
    setCollapsed(false);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const turns = draftTurns(entriesRef.current);
    const checkId = nextId++;
    update((prev) => [
      ...prev,
      {
        id: checkId,
        kind: "check",
        source,
        text: clean,
        verdict: null,
        meaning: "",
        suggestion: "",
        english: "",
        status: "streaming",
        used: false,
      },
    ]);

    let verdict: Verdict | "ask" | null = null;
    let meaning = "";
    let suggestion = "";
    let english = "";
    // The question's destination once the tutor has picked it: this log
    // (answerId) or one of their messages (external).
    let answerId: number | null = null;
    let external = false;
    let answer = "";
    const speaker = new SentenceSpeaker();

    const fail = (message: string) => {
      if (verdict === "ask") {
        if (external) propsRef.current.onTutorEvent({ type: "error", message });
        else if (answerId !== null) patch(answerId, { status: "error", text: answer || message });
        else update((prev) => [...prev, { id: nextId++, kind: "answer", text: message, status: "error" }]);
      } else {
        patch(checkId, { status: "error" });
      }
    };

    (async () => {
      try {
        for await (const event of api.coachDraftStream(
          {
            text: clean,
            session_id: sessionId,
            thread_message_id: propsRef.current.activeThreadId,
            draft_thread: turns,
            force: source === "final" ? "attempt" : (force ?? null),
          },
          controller.signal,
        )) {
          if (controller.signal.aborted) return;
          switch (event.type) {
            case "verdict":
              verdict = event.data.verdict;
              if (verdict === "ask") update((prev) => prev.filter((e) => e.id !== checkId));
              else patch(checkId, { verdict });
              break;
            case "meaning_chunk":
              meaning += event.data.delta;
              patch(checkId, { meaning });
              break;
            case "meaning_complete":
              meaning = event.data.meaning;
              patch(checkId, { meaning });
              if (verdict && verdict !== "ask") speakCoach(coachMeaningSpeech(verdict, meaning));
              break;
            case "suggestion_chunk":
              suggestion += event.data.delta;
              patch(checkId, { suggestion });
              break;
            case "suggestion_complete":
              suggestion = event.data.suggestion;
              patch(checkId, { suggestion });
              speakCoach(coachSuggestionSpeech(suggestion));
              break;
            case "option_chunk":
              if (event.data.index !== 0) break;
              if (event.data.field === "spanish") suggestion += event.data.delta;
              else english += event.data.delta;
              patch(checkId, { suggestion, english });
              break;
            case "option_complete":
              if (event.data.index !== 0) break;
              if (event.data.field === "spanish") suggestion = event.data.text;
              else {
                english = event.data.text;
                speakCoach(coachOptionSpeech(0, suggestion, english));
              }
              patch(checkId, { suggestion, english });
              break;
            case "thread_anchor":
              if (event.data.message_id !== null) {
                external = true;
                update((prev) => [
                  ...prev,
                  { id: nextId++, kind: "note", text: `“${clean}” is about their message - answered under it ↑` },
                ]);
                propsRef.current.onTutorEvent({ type: "start", messageId: event.data.message_id, question: clean });
              } else {
                answerId = nextId++;
                const id = answerId;
                update((prev) => [
                  ...prev,
                  { id: nextId++, kind: "question", text: clean },
                  { id, kind: "answer", text: "", status: "streaming" },
                ]);
              }
              break;
            case "answer_chunk":
              answer += event.data.delta;
              speaker.push(event.data.delta);
              if (external) propsRef.current.onTutorEvent({ type: "chunk", delta: event.data.delta });
              else if (answerId !== null) patch(answerId, { text: answer });
              break;
            case "answer_complete":
              speaker.flush();
              answer = event.data.answer;
              if (external) propsRef.current.onTutorEvent({ type: "done", answer, entries: event.data.entries });
              else if (answerId !== null) patch(answerId, { text: answer, status: "done" });
              break;
            case "error":
              fail(event.data.message);
              return;
          }
        }
        if (controller.signal.aborted) return;
        if (verdict && verdict !== "ask") {
          patch(checkId, { status: "done" });
          // Text that came from the message box and needs no change is
          // ready to send as it is.
          if (verdict === "clean" && source === "final") propsRef.current.onReady(clean);
        }
      } catch (err) {
        if (!controller.signal.aborted) fail(err instanceof Error ? err.message : String(err));
      }
    })();
  }

  const takeSuggestion = (entry: Extract<DraftEntry, { kind: "check" }>) => {
    const text = entry.verdict === "clean" ? entry.text : entry.suggestion;
    if (!text) return;
    update((prev) => prev.map((e) => (e.kind === "check" ? { ...e, used: e.id === entry.id } : e)));
    propsRef.current.onUse(text);
    propsRef.current.onReady(text);
  };

  const askInstead = (entry: Extract<DraftEntry, { kind: "check" }>) => {
    update((prev) => prev.filter((e) => e.id !== entry.id));
    unlockSpeech();
    run(entry.text, "brainstorm", "ask");
  };

  // Talk-it-through dictation has its own box and mic.
  const dictation = useDictation({ sessionId, onText: (text) => setValue(text) });
  const submitBox = (text: string) => {
    if (!text.trim()) return;
    setValue("");
    run(text, "brainstorm");
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
      check: (text) => run(text, "final", "attempt"),
      submit: (text, force) => run(text, "brainstorm", force),
      takeThread: () => {
        abortRef.current?.abort();
        const thread = toThread(entriesRef.current);
        update(() => []);
        return thread;
      },
      clear: () => {
        abortRef.current?.abort();
        update(() => []);
      },
    };
  });

  useEffect(() => () => abortRef.current?.abort(), []);

  const count = entries.filter((e) => e.kind === "check" || e.kind === "question").length;

  return (
    <section className="next-message" aria-label="Your next message">
      {entries.length > 0 && (
        <button
          type="button"
          className="next-message__header"
          onClick={() => setCollapsed((c) => !c)}
          aria-expanded={!collapsed}
        >
          <span className="next-message__title">Your next message</span>
          <span className="next-message__meta">
            {count} {collapsed ? "▸" : "▾"}
          </span>
        </button>
      )}
      {entries.length > 0 && !collapsed && (
        <div className="next-message__log" ref={logRef}>
          {entries.map((e) => {
            if (e.kind === "note") {
              return (
                <div key={e.id} className="next-message__note">
                  {e.text}
                </div>
              );
            }
            if (e.kind === "question") {
              return (
                <div key={e.id} className="next-message__line next-message__line--learner">
                  <span className="next-message__who">You</span>
                  {e.text}
                </div>
              );
            }
            if (e.kind === "answer") {
              return (
                <div key={e.id} className="next-message__line">
                  <span className="next-message__who">Tutor</span>
                  {e.status === "error" ? <span className="next-message__error">{e.text}</span> : e.text || "…"}
                </div>
              );
            }
            return <CheckCard key={e.id} entry={e} onUse={() => takeSuggestion(e)} onAskInstead={() => askInstead(e)} />;
          })}
        </div>
      )}
      <div className="next-message__input">
        <textarea
          ref={textareaRef}
          className="next-message__textarea"
          rows={1}
          value={value}
          placeholder={
            dictation.active ? "Listening - tap the bubble when you're done" : "Work it out here - ask anything"
          }
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
        <TalkButton state={dictation.state} onClick={handleTalk} finishLabel="Finish - ask the coach" />
        <button
          type="button"
          className="next-message__ask"
          hidden={dictation.active}
          onClick={handleAsk}
          disabled={!value.trim()}
          aria-label="Ask the coach"
          title="Ask the coach"
        >
          ↑
        </button>
      </div>
      {dictation.error && !dictation.active && <div className="next-message__error">{dictation.error}</div>}
    </section>
  );
}

function CheckCard({
  entry,
  onUse,
  onAskInstead,
}: {
  entry: Extract<DraftEntry, { kind: "check" }>;
  onUse: () => void;
  onAskInstead: () => void;
}) {
  const { verdict, meaning, suggestion, english, status, used, source } = entry;
  const m = meaning.trim().replace(/[.?!]+$/, "");
  const usable = verdict === "clean" ? source === "brainstorm" : !!suggestion && (verdict === "fix" ? true : verdict === "minor");
  return (
    <div className="next-message__check">
      <div className="next-message__line next-message__line--learner">
        <span className="next-message__who">{source === "final" ? "Check" : "You"}</span>
        {entry.text}
      </div>
      <div className="next-message__line">
        <span className="next-message__who">Coach</span>
        {status === "error" ? (
          <span className="next-message__error">Couldn't reach the coach - try again</span>
        ) : verdict === null ? (
          "…"
        ) : verdict === "fix" ? (
          m ? `Did you mean “${m}”?` : "…"
        ) : verdict === "minor" ? (
          m ? `You mean “${m}” - one small fix:` : "…"
        ) : m ? (
          `Looks good: “${m}”`
        ) : (
          "Looks good"
        )}
      </div>
      {(verdict === "fix" || verdict === "minor") && (suggestion || status === "streaming") && (
        <div className={`next-message__suggestion${used ? " next-message__suggestion--used" : ""}`}>
          <span className="next-message__suggestion-es">{suggestion || "…"}</span>
          {verdict === "fix" && <span className="next-message__suggestion-en">{english || (status === "streaming" ? "…" : "")}</span>}
        </div>
      )}
      {status === "done" && (
        <div className="next-message__actions">
          {verdict === "clean" && source === "final" ? (
            <span className="next-message__ready">✓ Ready to send</span>
          ) : usable ? (
            <button type="button" className="next-message__use" onClick={onUse} disabled={used}>
              {used ? "✓ In your message" : "Use this"}
            </button>
          ) : null}
          {source === "brainstorm" && (
            <button type="button" className="next-message__link" onClick={onAskInstead}>
              That was a question
            </button>
          )}
        </div>
      )}
    </div>
  );
}
