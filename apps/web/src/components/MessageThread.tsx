import { useLayoutEffect, useRef, useState } from "react";
import type { ThreadEntry } from "../api/client";
import { CoachText } from "./CoachText";
import { useStickToBottom } from "../lib/useStickToBottom";
import { TalkButton } from "./TalkButton";
import { useDictation } from "../voice/useDictation";
import { unlockSpeech } from "../voice/speech";
import "./MessageThread.css";

// The side conversation about a chat message, opened from its 💬 (shown in
// ThreadOverlay): talk
// with the coach about that message - what it says, its words, how to
// answer it (or, under your own, how you worked it out) - in its own box
// with a mic. The main chat stays just the phrases actually exchanged.

export interface LiveCoachReply {
  question: string;
  reply: string;
  error: string | null;
}

interface CheckData {
  verdict?: string;
  meaning?: string;
  suggestion?: string;
  options?: { formality?: string; spanish?: string; english?: string }[];
  picked?: number | null;
}

function parseData(data: string): CheckData {
  try {
    const parsed = JSON.parse(data) as unknown;
    return parsed && typeof parsed === "object" ? (parsed as CheckData) : {};
  } catch {
    return {};
  }
}

// Threads saved before the coach became one conversation stored checks as
// a structured result; show them as a You line and a Coach line.
function OldCheck({ entry }: { entry: ThreadEntry }) {
  const check = parseData(entry.data);
  const meaning = check.meaning?.trim().replace(/[.?!]+$/, "");
  const suggestion =
    check.verdict === "minor" ? check.suggestion : check.verdict === "fix" ? check.options?.[check.picked ?? 0]?.spanish : "";
  const english = check.verdict === "fix" ? check.options?.[check.picked ?? 0]?.english : "";
  const text = meaning
    ? `${check.verdict === "clean" ? "Looks good" : "You mean"}: “${meaning}”${suggestion ? ` [[${suggestion} | ${english ?? ""}]]` : ""}`
    : "";
  return (
    <>
      <div className="talk-line talk-line--you">
        <span className="talk-line__who">You</span>
        <span>{entry.text}</span>
      </div>
      {text && (
        <div className="talk-line talk-line--coach">
          <span className="talk-line__who">Coach</span>
          <CoachText text={text} />
        </div>
      )}
    </>
  );
}

// This conversation's own box: type or talk, then send.
function ThreadInput({
  sessionId,
  canAsk,
  onAsk,
}: {
  sessionId: string;
  canAsk: boolean;
  onAsk: (question: string) => void;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const dictation = useDictation({ sessionId, onText: setValue });
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + 2}px`;
  }, [value]);

  const submit = (text: string) => {
    const question = text.trim();
    if (!question || !canAsk) return;
    setValue("");
    onAsk(question);
  };
  const handleSend = () => {
    unlockSpeech();
    if (dictation.active) {
      void dictation.done().then(submit);
      return;
    }
    submit(value);
  };
  const handleTalk = () => {
    unlockSpeech();
    if (dictation.active) {
      void dictation.done().then(submit);
      return;
    }
    textareaRef.current?.blur();
    void dictation.start(value);
  };

  return (
    <div className="message-thread__input">
      <textarea
        ref={textareaRef}
        className="message-thread__textarea"
        rows={1}
        value={value}
        placeholder={dictation.active ? "Listening - tap the bubble when you're done" : "Ask the coach…"}
        autoCorrect="off"
        spellCheck={false}
        enterKeyHint="send"
        onChange={(e) => {
          if (!dictation.active) setValue(e.target.value);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
          }
        }}
      />
      <TalkButton state={dictation.state} onClick={handleTalk} finishLabel="Finish - send to your coach" />
      <button
        type="button"
        className="message-thread__send"
        hidden={dictation.active}
        // Keeps the box focused (and the keyboard up) for the next question:
        // no focus change on a mouse press, and on touch the send happens
        // on touchend with the default (blur + synthetic click) cancelled.
        onMouseDown={(e) => e.preventDefault()}
        onTouchEnd={(e) => {
          e.preventDefault();
          if (value.trim() && canAsk) handleSend();
        }}
        onClick={handleSend}
        disabled={!value.trim() || !canAsk}
        aria-label="Send to your coach"
        title="Send to your coach"
      >
        ↑
      </button>
    </div>
  );
}

export function MessageThread({
  entries,
  live,
  sessionId,
  canAsk,
  onAsk,
}: {
  entries: ThreadEntry[];
  live: LiveCoachReply | null;
  sessionId: string;
  canAsk: boolean;
  onAsk: (question: string) => void;
}) {
  const logRef = useRef<HTMLDivElement | null>(null);
  useStickToBottom(logRef, [entries.length, live?.reply, live?.error]);
  const empty = entries.length === 0 && !live;
  return (
    <div className="message-thread">
      {!empty && (
      <div className="message-thread__log" ref={logRef}>
        {entries.map((entry) =>
          entry.kind === "check" ? (
            <OldCheck key={entry.id} entry={entry} />
          ) : entry.kind === "question" ? (
            <div key={entry.id} className="talk-line talk-line--you">
              <span className="talk-line__who">You</span>
              <span>{entry.text}</span>
            </div>
          ) : (
            <div key={entry.id} className="talk-line talk-line--coach">
              <span className="talk-line__who">Coach</span>
              <CoachText text={entry.text} />
            </div>
          ),
        )}
        {live && (
          <>
            <div className="talk-line talk-line--you">
              <span className="talk-line__who">You</span>
              <span>{live.question}</span>
            </div>
            <div className="talk-line talk-line--coach" data-status={live.error ? "error" : "streaming"}>
              <span className="talk-line__who">Coach</span>
              {live.error ? <span className="talk-line__error">{live.error}</span> : <CoachText text={live.reply} streaming />}
            </div>
          </>
        )}
      </div>
      )}
      <ThreadInput sessionId={sessionId} canAsk={canAsk} onAsk={onAsk} />
    </div>
  );
}
