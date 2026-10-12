import { useRef } from "react";
import type { ThreadEntry } from "../api/client";
import { CoachText } from "./CoachText";
import { useStickToBottom } from "../lib/useStickToBottom";
import "./MessageThread.css";

// The side conversation under a chat message: talk with the coach about
// their message, or - under your own message - how you worked it out. The
// main chat stays just the phrases actually exchanged.

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

export function MessageThread({ entries, live }: { entries: ThreadEntry[]; live: LiveCoachReply | null }) {
  const logRef = useRef<HTMLDivElement | null>(null);
  useStickToBottom(logRef, [entries.length, live?.reply, live?.error]);
  return (
    <div className="message-thread">
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
    </div>
  );
}
