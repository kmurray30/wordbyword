import type { ThreadEntry } from "../api/client";
import "./MessageThread.css";

// The side conversation under a chat message: the coach checks behind a
// message the learner sent, and questions about a message with the tutor's
// answers. The main chat stays just the phrases actually exchanged.

export interface LiveTutorAnswer {
  question: string;
  answer: string;
  error: string | null;
}

interface CheckData {
  verdict?: string;
  meaning?: string;
  suggestion?: string;
  options?: { formality?: string; spanish?: string; english?: string }[];
  picked?: number | null;
}

const FORMALITY_LABEL: Record<string, string> = { neutral: "Natural", casual: "Casual", formal: "Formal" };

function parseCheck(data: string): CheckData {
  try {
    const parsed = JSON.parse(data) as unknown;
    return parsed && typeof parsed === "object" ? (parsed as CheckData) : {};
  } catch {
    return {};
  }
}

function CheckEntry({ entry }: { entry: ThreadEntry }) {
  const check = parseCheck(entry.data);
  const meaning = check.meaning?.trim();
  return (
    <div className="message-thread__check">
      <div className="message-thread__line message-thread__line--learner">
        <span className="message-thread__who">You tried</span>
        {entry.text}
      </div>
      {meaning && (
        <div className="message-thread__line">
          <span className="message-thread__who">Coach</span>
          {check.verdict === "fix" ? `Did you mean "${meaning}"?` : check.verdict === "minor" ? `You mean "${meaning}" - one small fix:` : `Looks good: "${meaning}"`}
        </div>
      )}
      {check.verdict === "minor" && check.suggestion && (
        <div className="message-thread__option message-thread__option--picked">{check.suggestion}</div>
      )}
      {check.verdict === "fix" &&
        (check.options ?? []).map((opt, i) => (
          <div key={i} className={`message-thread__option${check.picked === i ? " message-thread__option--picked" : ""}`}>
            <span className="message-thread__option-label">
              {FORMALITY_LABEL[opt.formality ?? ""] ?? ""}
              {check.picked === i ? " ✓" : ""}
            </span>
            <span>{opt.spanish}</span>
            {opt.english && <span className="message-thread__option-english">{opt.english}</span>}
          </div>
        ))}
    </div>
  );
}

export function MessageThread({
  entries,
  live,
  onSayInstead,
}: {
  entries: ThreadEntry[];
  live: LiveTutorAnswer | null;
  // Re-routes a question the tutor answered back to the partner as an
  // attempt (the automatic routing guessed wrong).
  onSayInstead: (question: string) => void;
}) {
  const lastQuestion = live ? live.question : [...entries].reverse().find((e) => e.kind === "question")?.text;
  return (
    <div className="message-thread">
      {entries.map((entry) =>
        entry.kind === "check" ? (
          <CheckEntry key={entry.id} entry={entry} />
        ) : (
          <div
            key={entry.id}
            className={`message-thread__line${entry.kind === "question" ? " message-thread__line--learner" : ""}`}
          >
            <span className="message-thread__who">{entry.kind === "question" ? "You" : "Tutor"}</span>
            {entry.text}
          </div>
        ),
      )}
      {live && (
        <>
          <div className="message-thread__line message-thread__line--learner">
            <span className="message-thread__who">You</span>
            {live.question}
          </div>
          <div className="message-thread__line">
            <span className="message-thread__who">Tutor</span>
            {live.error ? <span className="message-thread__error">{live.error}</span> : live.answer || "…"}
          </div>
        </>
      )}
      {lastQuestion && (
        <button type="button" className="message-thread__say-instead" onClick={() => onSayInstead(lastQuestion)}>
          I meant to say that to my partner
        </button>
      )}
    </div>
  );
}
