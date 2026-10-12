import { useState } from "react";
import type { BrainstormLogTurn } from "../api/client";
import "./MessageNotes.css";

interface MessageNotesProps {
  notes: string[];
  // User bubbles only: the raw brainstorm log that led up to this reply.
  brainstorm: BrainstormLogTurn[];
  pending: boolean;
}

// The little dropdown under a bubble holding what was worked out in the
// brainstorm around it: folded notes (about the partner's message on the
// partner's bubble, about composing the reply on the learner's), plus -
// on the learner's bubble - the raw brainstorm log itself.
export function MessageNotes({ notes, brainstorm, pending }: MessageNotesProps) {
  const [open, setOpen] = useState(false);
  const [showLog, setShowLog] = useState(false);
  if (!pending && notes.length === 0 && brainstorm.length === 0) return null;

  const label = pending ? "Organizing notes…" : notes.length > 0 ? `Notes (${notes.length})` : "Brainstorm";
  return (
    <div className={`message-notes${open ? " message-notes--open" : ""}`}>
      <button type="button" className="message-notes__toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? "▾" : "▸"} {label}
      </button>
      {open && (
        <div className="message-notes__body">
          {notes.length > 0 && (
            <ul className="message-notes__list">
              {notes.map((note, i) => (
                <li key={i}>{note}</li>
              ))}
            </ul>
          )}
          {brainstorm.length > 0 && (
            <>
              <button
                type="button"
                className="message-notes__toggle message-notes__toggle--nested"
                onClick={() => setShowLog((s) => !s)}
                aria-expanded={showLog}
              >
                {showLog ? "▾" : "▸"} Brainstorm log ({brainstorm.length})
              </button>
              {showLog && (
                <div className="message-notes__log">
                  {brainstorm.map((turn) => (
                    <div key={turn.id} className={`message-notes__turn message-notes__turn--${turn.role}`}>
                      <span className="message-notes__who">{turn.role === "learner" ? "You" : "Coach"}</span>
                      <span>
                        {turn.text}
                        {turn.draft && <span className="message-notes__draft"> → {turn.draft}</span>}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
