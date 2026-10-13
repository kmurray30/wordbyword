import { useEffect } from "react";
import type { DisplayMessage } from "./ChatMessage";
import { MessageThread } from "./MessageThread";
import type { LiveCoachReply } from "./MessageThread";
import "./ThreadOverlay.css";

// A message's coach conversation, opened from its 💬: a sheet over most of
// the screen with the chat greyed out behind it. It lives inside .app, which
// always fills exactly the visible part of the screen (see App.css), so with
// the keyboard up the sheet simply gets shorter and its box stays in view.
// Tap outside, ✕ or Escape closes it.
export function ThreadOverlay({
  message,
  live,
  sessionId,
  canAsk,
  onAsk,
  onClose,
}: {
  message: DisplayMessage;
  live: LiveCoachReply | null;
  sessionId: string;
  canAsk: boolean;
  onAsk: (question: string) => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const theirs = message.role === "assistant";
  return (
    <div className="thread-overlay" role="dialog" aria-modal="true" aria-label="Talk to the coach about this message">
      <div className="thread-overlay__backdrop" onClick={onClose} />
      <div className={`thread-overlay__sheet thread-overlay__sheet--${theirs ? "theirs" : "yours"}`}>
        <div className="thread-overlay__head">
          <span className="thread-overlay__label">{theirs ? "💬 About their message" : "💬 About your message"}</span>
          <button type="button" className="thread-overlay__close" onClick={onClose} aria-label="Close" title="Close">
            ✕
          </button>
        </div>
        <div className="thread-overlay__quote">{message.text}</div>
        <MessageThread
          entries={message.thread ?? []}
          live={live}
          sessionId={sessionId}
          canAsk={canAsk}
          onAsk={onAsk}
        />
      </div>
    </div>
  );
}
