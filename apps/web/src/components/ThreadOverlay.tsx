import { useEffect, useLayoutEffect, useRef } from "react";
import type { DisplayMessage } from "./ChatMessage";
import { MessageThread } from "./MessageThread";
import type { LiveCoachReply } from "./MessageThread";
import "./ThreadOverlay.css";

// A message's coach conversation, opened from its 💬: a sheet over the
// middle of the screen with the chat greyed out (but visible) around it. It
// lives inside .app, which always fills exactly the visible part of the
// screen (see App.css), so with the keyboard up the sheet simply gets
// shorter and its box stays in view. Tap outside, ✕ or Escape closes it.
//
// Opening, the message's bubble flies from its place in the chat to the top
// of the sheet, the sheet grows out of it, and the conversation unfolds
// downward; closing runs the same in reverse, landing the bubble back in
// the chat.

const DURATION = 320;
const EASE = "cubic-bezier(0.2, 0.8, 0.2, 1)";
const SHEET_RADIUS = 18;

type Rect = { left: number; top: number; width: number; height: number };

const nextFrames = (n: number) =>
  new Promise<void>((resolve) => {
    const step = (left: number) => (left <= 0 ? resolve() : requestAnimationFrame(() => step(left - 1)));
    step(n);
  });

function relative(r: DOMRect, to: DOMRect): Rect {
  return { left: r.left - to.left, top: r.top - to.top, width: r.width, height: r.height };
}

const px = (r: Rect) => ({ left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });

// The sheet clipped down to (the part of it under) the bubble.
function clipTo(bubble: DOMRect, sheet: DOMRect, radius: string): string {
  const top = Math.max(0, bubble.top - sheet.top);
  const left = Math.max(0, bubble.left - sheet.left);
  const right = Math.max(0, sheet.right - bubble.right);
  const bottom = Math.max(0, Math.min(sheet.height - top - 1, sheet.bottom - bubble.bottom));
  return `inset(${top}px ${right}px ${bottom}px ${left}px round ${radius})`;
}

export function ThreadOverlay({
  messageId,
  message,
  live,
  sessionId,
  canAsk,
  onAsk,
  onClose,
}: {
  // The message's display id (finds its bubble in the chat).
  messageId: number;
  message: DisplayMessage;
  live: LiveCoachReply | null;
  sessionId: string;
  canAsk: boolean;
  onAsk: (question: string) => void;
  onClose: () => void;
}) {
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const backdropRef = useRef<HTMLDivElement | null>(null);
  const sheetRef = useRef<HTMLDivElement | null>(null);
  const quoteRef = useRef<HTMLDivElement | null>(null);
  const flyerRef = useRef<HTMLDivElement | null>(null);
  const closingRef = useRef(false);

  const findBubble = () =>
    document.querySelector<HTMLElement>(`[data-message-id="${messageId}"] .chat-message__bubble`);
  const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

  // Fly the bubble between the chat and the sheet (forward = opening).
  const animate = (forward: boolean): Promise<void> | null => {
    const overlay = overlayRef.current;
    const sheet = sheetRef.current;
    const quote = quoteRef.current;
    const flyer = flyerRef.current;
    const thread = sheet?.querySelector<HTMLElement>(".message-thread");
    const bubble = findBubble();
    if (!overlay || !sheet || !quote || !flyer || !bubble || reducedMotion()) return null;
    const o = overlay.getBoundingClientRect();
    const b = bubble.getBoundingClientRect();
    const q = quote.getBoundingClientRect();
    const s = sheet.getBoundingClientRect();
    const bubbleRadius = getComputedStyle(bubble).borderRadius || "14px";
    const quoteRadius = getComputedStyle(quote).borderRadius || "12px";
    const atBubble = { ...px(relative(b, o)), borderRadius: bubbleRadius };
    const atQuote = { ...px(relative(q, o)), borderRadius: quoteRadius };
    const opts = { duration: DURATION, easing: EASE, fill: "both" as const };

    // A newer flight replaces any still running.
    for (const el of [flyer, sheet, backdropRef.current, thread]) el?.getAnimations().forEach((a) => a.cancel());
    bubble.style.visibility = "hidden";
    quote.style.visibility = "hidden";
    flyer.style.display = "block";
    const flight = flyer.animate(forward ? [atBubble, atQuote] : [atQuote, atBubble], opts);
    const closed = clipTo(b, s, `${bubbleRadius}`);
    const open = `inset(0px 0px 0px 0px round ${SHEET_RADIUS}px)`;
    sheet.animate({ clipPath: forward ? [closed, open] : [open, closed] }, opts);
    backdropRef.current?.animate({ opacity: forward ? [0, 1] : [1, 0] }, opts);
    if (thread) {
      const folded = { clipPath: "inset(0 0 100% 0)", opacity: 0 };
      const unfolded = { clipPath: "inset(0 0 0% 0)", opacity: 1 };
      thread.animate(forward ? [folded, unfolded] : [unfolded, folded], {
        ...opts,
        duration: forward ? DURATION : DURATION * 0.6,
        delay: forward ? DURATION * 0.25 : 0,
      });
    }
    // Landing: the real one (the quote, or the bubble back in the chat) is
    // shown first, underneath the flyer, and the flyer only goes a couple
    // of frames later - a just-unhidden element can take a frame to paint
    // (iOS especially), and swapping both at once flickers.
    return flight.finished.then(
      async () => {
        if (forward) quote.style.visibility = "";
        else bubble.style.visibility = "";
        await nextFrames(2);
        // Closing: the flyer stays over the bubble until the overlay goes.
        if (forward) flyer.style.display = "none";
      },
      () => {
        // Cancelled by a newer flight - that one tidies up.
      },
    );
  };

  useLayoutEffect(() => {
    void animate(true);
    return () => {
      // However it closes (or the chat is cleared), the bubble comes back.
      const bubble = findBubble();
      if (bubble) bubble.style.visibility = "";
    };
    // Opening plays once, for the message it opened on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const close = () => {
    if (closingRef.current) return;
    closingRef.current = true;
    const done = animate(false);
    if (done) void done.then(onClose, onClose);
    else onClose();
  };
  const closeRef = useRef(close);
  closeRef.current = close;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeRef.current();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const theirs = message.role === "assistant";
  const side = theirs ? "theirs" : "yours";
  return (
    <div
      className="thread-overlay"
      ref={overlayRef}
      role="dialog"
      aria-modal="true"
      aria-label="Talk to the coach about this message"
    >
      <div className="thread-overlay__backdrop" ref={backdropRef} onClick={close} />
      <div className={`thread-overlay__sheet thread-overlay__sheet--${side}`} ref={sheetRef}>
        <div className="thread-overlay__head">
          <span className="thread-overlay__label">{theirs ? "💬 About their message" : "💬 About your message"}</span>
          <button type="button" className="thread-overlay__close" onClick={close} aria-label="Close" title="Close">
            ✕
          </button>
        </div>
        <div className="thread-overlay__quote" ref={quoteRef}>
          {message.text}
        </div>
        <MessageThread
          entries={message.thread ?? []}
          live={live}
          sessionId={sessionId}
          canAsk={canAsk}
          onAsk={onAsk}
        />
      </div>
      {/* The bubble in flight between the chat and the sheet. */}
      <div className={`thread-overlay__flyer thread-overlay__flyer--${side}`} ref={flyerRef} aria-hidden="true">
        {message.text}
      </div>
    </div>
  );
}
