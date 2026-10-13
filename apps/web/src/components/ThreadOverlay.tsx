import { useEffect, useLayoutEffect, useRef } from "react";
import { ChatMessage } from "./ChatMessage";
import type { DisplayMessage } from "./ChatMessage";
import { MessageThread } from "./MessageThread";
import type { LiveCoachReply } from "./MessageThread";
import "./ThreadOverlay.css";

// A message's coach conversation, opened from its 💬: a sheet over the
// middle of the screen with the chat greyed out (but visible) around it. At
// its top, the message itself - the same bubble as in the chat, with all of
// its taps (word and phrase meanings, add word, 🌐, audio). It lives inside
// .app, which always fills exactly the visible part of the screen (see
// App.css), so with the keyboard up the sheet simply gets shorter and its
// box stays in view. Tap outside, ✕ or Escape closes it.
//
// Opening, a copy of the bubble flies from the chat to the top of the
// sheet while a plain plate (the sheet's background) grows out of it; once
// they land, the real sheet is shown under them and the conversation
// unfolds downward. Closing runs it in reverse and the bubble lands back in
// the chat. Only the copy and the plate move - plain boxes, no clipping of
// the real sheet - and each hand-off shows the real thing underneath first,
// removing the copy a couple of frames later, so nothing blinks.

const DURATION = 320;
const EASE = "cubic-bezier(0.2, 0.8, 0.2, 1)";

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

// A static copy of a bubble (without any open popup) to fly around.
function copyOf(bubble: HTMLElement): HTMLElement {
  const copy = bubble.cloneNode(true) as HTMLElement;
  copy.querySelectorAll(".docked-popover-wrap").forEach((el) => el.remove());
  copy.style.visibility = "visible";
  return copy;
}

export function ThreadOverlay({
  messageId,
  message,
  live,
  sessionId,
  canAsk,
  onAsk,
  onClose,
  voice,
  savedWords,
  onSaveWord,
  onRemoveWord,
  onExplain,
}: {
  // The message's display id (finds its bubble in the chat).
  messageId: number;
  message: DisplayMessage;
  live: LiveCoachReply | null;
  sessionId: string;
  canAsk: boolean;
  onAsk: (question: string) => void;
  onClose: () => void;
  voice?: string;
  savedWords?: Set<string>;
  onSaveWord?: Parameters<typeof ChatMessage>[0]["onSaveWord"];
  onRemoveWord?: (word: string) => void;
  // A word's/phrase's "?" in the bubble: ask the coach here.
  onExplain?: (text: string) => void;
}) {
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const backdropRef = useRef<HTMLDivElement | null>(null);
  const sheetRef = useRef<HTMLDivElement | null>(null);
  const quoteRef = useRef<HTMLDivElement | null>(null);
  const plateRef = useRef<HTMLDivElement | null>(null);
  const flyerRef = useRef<HTMLDivElement | null>(null);
  const closingRef = useRef(false);

  const chatBubble = () =>
    document.querySelector<HTMLElement>(`[data-message-id="${messageId}"] .chat-message__bubble`);
  const sheetBubble = () => quoteRef.current?.querySelector<HTMLElement>(".chat-message__bubble") ?? null;
  const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

  // Fly between the chat and the sheet (forward = opening). Resolves once
  // the real thing is showing where the flight landed.
  const animate = async (forward: boolean): Promise<boolean> => {
    const overlay = overlayRef.current;
    const sheet = sheetRef.current;
    const plate = plateRef.current;
    const flyer = flyerRef.current;
    const inChat = chatBubble();
    const inSheet = sheetBubble();
    const thread = sheet?.querySelector<HTMLElement>(".message-thread");
    if (!overlay || !sheet || !plate || !flyer || !inChat || !inSheet || reducedMotion()) return false;

    const o = overlay.getBoundingClientRect();
    const atChat = { ...px(relative(inChat.getBoundingClientRect(), o)), borderRadius: getComputedStyle(inChat).borderRadius };
    const atSheet = { ...px(relative(inSheet.getBoundingClientRect(), o)), borderRadius: getComputedStyle(inSheet).borderRadius };
    const sheetBox = { ...px(relative(sheet.getBoundingClientRect(), o)), borderRadius: getComputedStyle(sheet).borderRadius };
    const plateAtChat = { ...atChat, opacity: 0 };
    const plateAtSheet = { ...sheetBox, opacity: 1 };

    // Set up the copy where the flight starts, shown over the real one
    // there; the real one hides a couple of frames later.
    flyer.replaceChildren(copyOf(forward ? inChat : inSheet));
    flyer.style.display = "block";
    plate.style.display = "block";
    const opts = { duration: DURATION, easing: EASE, fill: "both" as const };
    const flight = flyer.animate(forward ? [atChat, atSheet] : [atSheet, atChat], opts);
    plate.animate(forward ? [plateAtChat, plateAtSheet] : [plateAtSheet, plateAtChat], opts);
    backdropRef.current?.animate({ opacity: forward ? [0, 1] : [1, 0] }, { ...opts, fill: forward ? "backwards" : "forwards" });
    if (forward) {
      inChat.style.visibility = "hidden";
    } else {
      await nextFrames(1);
      sheet.style.visibility = "hidden";
    }

    try {
      await flight.finished;
    } catch {
      return false; // cancelled
    }

    if (forward) {
      // Land: the real sheet appears under the copy and the plate, its
      // conversation unfolding; the copy and plate go a couple of frames on.
      sheet.style.visibility = "";
      thread?.animate(
        [
          { clipPath: "inset(0 0 100% 0)", opacity: 0 },
          { clipPath: "inset(0 0 0% 0)", opacity: 1 },
        ],
        { duration: DURATION * 0.8, easing: EASE, fill: "backwards" },
      );
      await nextFrames(2);
      flyer.style.display = "none";
      plate.style.display = "none";
    } else {
      // Back in the chat: the bubble shows under the copy, which goes with
      // the overlay.
      inChat.style.visibility = "";
      await nextFrames(2);
    }
    return true;
  };

  useLayoutEffect(() => {
    // Hidden until the flight lands (or right away without one).
    const sheet = sheetRef.current;
    if (sheet && !reducedMotion() && chatBubble()) sheet.style.visibility = "hidden";
    void animate(true).then((ran) => {
      if (!ran && sheetRef.current) sheetRef.current.style.visibility = "";
    });
    return () => {
      // However it closes (or the chat is cleared), the bubble comes back.
      const bubble = chatBubble();
      if (bubble) bubble.style.visibility = "";
    };
    // Opening plays once, for the message it opened on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const close = () => {
    if (closingRef.current) return;
    closingRef.current = true;
    void animate(false).finally(onClose);
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
          <ChatMessage
            message={message}
            variant="sheet"
            voice={voice}
            savedWords={savedWords}
            onSaveWord={onSaveWord}
            onRemoveWord={onRemoveWord}
            onExplain={onExplain}
          />
        </div>
        <MessageThread
          entries={message.thread ?? []}
          live={live}
          sessionId={sessionId}
          canAsk={canAsk}
          onAsk={onAsk}
        />
      </div>
      {/* What flies: the sheet's background, and a copy of the bubble. */}
      <div className={`thread-overlay__plate thread-overlay__plate--${side}`} ref={plateRef} aria-hidden="true" />
      <div className={`thread-overlay__flyer chat-message chat-message--${message.role}`} ref={flyerRef} aria-hidden="true" />
    </div>
  );
}
