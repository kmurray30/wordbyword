import { useLayoutEffect, useRef, useState } from "react";
import { FloatingPopover } from "./FloatingPopover";
import { TranslatePopover } from "./TranslatePopover";
import "./WordToken.css";

interface WordTokenProps {
  surface: string;
  gloss: string;
  note?: string;
  isNew: boolean;
  onHover?: () => void;
}

export function WordToken({ surface, gloss, note, isNew, onHover }: WordTokenProps) {
  const [hovering, setHovering] = useState(false);
  // Tapping (as opposed to hovering) pins the popover open - a tap has no
  // "leave" event to close it on the way a mouse does, so it needs its own
  // explicit toggle, same pattern as ChatMessage's translate-pin button.
  const [pinned, setPinned] = useState(false);
  const open = hovering || pinned;
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);
  // Whatever text field had focus right before this tap, if any - captured
  // in onPointerDown (before the browser's own "tapping somewhere else
  // blurs the current field" behavior takes it away), so a quick check of
  // a word's meaning mid-draft doesn't cost the user their keyboard. Not
  // done by preventing default on the touch/click events themselves - that
  // would also suppress the click this component itself depends on.
  const focusedBeforeRef = useRef<HTMLElement | null>(null);

  // The chat feed (App.css's .app__chat) scrolls and stacks many sibling
  // message bubbles, each its own stacking context - an absolutely
  // positioned popover nested inside one bubble gets clipped by the feed's
  // overflow the moment it extends past it, and can render behind a later
  // sibling bubble regardless of z-index (that only wins within the same
  // stacking context). Rendering it into a portal at the document root,
  // positioned from the word's actual screen coordinates, sidesteps both.
  useLayoutEffect(() => {
    if (!open || !anchorRef.current) {
      setCoords(null);
      return;
    }
    const rect = anchorRef.current.getBoundingClientRect();
    setCoords({ top: rect.bottom, left: rect.left + rect.width / 2 });
  }, [open]);

  // Scrolling the feed while open would leave a portal-rendered popover
  // stranded at its old coordinates - simplest fix is to just close it.
  useLayoutEffect(() => {
    if (!open) return;
    const close = () => {
      setHovering(false);
      setPinned(false);
    };
    window.addEventListener("scroll", close, true);
    return () => window.removeEventListener("scroll", close, true);
  }, [open]);

  if (!gloss) {
    // Punctuation or a token we don't track in the word bank - render plain.
    return <span>{surface}</span>;
  }

  return (
    <span
      ref={anchorRef}
      className={`word-token${isNew ? " word-token--new" : ""}${open ? " word-token--active" : ""}`}
      onMouseEnter={() => {
        setHovering(true);
        onHover?.();
      }}
      onMouseLeave={() => setHovering(false)}
      onPointerDown={() => {
        const active = document.activeElement;
        focusedBeforeRef.current =
          active instanceof HTMLTextAreaElement || active instanceof HTMLInputElement ? active : null;
      }}
      onClick={(e) => {
        // Tapping on mobile never fires onMouseEnter at all, and even on
        // desktop a click explicitly pinning it open (rather than just
        // relying on hover) matches ChatMessage's own pin-to-keep-open
        // pattern for its translation rows.
        e.stopPropagation();
        setPinned((p) => !p);
        onHover?.();
        focusedBeforeRef.current?.focus();
      }}
    >
      {surface}
      {open && coords && (
        <FloatingPopover anchor={coords} direction="down">
          <TranslatePopover candidates={[{ translation: gloss, description: note || undefined }]} />
        </FloatingPopover>
      )}
    </span>
  );
}
