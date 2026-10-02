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
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);

  // The chat feed (App.css's .app__chat) scrolls and stacks many sibling
  // message bubbles, each its own stacking context - an absolutely
  // positioned popover nested inside one bubble gets clipped by the feed's
  // overflow the moment it extends past it, and can render behind a later
  // sibling bubble regardless of z-index (that only wins within the same
  // stacking context). Rendering it into a portal at the document root,
  // positioned from the word's actual screen coordinates, sidesteps both.
  useLayoutEffect(() => {
    if (!hovering || !anchorRef.current) {
      setCoords(null);
      return;
    }
    const rect = anchorRef.current.getBoundingClientRect();
    setCoords({ top: rect.bottom, left: rect.left + rect.width / 2 });
  }, [hovering]);

  // Scrolling the feed while hovering would leave a portal-rendered popover
  // stranded at its old coordinates - simplest fix is to just close it.
  useLayoutEffect(() => {
    if (!hovering) return;
    const close = () => setHovering(false);
    window.addEventListener("scroll", close, true);
    return () => window.removeEventListener("scroll", close, true);
  }, [hovering]);

  if (!gloss) {
    // Punctuation or a token we don't track in the word bank - render plain.
    return <span>{surface}</span>;
  }

  return (
    <span
      ref={anchorRef}
      className={`word-token${isNew ? " word-token--new" : ""}`}
      onMouseEnter={() => {
        setHovering(true);
        onHover?.();
      }}
      onMouseLeave={() => setHovering(false)}
    >
      {surface}
      {hovering && coords && (
        <FloatingPopover anchor={coords} direction="down">
          <TranslatePopover candidates={[{ translation: gloss, description: note || undefined }]} />
        </FloatingPopover>
      )}
    </span>
  );
}
