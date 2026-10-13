import type { ReactNode } from "react";
import "./WordToken.css";

interface WordTokenProps {
  // The word as shown (its romanization, or kana for kanji, depending on
  // the reading settings - see ReadingText).
  surface: ReactNode;
  gloss: string;
  open: boolean;
  // Part of the phrase the open word belongs to (highlighted together).
  inPhrase?: boolean;
  // A little floating note right above the word ("add word?").
  badge?: ReactNode;
  // Asking "add word?" (green) or "remove?" (red).
  tone?: "add" | "remove" | null;
  onOpen: () => void;
  onClose: () => void;
  onTogglePin: () => void;
}

// Purely presentational - the actual popover is docked to the bottom of
// the whole message bubble (see ChatMessage), not floated next to this
// specific word, so ChatMessage owns which word (if any) is open across
// the whole message and just tells this one whether it's the one.
export function WordToken({ surface, gloss, open, inPhrase = false, badge, tone = null, onOpen, onClose, onTogglePin }: WordTokenProps) {
  if (!gloss) {
    // Punctuation or a token we don't track in the word bank - render plain.
    return <span>{surface}</span>;
  }

  return (
    <span
      className={`word-token${open ? " word-token--active" : ""}${inPhrase ? " word-token--phrase" : ""}${
        tone ? ` word-token--${tone}` : ""
      }`}
      // Hover previews only for a real mouse: a tap on iOS also fires an
      // emulated "hover" right before its click, which opened the popover
      // and then the click toggled it shut again (the first-tap flicker).
      onPointerEnter={(e) => {
        if (e.pointerType === "mouse") onOpen();
      }}
      onPointerLeave={(e) => {
        if (e.pointerType === "mouse") onClose();
      }}
      onClick={(e) => {
        // A click/tap pins it open; clicking it again closes it.
        e.stopPropagation();
        onTogglePin();
      }}
    >
      {surface}
      {badge}
    </span>
  );
}
