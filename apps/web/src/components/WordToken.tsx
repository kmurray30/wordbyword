import "./WordToken.css";

interface WordTokenProps {
  surface: string;
  gloss: string;
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
  onTogglePin: () => void;
}

// Purely presentational - the actual popover is docked to the bottom of
// the whole message bubble (see ChatMessage), not floated next to this
// specific word, so ChatMessage owns which word (if any) is open across
// the whole message and just tells this one whether it's the one.
export function WordToken({ surface, gloss, open, onOpen, onClose, onTogglePin }: WordTokenProps) {
  if (!gloss) {
    // Punctuation or a token we don't track in the word bank - render plain.
    return <span>{surface}</span>;
  }

  return (
    <span
      className={`word-token${open ? " word-token--active" : ""}`}
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
    </span>
  );
}
