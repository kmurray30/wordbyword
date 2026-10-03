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
      onMouseEnter={onOpen}
      onMouseLeave={onClose}
      onClick={(e) => {
        // Tapping on mobile never fires onMouseEnter at all, and even on
        // desktop a click explicitly pinning it open (rather than just
        // relying on hover) matches ChatMessage's own pin-to-keep-open
        // pattern for its translation rows.
        e.stopPropagation();
        onTogglePin();
      }}
    >
      {surface}
    </span>
  );
}
