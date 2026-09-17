import { useState } from "react";
import { TranslatePopover } from "./TranslatePopover";
import "./WordToken.css";

interface WordTokenProps {
  surface: string;
  gloss: string;
  isNew: boolean;
  onHover?: () => void;
}

export function WordToken({ surface, gloss, isNew, onHover }: WordTokenProps) {
  const [hovering, setHovering] = useState(false);

  if (!gloss) {
    // Punctuation or a token we don't track in the word bank - render plain.
    return <span>{surface}</span>;
  }

  return (
    <span
      className={`word-token${isNew ? " word-token--new" : ""}`}
      onMouseEnter={() => {
        setHovering(true);
        onHover?.();
      }}
      onMouseLeave={() => setHovering(false)}
    >
      {surface}
      {hovering && <TranslatePopover candidates={[{ translation: gloss }]} />}
    </span>
  );
}
