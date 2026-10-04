import { CandidateCycler } from "./CandidateCycler";
import "./TranslatePopover.css";

export interface PopoverCandidate {
  translation: string;
  description?: string;
}

interface TranslatePopoverProps {
  candidates: PopoverCandidate[];
  onSelect?: (translation: string) => void;
  // "down" (default) suits a word inside a chat bubble, which has room
  // below it. "up" is for anything pinned near the bottom of the page
  // (the input box's draft-translate button) - opening downward there
  // would render off the bottom of the viewport.
  direction?: "down" | "up";
  // The word being glossed, shown as "sourceWord → translation" instead of
  // just the bare translation.
  sourceWord?: string;
  // Non-empty only when the hovered token is part of a multi-word group
  // (see TokenAnnotation.literal) - a short word-by-word breakdown of how
  // the group's individual words combine, shown as a small secondary line.
  literal?: string;
}

export function TranslatePopover({ candidates, onSelect, direction = "down", sourceWord, literal }: TranslatePopoverProps) {
  if (candidates.length === 0) return null;

  return (
    <div className={`translate-popover translate-popover--${direction}`} role="tooltip">
      <CandidateCycler candidates={candidates} clickable={!!onSelect} onSelect={onSelect} sourceWord={sourceWord} />
      {literal && <div className="translate-popover__literal">{literal}</div>}
    </div>
  );
}
