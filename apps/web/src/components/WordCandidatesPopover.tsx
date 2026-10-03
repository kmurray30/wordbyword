import { CandidateCycler } from "./CandidateCycler";
import "./WordCandidatesPopover.css";

export interface WordCandidate {
  translation: string;
  description?: string;
}

interface WordCandidatesPopoverProps {
  candidates: WordCandidate[];
  // False when there's nothing to replace (the span is already natural,
  // correct Spanish as typed) - just its in-context gloss. True when it
  // offers a Spanish replacement to swap in.
  clickable: boolean;
  onSelect: (translation: string) => void;
}

// The one popover widget for hovering/tapping a word or group in the chat
// input, and for highlighting a phrase. No Spanish/English toggle - the
// backend's LLM call already has full sentence context, so it picks the
// one sense that applies here rather than needing two independent
// readings reconciled in the UI (see DraftSpan in app/schemas.py). A span
// with multiple senses still steps through them via CandidateCycler.
export function WordCandidatesPopover({ candidates, clickable, onSelect }: WordCandidatesPopoverProps) {
  if (candidates.length === 0) return null;
  return (
    <div className="word-candidates-popover" role="tooltip">
      <CandidateCycler candidates={candidates} clickable={clickable} onSelect={onSelect} />
    </div>
  );
}
