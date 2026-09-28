import { useState } from "react";
import "./CandidateCycler.css";

export interface Candidate {
  translation: string;
  description?: string;
}

interface CandidateCyclerProps {
  candidates: Candidate[];
  clickable: boolean;
  onSelect?: (translation: string) => void;
  // The word being glossed, shown as "sourceWord → translation" instead of
  // just the bare translation - updates as the user cycles, so it always
  // reads as a complete "X means Y" statement, not just "Y" on its own.
  sourceWord?: string;
}

// One translation candidate at a time - source word, its translation, and a
// short sense description all inline on one compact line, with a small
// "N/M" indicator and a single "next" button (stacked above one another,
// off to the side) to step through alternates (e.g. "tener" -> "to have" /
// "to be (years old)"). Both only show once there's more than one
// candidate to step through; cycling wraps back to the first after the
// last.
export function CandidateCycler({ candidates, clickable, onSelect, sourceWord }: CandidateCyclerProps) {
  const [index, setIndex] = useState(0);
  if (candidates.length === 0) return null;

  const clamped = Math.min(index, candidates.length - 1);
  const current = candidates[clamped];
  const hasMultiple = candidates.length > 1;

  const next = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIndex((i) => (i + 1) % candidates.length);
  };

  const body = (
    <span className="candidate-cycler__text">
      {sourceWord && <span className="source">{sourceWord}</span>}
      {sourceWord && <span className="arrow"> → </span>}
      <span className="translation">{current.translation || "…"}</span>
      {current.description && <span className="description">{current.description}</span>}
    </span>
  );

  return (
    <div className="candidate-cycler">
      {clickable && onSelect ? (
        <button type="button" className="candidate-cycler__main" onClick={() => onSelect(current.translation)}>
          {body}
        </button>
      ) : (
        <div className="candidate-cycler__main candidate-cycler__main--static">{body}</div>
      )}
      {hasMultiple && (
        <div className="candidate-cycler__control">
          <span className="candidate-cycler__count">
            {clamped + 1}/{candidates.length}
          </span>
          <button type="button" className="candidate-cycler__next" aria-label="Next option" onClick={next}>
            ↻
          </button>
        </div>
      )}
    </div>
  );
}
