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
}

// One translation candidate at a time - word and its short sense
// description inline on a single line, with a single "next" button to step
// through alternates (e.g. "tener" -> "to have" / "to be (years old)").
// Only shows the button once there's more than one candidate to step
// through; wraps back to the first after the last.
export function CandidateCycler({ candidates, clickable, onSelect }: CandidateCyclerProps) {
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
        <button type="button" className="candidate-cycler__next" aria-label="Next option" onClick={next}>
          ↻
        </button>
      )}
    </div>
  );
}
