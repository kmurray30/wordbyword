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

// One translation candidate at a time, with prev/next arrows to step
// through alternates (e.g. "tener" -> "to have" / "to be (years old)") and
// a short description of the shown sense - more compact than listing every
// candidate as its own row, and scales to however many candidates a given
// source (curated dictionary, LLM) actually has. Arrows/counter only show
// once there's more than one candidate to step through.
export function CandidateCycler({ candidates, clickable, onSelect }: CandidateCyclerProps) {
  const [index, setIndex] = useState(0);
  if (candidates.length === 0) return null;

  const clamped = Math.min(index, candidates.length - 1);
  const current = candidates[clamped];
  const hasMultiple = candidates.length > 1;

  const step = (delta: number) => (e: React.MouseEvent) => {
    e.stopPropagation();
    setIndex((i) => (i + delta + candidates.length) % candidates.length);
  };

  const body = (
    <>
      <span className="translation">{current.translation || "…"}</span>
      {current.description && <span className="description">{current.description}</span>}
    </>
  );

  return (
    <div className="candidate-cycler">
      {hasMultiple && (
        <button type="button" className="candidate-cycler__arrow" aria-label="Previous option" onClick={step(-1)}>
          ‹
        </button>
      )}
      {clickable && onSelect ? (
        <button type="button" className="candidate-cycler__main" onClick={() => onSelect(current.translation)}>
          {body}
        </button>
      ) : (
        <div className="candidate-cycler__main candidate-cycler__main--static">{body}</div>
      )}
      {hasMultiple && (
        <button type="button" className="candidate-cycler__arrow" aria-label="Next option" onClick={step(1)}>
          ›
        </button>
      )}
      {hasMultiple && (
        <span className="candidate-cycler__count">
          {clamped + 1}/{candidates.length}
        </span>
      )}
    </div>
  );
}
