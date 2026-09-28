import { CandidateCycler } from "./CandidateCycler";
import "./WordCandidatesPopover.css";

export interface WordCandidate {
  translation: string;
  description?: string;
}

export interface WordCandidateColumn {
  language: string; // "en" | "es"
  candidates: WordCandidate[];
  // False for a column just showing what a word already means in Spanish
  // (nothing to replace); true for a column offering a Spanish translation
  // to swap in for a word read as English.
  clickable: boolean;
}

interface WordCandidatesPopoverProps {
  columns: WordCandidateColumn[];
  onSelect: (translation: string) => void;
}

const LANGUAGE_LABEL: Record<string, string> = { en: "English", es: "Español" };

// Like TranslatePopover, but for hovering a word in the chat input: one
// row per language the word is independently valid in - e.g. "once" is
// Spanish for "eleven" (unclickable gloss) and also an English word
// (clickable Spanish translation), so it gets both, stacked. Each row is a
// CandidateCycler, so a word with multiple senses (e.g. "tener") steps
// through them in place instead of listing every one.
export function WordCandidatesPopover({ columns, onSelect }: WordCandidatesPopoverProps) {
  const visible = columns.filter((c) => c.candidates.length > 0);
  if (visible.length === 0) return null;

  return (
    <div
      className={`word-candidates-popover${visible.length > 1 ? " word-candidates-popover--dual" : ""}`}
      role="tooltip"
    >
      {visible.map((col) => (
        <div className="word-candidates-popover__column" key={col.language}>
          <div className="word-candidates-popover__heading">{LANGUAGE_LABEL[col.language] ?? col.language}</div>
          <CandidateCycler candidates={col.candidates} clickable={col.clickable} onSelect={onSelect} />
        </div>
      ))}
    </div>
  );
}
