import { useState } from "react";
import { CandidateCycler } from "./CandidateCycler";
import "./WordCandidatesPopover.css";

export interface WordCandidate {
  translation: string;
  description?: string;
}

export interface WordCandidateColumn {
  language: string; // "en" | "es" - the language these candidates translate INTO
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

// Just the language name each column's candidates ARE in - the word/phrase
// being translated is already right there under the pointer or highlighted,
// so repeating it (an arrow, "sourceWord -> translation") is redundant.
const DIRECTION_LABEL: Record<string, string> = { es: "Spanish", en: "English" };

// Fixed left-to-right order for the toggle header - independent of the
// columns prop's own order, which mirrors the backend's evaluation order
// (Spanish-validity checked first), not a display preference.
const TOGGLE_ORDER = ["es", "en"];

// The one popover widget for both hovering a single word and highlighting a
// phrase in the chat input. A word valid in only one language (the common
// case) just shows that one reading. A word valid in both (e.g. "hotel",
// "once" - Spanish for "eleven" and also an English word) gets a
// "Spanish | English" toggle instead of stacking both readings at once -
// showing them stacked was what made this popover look far busier than the
// single-candidate phrase popover it's meant to match. Each reading is a
// CandidateCycler, so a word with multiple senses (e.g. "tener") steps
// through them in place instead of listing every one.
export function WordCandidatesPopover({ columns, onSelect }: WordCandidatesPopoverProps) {
  const visible = columns.filter((c) => c.candidates.length > 0);
  const [selectedLanguage, setSelectedLanguage] = useState<string | null>(null);
  if (visible.length === 0) return null;

  if (visible.length === 1) {
    const col = visible[0];
    return (
      <div className="word-candidates-popover" role="tooltip">
        <div className="word-candidates-popover__heading">{DIRECTION_LABEL[col.language] ?? col.language}</div>
        <CandidateCycler candidates={col.candidates} clickable={col.clickable} onSelect={onSelect} />
      </div>
    );
  }

  const activeLanguage =
    selectedLanguage && visible.some((c) => c.language === selectedLanguage) ? selectedLanguage : visible[0].language;
  const activeColumn = visible.find((c) => c.language === activeLanguage)!;
  const toggleLanguages = TOGGLE_ORDER.filter((lang) => visible.some((c) => c.language === lang));

  return (
    <div className="word-candidates-popover" role="tooltip">
      <div className="word-candidates-popover__toggle">
        {toggleLanguages.map((lang, i) => (
          <span key={lang} className="word-candidates-popover__toggle-item">
            {i > 0 && <span className="word-candidates-popover__toggle-sep">|</span>}
            <button
              type="button"
              className={`word-candidates-popover__toggle-btn${
                lang === activeLanguage ? " word-candidates-popover__toggle-btn--active" : ""
              }`}
              onClick={() => setSelectedLanguage(lang)}
            >
              {DIRECTION_LABEL[lang] ?? lang}
            </button>
          </span>
        ))}
      </div>
      <CandidateCycler candidates={activeColumn.candidates} clickable={activeColumn.clickable} onSelect={onSelect} />
    </div>
  );
}
