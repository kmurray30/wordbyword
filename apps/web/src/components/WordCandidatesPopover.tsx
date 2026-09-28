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
// column per language the word is independently valid in - e.g. "once" is
// Spanish for "eleven" (unclickable gloss) and also an English word
// (clickable Spanish translation), so it gets both, side by side.
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
          <ul>
            {col.candidates.map((c, i) => (
              <li key={i}>
                {col.clickable ? (
                  <button type="button" onClick={() => onSelect(c.translation)}>
                    <span className="translation">{c.translation || "…"}</span>
                    {c.description && <span className="description">{c.description}</span>}
                  </button>
                ) : (
                  <div className="word-candidates-popover__static">
                    <span className="translation">{c.translation || "…"}</span>
                    {c.description && <span className="description">{c.description}</span>}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
