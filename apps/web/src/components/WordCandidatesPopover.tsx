import "./WordCandidatesPopover.css";

export interface WordCandidate {
  translation: string;
  description?: string;
}

export interface WordCandidateColumn {
  language: string; // "en" | "es"
  candidates: WordCandidate[];
}

interface WordCandidatesPopoverProps {
  columns: WordCandidateColumn[];
  onSelect: (translation: string) => void;
}

const LANGUAGE_LABEL: Record<string, string> = { en: "English", es: "Español" };

// Like TranslatePopover, but for hovering a word in the chat input: usually
// one column (the direction implied by whether the word looks Spanish or
// English), two side by side for a known cross-language cognate (e.g.
// "hotel") where either reading is valid.
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
                <button type="button" onClick={() => onSelect(c.translation)}>
                  <span className="translation">{c.translation || "…"}</span>
                  {c.description && <span className="description">{c.description}</span>}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
