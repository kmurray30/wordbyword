import { useState } from "react";
import { AudioButton } from "./AudioButton";
import "./TranslationRow.css";

interface TranslationRowProps {
  variant: "assistant" | "user-native" | "user-target";
  label: string;
  text: string | null;
  loading: boolean;
  language: string;
  voice?: string;
  editable?: boolean;
  onSave?: (newText: string) => void;
}

// One translation line under a chat bubble - e.g. "EN: I like reading."
// `editable` rows (the user's own guessed-English restatement) get a
// pencil icon that swaps the text for an inline input; committing calls
// `onSave`, whose caller is responsible for re-deriving anything downstream
// (the Spanish row, in ChatMessage's case).
export function TranslationRow({
  variant,
  label,
  text,
  loading,
  language,
  voice,
  editable,
  onSave,
}: TranslationRowProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(text ?? "");
  // Tracks the last `text` we've synced `draft` from, so an external change
  // (a fresh fetch, or the caller re-deriving this row after a save) can
  // reset the draft during render instead of via an effect - see "Adjusting
  // state when a prop changes" in the React docs.
  const [syncedText, setSyncedText] = useState(text);
  if (text !== syncedText) {
    setSyncedText(text);
    if (!editing) setDraft(text ?? "");
  }

  const commit = () => {
    setEditing(false);
    const trimmed = draft.trim();
    if (trimmed && trimmed !== text) onSave?.(trimmed);
  };

  if (loading) {
    return (
      <div className={`translation-row translation-row--${variant}`}>
        <span className="translation-row__badge">{label}</span>
        <span className="translation-row__loading">Translating…</span>
      </div>
    );
  }

  if (text === null) return null;

  return (
    <div className={`translation-row translation-row--${variant}`}>
      <span className="translation-row__badge">{label}</span>
      {editing ? (
        <input
          className="translation-row__input"
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commit();
            }
            if (e.key === "Escape") {
              setDraft(text ?? "");
              setEditing(false);
            }
          }}
          onBlur={commit}
        />
      ) : (
        <span className="translation-row__text">{text || "…"}</span>
      )}
      {editable && !editing && (
        <button
          type="button"
          className="translation-row__edit"
          onClick={() => setEditing(true)}
          aria-label="Edit this translation"
          title="Not quite right? Edit it"
        >
          ✏️
        </button>
      )}
      {!editing && text && <AudioButton text={text} language={language} voice={voice} label={`Play ${label} audio`} />}
    </div>
  );
}
