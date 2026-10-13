import { useEffect, useRef, useState } from "react";
import type { SavedWord } from "../api/client";
import { htmlLang } from "../lib/language";
import "./WordListPanel.css";

// The learner's saved word list (opened from ☰): everything they've kept by
// tapping words in chat, with a box to add a word by hand and a ✕ to take
// one off. These are the words contextual vocabulary favors when it picks
// what the partner might work into a reply.
export function WordListPanel({
  words,
  onAdd,
  onRemove,
  onClose,
}: {
  words: SavedWord[];
  // Resolves once the word is on the list (it fetches the meaning first).
  onAdd: (word: string) => Promise<void>;
  onRemove: (word: string) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState("");
  const [adding, setAdding] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const submit = async () => {
    const word = draft.trim();
    if (!word || adding) return;
    setAdding(true);
    try {
      await onAdd(word);
      setDraft("");
    } finally {
      setAdding(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="word-list" role="dialog" aria-label="Word list">
      <div className="word-list__backdrop" onClick={onClose} />
      <div className="word-list__sheet">
        <div className="word-list__head">
          <span className="word-list__title">Word list ({words.length})</span>
          <button type="button" className="word-list__close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <form
          className="word-list__add"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <input
            ref={inputRef}
            className="word-list__input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Add a word"
            lang={htmlLang()}
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            // Not a form field the iOS keyboard bar should chain through.
            enterKeyHint="done"
          />
          <button type="submit" className="word-list__add-button" disabled={!draft.trim() || adding}>
            {adding ? "Adding…" : "Add"}
          </button>
        </form>
        {words.length === 0 ? (
          <p className="word-list__empty">
            Nothing here yet. Add a word above, or tap a word in the chat three times to save it.
          </p>
        ) : (
          <ul className="word-list__list">
            {words.map((w) => (
              <li key={w.id} className="word-list__item">
                <div className="word-list__text">
                  <span className="word-list__word" lang={htmlLang()}>
                    {w.word}
                  </span>
                  {w.gloss && <span className="word-list__gloss"> {w.gloss}</span>}
                  {w.grammar && <div className="word-list__grammar">{w.grammar}</div>}
                  {w.sentence && (
                    <div className="word-list__sentence" lang={htmlLang()}>
                      {w.sentence}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  className="word-list__remove"
                  onClick={() => onRemove(w.word)}
                  aria-label={`Remove ${w.word}`}
                  title="Remove from the list"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
