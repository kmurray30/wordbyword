import { useEffect, useRef, useState } from "react";
import { LANGUAGES, LANGUAGE_CODES } from "../lib/language";
import type { LanguageCode } from "../lib/language";
import "./LanguageMenu.css";

// The current language's flag as a button; opens a list of every language
// (flag + name) to switch to. Each language is its own conversation.
export function LanguageMenu({ current, onSelect }: { current: LanguageCode; onSelect: (code: LanguageCode) => void }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="language-menu">
      <button
        type="button"
        className="language-menu__toggle"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`Language: ${LANGUAGES[current].name}`}
      >
        <span className="language-menu__flag" aria-hidden="true">
          {LANGUAGES[current].flag}
        </span>
        <svg
          width="10"
          height="10"
          viewBox="0 0 12 12"
          aria-hidden="true"
          className={`language-menu__chevron${open ? " language-menu__chevron--open" : ""}`}
        >
          <path d="M2 4.5 6 8.5 10 4.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <ul role="listbox" aria-label="Choose a language" className="language-menu__list">
          {LANGUAGE_CODES.map((code) => (
            <li key={code} role="option" aria-selected={code === current}>
              <button
                type="button"
                className={`language-menu__option${code === current ? " language-menu__option--current" : ""}`}
                onClick={() => {
                  setOpen(false);
                  if (code !== current) onSelect(code);
                }}
              >
                <span className="language-menu__flag" aria-hidden="true">
                  {LANGUAGES[code].flag}
                </span>
                <span className="language-menu__name">{LANGUAGES[code].name}</span>
                <span className="language-menu__native" lang={code}>
                  {LANGUAGES[code].nativeName}
                </span>
                {code === current && <span aria-hidden="true">✓</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
