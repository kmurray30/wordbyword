import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { InputTokenAnnotation } from "../api/client";
import { TranslatePopover } from "./TranslatePopover";
import "./ChatInput.css";

const TAG_DEBOUNCE_MS = 350;
const SPANISH_WORD_RE = /^[a-zA-Zñáéíóúü]+$/i;

interface Segment {
  text: string;
  token?: InputTokenAnnotation;
}

function buildSegments(text: string, tokens: InputTokenAnnotation[]): Segment[] {
  const hot = tokens.filter((t) => !t.is_spanish).sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let cursor = 0;
  for (const t of hot) {
    if (t.start > cursor) segments.push({ text: text.slice(cursor, t.start) });
    segments.push({ text: text.slice(t.start, t.end), token: t });
    cursor = t.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor) });
  return segments;
}

export function ChatInput({ onSend }: { onSend: (text: string) => void }) {
  const [value, setValue] = useState("");
  const [tags, setTags] = useState<InputTokenAnnotation[]>([]);
  const [openTokenKey, setOpenTokenKey] = useState<string | null>(null);
  const [draftTranslation, setDraftTranslation] = useState<string | null>(null);

  useEffect(() => {
    if (!value.trim()) return;
    const handle = setTimeout(() => {
      api
        .tagInput({ text: value })
        .then((res) => setTags(res.tokens))
        .catch(() => {});
    }, TAG_DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [value]);

  // Derived rather than synced via effect: avoids a stale word-candidate
  // popover surviving a moment after the user clears the input.
  const effectiveTags = value.trim() ? tags : [];

  const handleValueChange = (next: string) => {
    setValue(next);
    setDraftTranslation(null);
  };

  const handleReplace = (token: InputTokenAnnotation, translation: string) => {
    const next = value.slice(0, token.start) + translation + value.slice(token.end);
    handleValueChange(next);
    setOpenTokenKey(null);
  };

  const handleSend = () => {
    const text = value.trim();
    if (!text) return;

    for (const tok of effectiveTags) {
      if (tok.is_spanish && SPANISH_WORD_RE.test(tok.surface) && tok.surface.length > 1) {
        api.rewardEvent({ event_type: "userTypedSpanishWord", lemma: tok.lemma }).catch(() => {});
      }
    }

    onSend(text);
    handleValueChange("");
    setTags([]);
  };

  const handleTranslateDraft = () => {
    if (draftTranslation !== null || !value.trim()) return;
    const spanishCount = effectiveTags.filter((t) => t.is_spanish).length;
    const mostlySpanish = spanishCount >= effectiveTags.length / 2;
    api
      .translateText({
        text: value,
        source_lang: mostlySpanish ? "es" : "en",
        target_lang: mostlySpanish ? "en" : "es",
      })
      .then((res) => setDraftTranslation(res.translation))
      .catch(() => setDraftTranslation("(translation failed)"));
  };

  const segments = buildSegments(value, effectiveTags);

  return (
    <div className="chat-input">
      <div className="chat-input__editor">
        <div className="chat-input__highlight" aria-hidden="true">
          {segments.map((seg, i) =>
            seg.token ? (
              <span
                key={i}
                className="chat-input__flagged"
                onMouseEnter={() => setOpenTokenKey(`${seg.token!.start}-${seg.token!.end}`)}
                onMouseLeave={() => setOpenTokenKey(null)}
              >
                {seg.text}
                {openTokenKey === `${seg.token.start}-${seg.token.end}` && (
                  <TranslatePopover
                    candidates={seg.token.candidates}
                    onSelect={(translation) => handleReplace(seg.token!, translation)}
                  />
                )}
              </span>
            ) : (
              <span key={i}>{seg.text}</span>
            ),
          )}
        </div>
        <textarea
          className="chat-input__textarea"
          value={value}
          placeholder="Escribe en español... (English words you drop in get underlined)"
          onChange={(e) => handleValueChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />
      </div>
      <span className="chat-input__translate-all" onMouseEnter={handleTranslateDraft}>
        🌐
        {draftTranslation !== null && <TranslatePopover candidates={[{ translation: draftTranslation }]} />}
      </span>
      <button type="button" onClick={handleSend} disabled={!value.trim()}>
        Send
      </button>
    </div>
  );
}
