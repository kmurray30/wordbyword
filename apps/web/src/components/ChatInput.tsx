import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { InputTokenAnnotation } from "../api/client";
import { TranslatePopover } from "./TranslatePopover";
import { WordCandidatesPopover } from "./WordCandidatesPopover";
import "./ChatInput.css";

const TAG_DEBOUNCE_MS = 350;
const SPANISH_WORD_RE = /^[a-zA-Zñáéíóúü]+$/i;

interface Segment {
  text: string;
  token?: InputTokenAnnotation;
}

function buildSegments(text: string, tokens: InputTokenAnnotation[]): Segment[] {
  // Any word with at least one translation column is hoverable - that's
  // every real word now, not just ones flagged as non-Spanish (see
  // /translate/tag-input: a Spanish word gets an ES->EN column, an English
  // one gets EN->ES, a cognate like "hotel" gets both).
  const hot = tokens.filter((t) => t.columns.length > 0).sort((a, b) => a.start - b.start);
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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [value, setValue] = useState("");
  const [tags, setTags] = useState<InputTokenAnnotation[]>([]);
  const [openTokenKey, setOpenTokenKey] = useState<string | null>(null);
  const [draftTranslation, setDraftTranslation] = useState<string | null>(null);
  const [draftTranslateState, setDraftTranslateState] = useState<"idle" | "loading" | "error">("idle");
  const [showDraftPreview, setShowDraftPreview] = useState(false);

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
    // Whatever caused the text to change - typing, a word-candidate
    // replace, or applying the draft translation below - any previously
    // fetched draft translation is now of stale text, so drop it.
    setDraftTranslation(null);
    setDraftTranslateState("idle");
  };

  const handleReplace = (token: InputTokenAnnotation, translation: string) => {
    const next = value.slice(0, token.start) + translation + value.slice(token.end);
    handleValueChange(next);
    setOpenTokenKey(null);
  };

  // The highlight overlay's hoverable spans need pointer-events to catch
  // hover for the popover, which as a side effect blocks clicks from
  // reaching the real textarea underneath - you couldn't click into the
  // text near a hoverable word to place your cursor there. Redirect the
  // click ourselves: focus the textarea and place its caret at roughly the
  // clicked position within the word (proportional to where in the span's
  // width the click landed), so it still feels like clicking straight
  // through. Clicks inside the popover itself (picking a candidate) are
  // left alone.
  const handleWordMouseDown = (e: React.MouseEvent<HTMLSpanElement>, token: InputTokenAnnotation) => {
    if ((e.target as HTMLElement).closest(".word-candidates-popover")) return;
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
    const pos = Math.round(token.start + Math.min(1, Math.max(0, fraction)) * (token.end - token.start));
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.focus();
    textarea.setSelectionRange(pos, pos);
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

  // Hovering the globe fetches and previews the translation (without
  // touching the draft); clicking the preview's suggestion applies it -
  // same interaction pattern as the word-candidate popovers. Always targets
  // Spanish, regardless of what's typed so far - interpretInput handles a
  // draft that's already partly or fully Spanish (or a mix, or has typos)
  // and returns a single natural Spanish line for it, rather than us having
  // to guess a source language first.
  const fetchDraftTranslation = () => {
    if (draftTranslateState === "loading" || draftTranslation !== null || !value.trim()) return;
    setDraftTranslateState("loading");
    api
      .interpretInput({ text: value })
      .then((res) => {
        setDraftTranslation(res.target);
        setDraftTranslateState("idle");
      })
      .catch(() => setDraftTranslateState("error"));
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
                // Non-Spanish words keep the dashed underline (a visual
                // "this doesn't look like Spanish" cue); Spanish/cognate
                // words are still hoverable for a translation, just without
                // the underline, since flagging every ordinary Spanish word
                // as if it were wrong would be misleading.
                className={`chat-input__hoverable${!seg.token.is_spanish ? " chat-input__flagged" : ""}`}
                onMouseEnter={() => setOpenTokenKey(`${seg.token!.start}-${seg.token!.end}`)}
                onMouseLeave={() => setOpenTokenKey(null)}
                onMouseDown={(e) => handleWordMouseDown(e, seg.token!)}
              >
                {seg.text}
                {openTokenKey === `${seg.token.start}-${seg.token.end}` && (
                  <WordCandidatesPopover
                    columns={seg.token.columns}
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
          ref={textareaRef}
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
      <span
        className={`chat-input__translate-all chat-input__translate-all--${draftTranslateState}`}
        onMouseEnter={() => {
          setShowDraftPreview(true);
          fetchDraftTranslation();
        }}
        onMouseLeave={() => setShowDraftPreview(false)}
        role="button"
        tabIndex={value.trim() ? 0 : -1}
        aria-label="Preview draft translation"
        title={draftTranslateState === "error" ? "Translation failed - try again" : "Hover to preview, click the suggestion to use it"}
      >
        {draftTranslateState === "loading" ? "⏳" : "🌐"}
        {showDraftPreview && draftTranslation !== null && (
          <TranslatePopover
            candidates={[{ translation: draftTranslation }]}
            direction="up"
            onSelect={(translation) => {
              handleValueChange(translation);
              setShowDraftPreview(false);
            }}
          />
        )}
      </span>
      <button type="button" onClick={handleSend} disabled={!value.trim()}>
        Send
      </button>
    </div>
  );
}
