import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "../api/client";
import type { InputTokenAnnotation } from "../api/client";
import { TranslatePopover } from "./TranslatePopover";
import { WordCandidatesPopover } from "./WordCandidatesPopover";
import "./ChatInput.css";

const TAG_DEBOUNCE_MS = 350;
const PHRASE_DEBOUNCE_MS = 300;
const SPANISH_WORD_RE = /^[a-zA-Zñáéíóúü]+$/i;

interface Segment {
  text: string;
  start: number;
  end: number;
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
    if (t.start > cursor) segments.push({ text: text.slice(cursor, t.start), start: cursor, end: t.start });
    segments.push({ text: text.slice(t.start, t.end), start: t.start, end: t.end, token: t });
    cursor = t.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), start: cursor, end: text.length });
  return segments;
}

export function ChatInput({ onSend }: { onSend: (text: string) => void }) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const highlightRef = useRef<HTMLDivElement | null>(null);
  const dragAnchorRef = useRef<number | null>(null);
  const [value, setValue] = useState("");
  const [tags, setTags] = useState<InputTokenAnnotation[]>([]);
  const [openTokenKey, setOpenTokenKey] = useState<string | null>(null);
  const [draftTranslation, setDraftTranslation] = useState<string | null>(null);
  const [draftTranslateState, setDraftTranslateState] = useState<"idle" | "loading" | "error">("idle");
  const [showDraftPreview, setShowDraftPreview] = useState(false);
  const [phraseSelection, setPhraseSelection] = useState<{ text: string; start: number; end: number } | null>(null);
  const [phraseTranslation, setPhraseTranslation] = useState<string | null>(null);
  const [phraseState, setPhraseState] = useState<"idle" | "loading" | "error">("idle");
  const [phraseCoords, setPhraseCoords] = useState<{ top: number; left: number } | null>(null);

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
  const segments = buildSegments(value, effectiveTags);

  const handleValueChange = (next: string) => {
    setValue(next);
    // Whatever caused the text to change - typing, a word-candidate
    // replace, or applying the draft translation below - any previously
    // fetched draft translation is now of stale text, so drop it.
    setDraftTranslation(null);
    setDraftTranslateState("idle");
    setPhraseSelection(null);
  };

  const handleReplace = (token: InputTokenAnnotation, translation: string) => {
    const next = value.slice(0, token.start) + translation + value.slice(token.end);
    handleValueChange(next);
    setOpenTokenKey(null);
  };

  // Highlighting (selecting) a run of text - possibly several words - shows
  // a translation for that whole phrase rather than word-by-word, which
  // reads much better for idioms/collocations than concatenating each
  // word's isolated gloss would. Fires on any selection change, mouse or
  // keyboard - native <textarea> behavior, nothing custom needed for
  // detection itself (see handleWordMouseDown below for why *starting* a
  // mouse drag needs help).
  const handleTextareaSelect = () => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    const { selectionStart, selectionEnd } = textarea;
    if (selectionStart === selectionEnd) {
      setPhraseSelection(null);
      return;
    }
    const text = value.slice(selectionStart, selectionEnd);
    if (!text.trim()) {
      setPhraseSelection(null);
      return;
    }
    setPhraseSelection((prev) =>
      prev && prev.start === selectionStart && prev.end === selectionEnd ? prev : { text, start: selectionStart, end: selectionEnd },
    );
  };

  useEffect(() => {
    if (!phraseSelection) {
      setPhraseTranslation(null);
      setPhraseState("idle");
      return;
    }
    setPhraseState("loading");
    setPhraseTranslation(null);
    const handle = setTimeout(() => {
      api
        .interpretInput({ text: phraseSelection.text })
        .then((res) => {
          setPhraseTranslation(res.target);
          setPhraseState("idle");
        })
        .catch(() => setPhraseState("error"));
    }, PHRASE_DEBOUNCE_MS);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phraseSelection?.text]);

  // Positions the phrase popover near the end of the selection, using the
  // same segment spans rendered for word-hover - finds whichever one the
  // selection's end offset falls in and anchors there.
  useEffect(() => {
    if (!phraseSelection || !highlightRef.current) {
      setPhraseCoords(null);
      return;
    }
    const endOffset = phraseSelection.end;
    const spans = highlightRef.current.querySelectorAll<HTMLElement>("[data-start]");
    let anchorRect: DOMRect | null = null;
    for (const span of spans) {
      const start = Number(span.dataset.start);
      const end = Number(span.dataset.end);
      if (endOffset > start && endOffset <= end) {
        anchorRect = span.getBoundingClientRect();
        break;
      }
    }
    setPhraseCoords(anchorRect ? { top: anchorRect.bottom, left: anchorRect.right } : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phraseSelection?.start, phraseSelection?.end, segments.length]);

  // charOffsetAtPoint() below needs every segment's rendered position, so
  // every span (not just hoverable ones) carries its own [start, end) as
  // data attributes - the plain-text gaps between hoverable words need to
  // participate in hit-testing too, or a drag passing through one would
  // have nowhere to resolve to and just stall.
  const charOffsetAtPoint = (clientX: number, clientY: number): number | null => {
    const container = highlightRef.current;
    if (!container) return null;
    const spans = container.querySelectorAll<HTMLElement>("[data-start]");
    for (const span of spans) {
      const rect = span.getBoundingClientRect();
      if (clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom) {
        const start = Number(span.dataset.start);
        const end = Number(span.dataset.end);
        const fraction = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
        return Math.round(start + Math.min(1, Math.max(0, fraction)) * (end - start));
      }
    }
    return null;
  };

  // The highlight overlay's hoverable spans need pointer-events to catch
  // hover for the popover, which as a side effect blocks clicks - and,
  // worse, drags - from reaching the real textarea underneath (a mousedown
  // that starts on a hoverable span never reaches the textarea at all, so
  // the browser's own drag-to-select never starts there either - you could
  // only ever select text that happened to start in a gap between hoverable
  // words). Redirect the interaction ourselves instead: on mousedown,
  // position the caret (same as before) AND start tracking mousemove/mouseup
  // on the document, manually extending the textarea's selection to
  // wherever the pointer currently is - a hand-rolled drag-select that
  // works the same regardless of what it started on. A plain click (no
  // movement) behaves exactly as before. Clicks inside the popover itself
  // (picking a candidate) are left alone.
  const handleWordMouseDown = (e: React.MouseEvent<HTMLSpanElement>, token: InputTokenAnnotation) => {
    if ((e.target as HTMLElement).closest(".word-candidates-popover")) return;
    e.preventDefault();
    const textarea = textareaRef.current;
    if (!textarea) return;
    const pos = charOffsetAtPoint(e.clientX, e.clientY) ?? token.start;
    textarea.focus();
    textarea.setSelectionRange(pos, pos);
    dragAnchorRef.current = pos;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const anchor = dragAnchorRef.current;
      if (anchor === null) return;
      const current = charOffsetAtPoint(moveEvent.clientX, moveEvent.clientY);
      if (current === null) return;
      textarea.setSelectionRange(Math.min(anchor, current), Math.max(anchor, current));
      handleTextareaSelect();
    };
    const handleMouseUp = () => {
      dragAnchorRef.current = null;
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
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

  return (
    <div className="chat-input">
      <div className="chat-input__editor">
        <div className="chat-input__highlight" aria-hidden="true" ref={highlightRef}>
          {segments.map((seg, i) =>
            seg.token ? (
              <span
                key={i}
                className="chat-input__hoverable"
                data-start={seg.start}
                data-end={seg.end}
                onMouseEnter={() => setOpenTokenKey(`${seg.token!.start}-${seg.token!.end}`)}
                onMouseLeave={() => setOpenTokenKey(null)}
                onMouseDown={(e) => handleWordMouseDown(e, seg.token!)}
              >
                {seg.text}
                {/* Suppressed while a real (multi-char) selection is active - a drag
                    can end with the pointer resting over a word, which would otherwise
                    leave this open at the same time as the phrase popover below. */}
                {openTokenKey === `${seg.token.start}-${seg.token.end}` && !phraseSelection && (
                  <WordCandidatesPopover
                    word={seg.token.surface}
                    columns={seg.token.columns}
                    onSelect={(translation) => handleReplace(seg.token!, translation)}
                  />
                )}
              </span>
            ) : (
              <span key={i} data-start={seg.start} data-end={seg.end}>
                {seg.text}
              </span>
            ),
          )}
        </div>
        <textarea
          ref={textareaRef}
          className="chat-input__textarea"
          value={value}
          placeholder="Escribe en español... (hover or select any text for a translation)"
          onChange={(e) => handleValueChange(e.target.value)}
          onSelect={handleTextareaSelect}
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
      {phraseSelection &&
        phraseCoords &&
        createPortal(
          <div className="chat-input__phrase-popover-anchor" style={{ top: phraseCoords.top, left: phraseCoords.left }}>
            <TranslatePopover
              sourceWord={phraseSelection.text}
              candidates={[
                phraseState === "loading"
                  ? { translation: "…" }
                  : phraseState === "error"
                    ? { translation: "(translation failed)" }
                    : { translation: phraseTranslation ?? "…" },
              ]}
              onSelect={(translation) => {
                const next = value.slice(0, phraseSelection.start) + translation + value.slice(phraseSelection.end);
                handleValueChange(next);
                setPhraseSelection(null);
              }}
            />
          </div>,
          document.body,
        )}
    </div>
  );
}
