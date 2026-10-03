import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { DraftSpan, DraftToken } from "../api/client";
import { CoachPopover } from "./CoachPopover";
import { DockedPopover } from "./DockedPopover";
import { WordCandidatesPopover } from "./WordCandidatesPopover";
import "./ChatInput.css";

// Fired almost immediately after the just-typed character completes a word
// boundary (space/punctuation) - just enough delay to coalesce a fast burst
// (paste, double punctuation) without feeling laggy. Falls back to a longer
// idle debounce for the word currently being typed, with no trailing
// boundary yet - bumped up from the old single-rate 350ms debounce since
// this now costs a real LLM call (/translate/tag-input), not an instant
// dictionary lookup.
const BOUNDARY_DEBOUNCE_MS = 60;
const IDLE_DEBOUNCE_MS = 550;
const WORD_BOUNDARY_RE = /[\s.,!?;:¡¿()"'«»…]/;
const PHRASE_DEBOUNCE_MS = 300;
const SPANISH_WORD_RE = /^[a-zA-Zñáéíóúü]+$/i;
// How far a touch can drift between start and end before it's treated as a
// drag/scroll/selection rather than a tap.
const TAP_MOVE_TOLERANCE_PX = 10;
// Grace period before closing the word-tap popover on mouseleave - it's
// now docked flush against the input bar rather than floating right next
// to the hovered word, so the pointer has to travel there to click a
// candidate, leaving the word's own bounds along the way. Without this,
// that departure closes (unmounts) the popover before the click ever
// lands on it.
const HOVER_CLOSE_DELAY_MS = 200;

interface Segment {
  text: string;
  start: number;
  end: number;
  span?: DraftSpan;
}

function buildSegments(text: string, spans: DraftSpan[]): Segment[] {
  // Every returned span is hoverable by construction (the backend only
  // returns one when it has something usable to show) - unlike the old
  // per-token columns, no filtering needed here. Spans are already
  // non-overlapping (app.translate.span_matching guarantees it), but sort
  // defensively anyway since nothing else here depends on response order.
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let cursor = 0;
  for (const s of sorted) {
    if (s.start < cursor) continue;
    if (s.start > cursor) segments.push({ text: text.slice(cursor, s.start), start: cursor, end: s.start });
    segments.push({ text: text.slice(s.start, s.end), start: s.start, end: s.end, span: s });
    cursor = s.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), start: cursor, end: text.length });
  return segments;
}

export function ChatInput({ onSend, sessionId }: { onSend: (text: string) => void; sessionId: string }) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const highlightRef = useRef<HTMLDivElement | null>(null);
  const dragAnchorRef = useRef<number | null>(null);
  const touchStartRef = useRef<{ x: number; y: number } | null>(null);
  const touchMovedRef = useRef(false);
  const wasFocusedRef = useRef(false);
  const hoverCloseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Bumped on every /translate/tag-input request fired, and checked again
  // when each one resolves - since an LLM-backed call can take real time
  // (unlike the old near-instant dictionary lookup), a request for an
  // earlier, shorter draft can resolve AFTER one for a later, longer draft.
  // A response is only applied if it's still the most recent request
  // issued, so a slow stale response can never clobber a newer result.
  const tagSeqRef = useRef(0);
  const [value, setValue] = useState("");
  const [tokens, setTokens] = useState<DraftToken[]>([]);
  const [spans, setSpans] = useState<DraftSpan[]>([]);
  const [openTokenKey, setOpenTokenKey] = useState<string | null>(null);
  // The help button's coaching result for the current draft - cleared
  // whenever the draft text changes (handleValueChange below), same as
  // the old draft-translation preview it replaced.
  const [coachResult, setCoachResult] = useState<{
    meaning: string;
    feedback: string;
    options: { formality: string; spanish: string }[];
  } | null>(null);
  const [coachState, setCoachState] = useState<"idle" | "loading" | "error">("idle");
  const [showCoach, setShowCoach] = useState(false);
  const [phraseSelection, setPhraseSelection] = useState<{ text: string; start: number; end: number } | null>(null);
  const [phraseTranslation, setPhraseTranslation] = useState<string | null>(null);
  const [phraseState, setPhraseState] = useState<"idle" | "loading" | "error">("idle");
  // Which way the phrase popover translates: "es" (candidate IS Spanish -
  // a mostly-English/mixed selection, corrected and translated INTO
  // Spanish, clickable to swap in) or "en" (candidate IS English - a
  // mostly-Spanish selection, translated INTO English, just a gloss).
  // Decided per-selection below from the majority language of the words it
  // covers.
  const [phraseDirection, setPhraseDirection] = useState<"es" | "en">("es");

  const fireTagInput = (text: string) => {
    const seq = ++tagSeqRef.current;
    api
      .tagInput({ text })
      .then((res) => {
        if (seq !== tagSeqRef.current) return; // a newer request has since been issued - drop this stale result
        setTokens(res.tokens);
        setSpans(res.spans);
      })
      .catch(() => {});
  };

  useEffect(() => {
    if (!value.trim()) {
      setTokens([]);
      setSpans([]);
      return;
    }
    const lastChar = value[value.length - 1];
    const delay = WORD_BOUNDARY_RE.test(lastChar) ? BOUNDARY_DEBOUNCE_MS : IDLE_DEBOUNCE_MS;
    const handle = setTimeout(() => fireTagInput(value), delay);
    return () => clearTimeout(handle);
  }, [value]);

  // Tapping virtually anything else on the page while this textarea is
  // focused - a chat-bubble word, the clear-chat button, a message's audio
  // button, empty space in the feed - was blurring it and taking the
  // mobile keyboard down with it, since that's a real text input and
  // losing focus is the browser's normal behavior for a tap elsewhere.
  // Checking a word's meaning or reading a past message mid-draft doesn't
  // need to end the draft, so this puts focus right back once the tap
  // resolves - unless it landed on a genuine different form control
  // (another input/textarea/select), which should keep its own focus.
  // Done by restoring focus afterward rather than preventing default on
  // the pointerdown/click themselves, which would also risk suppressing
  // whatever click the tapped element itself depends on.
  useEffect(() => {
    const handlePointerDown = () => {
      wasFocusedRef.current = document.activeElement === textareaRef.current;
    };
    const handleClick = (e: MouseEvent) => {
      if (!wasFocusedRef.current) return;
      const textarea = textareaRef.current;
      if (!textarea || document.activeElement === textarea) return;
      const target = e.target as HTMLElement | null;
      if (target?.closest("input, textarea, select")) return;
      textarea.focus();
    };
    document.addEventListener("pointerdown", handlePointerDown, true);
    document.addEventListener("click", handleClick, true);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown, true);
      document.removeEventListener("click", handleClick, true);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (hoverCloseTimerRef.current !== null) clearTimeout(hoverCloseTimerRef.current);
    };
  }, []);

  // Derived rather than synced via effect: avoids a stale word-candidate
  // popover surviving a moment after the user clears the input.
  const effectiveTokens = value.trim() ? tokens : [];
  const effectiveSpans = value.trim() ? spans : [];
  const segments = buildSegments(value, effectiveSpans);
  // Suppressed while a real (multi-char) selection is active - a drag can
  // end with the pointer resting over a word, which would otherwise leave
  // this open at the same time as the phrase popover.
  const openToken = !phraseSelection
    ? effectiveSpans.find((s) => `${s.start}-${s.end}` === openTokenKey)
    : undefined;

  const handleValueChange = (next: string) => {
    setValue(next);
    // Whatever caused the text to change - typing, a word-candidate
    // replace, or applying a coaching suggestion below - any previously
    // fetched coaching result is now of stale text, so drop it.
    setCoachResult(null);
    setCoachState("idle");
    setPhraseSelection(null);
  };

  const handleReplace = (span: DraftSpan, translation: string) => {
    const next = value.slice(0, span.start) + translation + value.slice(span.end);
    handleValueChange(next);
    setOpenTokenKey(null);
    // The candidate button just taken focus (it's a real <button>, inside a
    // portal) blurred the textarea and would otherwise take the keyboard
    // down with it right as the user's about to keep editing.
    textareaRef.current?.focus();
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
    // Majority-vote the language of the real words the selection covers
    // (same per-word is_spanish flags /translate/tag-input's cheap token
    // pass already computed) to decide which way to translate: a mostly-
    // Spanish selection goes to English, a mostly-English (or mixed/typo-
    // ridden) one goes to Spanish, same as the per-word popover's two
    // directions.
    const overlapping = effectiveTokens.filter(
      (t) => t.end > phraseSelection.start && t.start < phraseSelection.end,
    );
    const spanishCount = overlapping.filter((t) => t.is_spanish).length;
    const isSpanishPhrase = overlapping.length > 0 && spanishCount * 2 > overlapping.length;
    setPhraseDirection(isSpanishPhrase ? "en" : "es");

    setPhraseState("loading");
    setPhraseTranslation(null);
    const handle = setTimeout(() => {
      const translation = isSpanishPhrase
        ? api
            .translateText({ text: phraseSelection.text, source_lang: "es", target_lang: "en" })
            .then((res) => res.translation)
        : api.interpretInput({ text: phraseSelection.text }).then((res) => res.target);
      translation
        .then((text) => {
          setPhraseTranslation(text);
          setPhraseState("idle");
        })
        .catch(() => setPhraseState("error"));
    }, PHRASE_DEBOUNCE_MS);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phraseSelection?.text]);

  // charOffsetAtPoint() below needs every segment's rendered position, so
  // every span (not just hoverable ones) carries its own [start, end) as
  // data attributes - the plain-text gaps between hoverable words need to
  // participate in hit-testing too, or a drag passing through one would
  // have nowhere to resolve to and just stall.
  const charOffsetAtPoint = (clientX: number, clientY: number): number | null => {
    const container = highlightRef.current;
    if (!container) return null;
    const els = container.querySelectorAll<HTMLElement>("[data-start]");
    for (const el of els) {
      const rect = el.getBoundingClientRect();
      if (clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom) {
        const start = Number(el.dataset.start);
        const end = Number(el.dataset.end);
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
  const handleWordMouseDown = (e: React.MouseEvent<HTMLSpanElement>, span: DraftSpan) => {
    if ((e.target as HTMLElement).closest(".word-candidates-popover")) return;
    e.preventDefault();
    const textarea = textareaRef.current;
    if (!textarea) return;
    const pos = charOffsetAtPoint(e.clientX, e.clientY) ?? span.start;
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

  // Touch's equivalent of hover: the overlay spans can't receive touch
  // events at all here (pointer-events:none on touch, see ChatInput.css -
  // required so a plain tap reaches the textarea underneath for cursor
  // placement), so this listens on the textarea itself instead, which
  // always gets every touch.
  //
  // This used to wait for a ~450ms hold (a long-press) before opening the
  // popover, on the theory that a plain tap should be left alone for
  // cursor placement. In practice that collided with iOS's OWN long-press-
  // on-a-text-field gesture (the selection magnifier + callout menu), which
  // engages on a very similar timescale - racing our timer against it
  // produced exactly the reported symptoms (translation not showing up,
  // odd focus/keyboard behavior) rather than a clean popover. A plain tap
  // never approaches that threshold at all, so resolving on touchend
  // (only if the touch didn't move - see TAP_MOVE_TOLERANCE_PX - so an
  // actual drag-to-select still reaches the textarea's native selection
  // untouched) sidesteps the conflict entirely: native caret placement AND
  // the popover both happen from the same tap, same as clicking a word in
  // a chat bubble already does.
  const handleTextareaTouchStart = (e: React.TouchEvent<HTMLTextAreaElement>) => {
    if (openTokenKey) setOpenTokenKey(null);
    const touch = e.touches[0];
    touchStartRef.current = touch ? { x: touch.clientX, y: touch.clientY } : null;
    touchMovedRef.current = false;
  };

  const handleTextareaTouchMove = (e: React.TouchEvent<HTMLTextAreaElement>) => {
    const start = touchStartRef.current;
    const touch = e.touches[0];
    if (!start || !touch || touchMovedRef.current) return;
    const dx = touch.clientX - start.x;
    const dy = touch.clientY - start.y;
    if (Math.hypot(dx, dy) > TAP_MOVE_TOLERANCE_PX) touchMovedRef.current = true;
  };

  const handleTextareaTouchEnd = (e: React.TouchEvent<HTMLTextAreaElement>) => {
    const hadStart = touchStartRef.current !== null;
    const moved = touchMovedRef.current;
    touchStartRef.current = null;
    touchMovedRef.current = false;
    if (!hadStart || moved) return;

    const touch = e.changedTouches[0];
    if (!touch) return;
    const offset = charOffsetAtPoint(touch.clientX, touch.clientY);
    if (offset === null) return;
    const span = effectiveSpans.find((s) => offset >= s.start && offset <= s.end);
    if (!span) return;
    setShowCoach(false);
    setOpenTokenKey(`${span.start}-${span.end}`);
  };

  const cancelHoverClose = () => {
    if (hoverCloseTimerRef.current !== null) {
      clearTimeout(hoverCloseTimerRef.current);
      hoverCloseTimerRef.current = null;
    }
  };

  // See HOVER_CLOSE_DELAY_MS - a grace period rather than closing on the
  // spot, so the pointer has time to reach the docked popover (now flush
  // against the input bar, not right next to the word) before it unmounts.
  const scheduleHoverClose = () => {
    cancelHoverClose();
    hoverCloseTimerRef.current = setTimeout(() => {
      hoverCloseTimerRef.current = null;
      setOpenTokenKey(null);
    }, HOVER_CLOSE_DELAY_MS);
  };

  const handleSend = () => {
    const text = value.trim();
    if (!text) return;

    for (const tok of effectiveTokens) {
      if (tok.is_spanish && SPANISH_WORD_RE.test(tok.surface) && tok.surface.length > 1) {
        api.rewardEvent({ event_type: "userTypedSpanishWord", lemma: tok.lemma }).catch(() => {});
      }
    }

    onSend(text);
    handleValueChange("");
    tagSeqRef.current += 1; // invalidate any outstanding tagInput request for the just-sent draft
    setTokens([]);
    setSpans([]);
  };

  // The help button's draft coaching - click-toggled rather than hover-
  // triggered (a hover-only trigger never fires on a touch device at all,
  // and the old globe button's only path to showing up on mobile was an
  // iOS "ghost hover" on a first tap, which left a second tap with nothing
  // left to do - no click handler ever existed for it to toggle). Fetches
  // once per draft (cached in coachResult until handleValueChange clears
  // it) and uses the whole recent conversation for context, unlike the
  // flat interpretInput call it replaced.
  const fetchCoach = () => {
    if (coachState === "loading" || coachResult !== null || !value.trim()) return;
    setCoachState("loading");
    api
      .coachDraft({ text: value, session_id: sessionId })
      .then((res) => {
        setCoachResult({ meaning: res.meaning, feedback: res.feedback, options: res.options });
        setCoachState("idle");
      })
      .catch(() => setCoachState("error"));
  };

  const closeCoach = () => {
    setShowCoach(false);
    textareaRef.current?.focus();
  };

  const handleToggleCoach = () => {
    if (!value.trim()) return;
    const next = !showCoach;
    setShowCoach(next);
    if (next) {
      fetchCoach();
      setOpenTokenKey(null);
      setPhraseSelection(null);
    }
    // This button is tabIndex-focusable (for keyboard/a11y use), which
    // blurs the textarea on tap - checking a draft shouldn't cost the
    // keyboard when you're about to go right back to editing.
    textareaRef.current?.focus();
  };

  return (
    <div className="chat-input">
      <div className="chat-input__editor">
        <div className="chat-input__highlight" aria-hidden="true" ref={highlightRef}>
          {segments.map((seg, i) =>
            seg.span ? (
              <span
                key={i}
                className={`chat-input__hoverable${
                  openTokenKey === `${seg.span.start}-${seg.span.end}` ? " chat-input__hoverable--active" : ""
                }`}
                data-start={seg.start}
                data-end={seg.end}
                onMouseEnter={() => {
                  cancelHoverClose();
                  setOpenTokenKey(`${seg.span!.start}-${seg.span!.end}`);
                  setShowCoach(false);
                }}
                onMouseLeave={scheduleHoverClose}
                onMouseDown={(e) => handleWordMouseDown(e, seg.span!)}
              >
                {seg.text}
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
          placeholder="Escribe en español... (hover, tap, or select text for a translation)"
          onChange={(e) => handleValueChange(e.target.value)}
          onSelect={handleTextareaSelect}
          onTouchStart={handleTextareaTouchStart}
          onTouchMove={handleTextareaTouchMove}
          onTouchEnd={handleTextareaTouchEnd}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />
      </div>
      <span
        className={`chat-input__help chat-input__help--${coachState}${showCoach ? " chat-input__help--active" : ""}`}
        onClick={handleToggleCoach}
        role="button"
        tabIndex={value.trim() ? 0 : -1}
        aria-label="Help with this draft"
        aria-pressed={showCoach}
        title="What am I trying to say, and how should I actually say it?"
      >
        {coachState === "loading" ? "⏳" : "🪄"}
      </span>
      <button type="button" onClick={handleSend} disabled={!value.trim()}>
        Send
      </button>
      {/* Docked flush against the input bar's own top edge (see
          DockedPopover) rather than floating next to whatever word/button
          triggered it - a single, predictable, always-on-screen spot
          regardless of where in the draft that word or selection sits. */}
      {openToken && (
        <DockedPopover onMouseEnter={cancelHoverClose} onMouseLeave={scheduleHoverClose}>
          <WordCandidatesPopover
            candidates={openToken.candidates}
            clickable={openToken.clickable}
            onSelect={(translation) => handleReplace(openToken, translation)}
          />
        </DockedPopover>
      )}
      {phraseSelection && (
        <DockedPopover>
          <WordCandidatesPopover
            candidates={[
              phraseState === "loading"
                ? { translation: "…" }
                : phraseState === "error"
                  ? { translation: "(translation failed)" }
                  : { translation: phraseTranslation ?? "…" },
            ]}
            clickable={phraseDirection === "es"}
            onSelect={(translation) => {
              const next = value.slice(0, phraseSelection.start) + translation + value.slice(phraseSelection.end);
              handleValueChange(next);
              setPhraseSelection(null);
              textareaRef.current?.focus();
            }}
          />
        </DockedPopover>
      )}
      {showCoach && coachResult && (
        <DockedPopover position="below">
          <CoachPopover
            meaning={coachResult.meaning}
            feedback={coachResult.feedback}
            options={coachResult.options}
            loading={false}
            error={false}
            onSelect={(spanish) => {
              handleValueChange(spanish);
              closeCoach();
            }}
            onClose={closeCoach}
          />
        </DockedPopover>
      )}
      {showCoach && coachState === "loading" && (
        <DockedPopover position="below">
          <CoachPopover meaning="" feedback="" options={[]} loading error={false} onSelect={() => {}} onClose={closeCoach} />
        </DockedPopover>
      )}
      {showCoach && coachState === "error" && (
        <DockedPopover position="below">
          <CoachPopover meaning="" feedback="" options={[]} loading={false} error onSelect={() => {}} onClose={closeCoach} />
        </DockedPopover>
      )}
    </div>
  );
}
