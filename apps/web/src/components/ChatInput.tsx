import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { CoachOption, DraftSpan, DraftToken } from "../api/client";
import { CoachPopover } from "./CoachPopover";
import { DockedPopover } from "./DockedPopover";
import { WordCandidatesPopover } from "./WordCandidatesPopover";
import "./ChatInput.css";

// The cheap, LLM-free per-word pass (/translate/tag-input - spaCy
// tokenize/lemmatize only) still runs on a short debounce as the user
// types - fast enough that this costs nothing. It also doubles as the
// NAIVE hoverable word boundaries shown before the real (LLM-backed,
// app.translate.llm_translate.tag_draft) gloss for this exact text has
// been fetched - see ensureGlossSpans below, which fetches that
// separately and lazily, only once a word is actually hovered/tapped.
const BOUNDARY_DEBOUNCE_MS = 60;
const IDLE_DEBOUNCE_MS = 400;
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
  hoverable: boolean;
  // Present only once the real (LLM-backed) gloss data for this exact
  // span has been fetched - absent for a naive, not-yet-glossed hoverable
  // range (see buildSegments).
  span?: DraftSpan;
}

// Builds the highlight overlay's segments from whichever hoverable-range
// set is authoritative right now: the real, LLM-glossed DraftSpan[] once
// ensureGlossSpans has resolved for the CURRENT text, or (initially, or
// while that's stale/in flight) a naive one-range-per-cheap-token
// fallback - see the hoverRanges computation in ChatInput below. Either
// way, ranges are assumed non-overlapping and are sorted defensively since
// nothing here depends on the caller's order.
function buildSegments(text: string, ranges: { start: number; end: number; span?: DraftSpan }[]): Segment[] {
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let cursor = 0;
  for (const r of sorted) {
    if (r.start < cursor) continue;
    if (r.start > cursor) segments.push({ text: text.slice(cursor, r.start), start: cursor, end: r.start, hoverable: false });
    segments.push({ text: text.slice(r.start, r.end), start: r.start, end: r.end, hoverable: true, span: r.span });
    cursor = r.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), start: cursor, end: text.length, hoverable: false });
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
  // when each one resolves - a request for an earlier, shorter draft can
  // resolve AFTER one for a later, longer draft. A response is only
  // applied if it's still the most recent request issued, so a slow stale
  // response can never clobber a newer result.
  const tagSeqRef = useRef(0);
  const [value, setValue] = useState("");
  const [tokens, setTokens] = useState<DraftToken[]>([]);
  // The real, LLM-backed word/group glossing (app.translate.llm_translate.
  // tag_draft via /translate/gloss-spans) for one specific past value of
  // the draft - fetched lazily (see ensureGlossSpans), not on a typing
  // timer, since unlike `tokens` above this is a real LLM call. `trimmedKey`
  // is that snapshot's trimmed text - trailing/leading whitespace changes
  // since the fetch don't invalidate it (see glossCacheValid below), but
  // any other content change does.
  const [glossCache, setGlossCache] = useState<{ trimmedKey: string; spans: DraftSpan[] } | null>(null);
  const [glossState, setGlossState] = useState<"idle" | "loading" | "error">("idle");
  const glossSeqRef = useRef(0);
  // The exact trimmed text a gloss-spans request is currently in flight
  // for - lets a second interaction against the SAME stale text (e.g.
  // hovering a different word before the first fetch returns) skip firing
  // a duplicate request, while a hover against text that's since changed
  // still fires its own.
  const glossPendingKeyRef = useRef<string | null>(null);
  // The raw character offset last hovered/tapped - re-resolved on every
  // render against whichever hoverable-range set is authoritative right
  // now (see openSpan below), rather than a boundary-keyed id, since a
  // naively-hovered single word can later resolve to a wider real
  // multi-word group once ensureGlossSpans's fetch completes.
  const [openOffset, setOpenOffset] = useState<number | null>(null);
  // Per-word-instance memory of the dual Spanish/English toggle (see
  // DraftSpan.alternate_gloss), keyed by that span's own start offset -
  // reset whenever a fresh gloss-spans fetch lands (a genuinely new
  // instance, offsets may no longer mean the same thing).
  const [altToggles, setAltToggles] = useState<Record<number, boolean>>({});
  // The help button's coaching result for the current draft - cleared
  // whenever the draft text changes from typing/replacing a word
  // (handleValueChange below), same as the old draft-translation preview
  // it replaced. NOT cleared by picking one of its own options (see
  // handleSelectCoachOption) - the whole point of #9 is that the popover
  // (and this cached result) survives a pick, so the learner can compare
  // a different option afterward.
  const [coachResult, setCoachResult] = useState<{
    meaning: string;
    feedback: string;
    options: CoachOption[];
  } | null>(null);
  const [coachState, setCoachState] = useState<"idle" | "loading" | "error">("idle");
  // True while the SAME streamed /translate/coach call's second phase
  // (each option's own English translation + word-by-word breakdown) is
  // still in flight - the options themselves (coachState above) are
  // already usable before this clears.
  const [coachTranslationsPending, setCoachTranslationsPending] = useState(false);
  // Which option the learner has most recently picked, if any - purely
  // for the popover's own "which one is this" highlight; applying it to
  // the draft happens immediately in handleSelectCoachOption, not here.
  const [selectedCoachOptionIndex, setSelectedCoachOptionIndex] = useState<number | null>(null);
  const coachAbortRef = useRef<AbortController | null>(null);
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
      })
      .catch(() => {});
  };

  useEffect(() => {
    if (!value.trim()) {
      setTokens([]);
      return;
    }
    const lastChar = value[value.length - 1];
    const delay = WORD_BOUNDARY_RE.test(lastChar) ? BOUNDARY_DEBOUNCE_MS : IDLE_DEBOUNCE_MS;
    const handle = setTimeout(() => fireTagInput(value), delay);
    return () => clearTimeout(handle);
  }, [value]);

  // The trimmed text glossCache's `spans` actually cover - an extra
  // leading/trailing space since that fetch is still the SAME instance
  // (no refetch needed), but any other content change makes it stale.
  const glossCacheValid = glossCache !== null && glossCache.trimmedKey === value.trim();

  // Fetches the real, LLM-backed word/group glossing for the CURRENT
  // draft, but only if it isn't already cached (or in flight) for this
  // exact trimmed text - called lazily, on hover/tap/click of a word,
  // never on a typing timer (see BOUNDARY_DEBOUNCE_MS's comment above).
  const ensureGlossSpans = () => {
    const trimmedKey = value.trim();
    if (!trimmedKey) return;
    if (glossCache?.trimmedKey === trimmedKey) return;
    if (glossPendingKeyRef.current === trimmedKey) return;
    glossPendingKeyRef.current = trimmedKey;
    setGlossState("loading");
    const seq = ++glossSeqRef.current;
    const textSnapshot = value;
    api
      .glossSpans({ text: textSnapshot })
      .then((res) => {
        if (seq !== glossSeqRef.current) return;
        glossPendingKeyRef.current = null;
        setGlossCache({ trimmedKey, spans: res.spans });
        setGlossState("idle");
        setAltToggles({});
      })
      .catch(() => {
        if (seq !== glossSeqRef.current) return;
        glossPendingKeyRef.current = null;
        setGlossState("error");
      });
  };

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
  // The hoverable-range set that actually backs the highlight overlay right
  // now: the real glossed spans once they're fetched and still fresh for
  // this exact text, or a naive one-range-per-cheap-token fallback
  // otherwise - so every word is hoverable (and tap/click-openable)
  // immediately, even before any LLM call has resolved.
  const hoverRanges: { start: number; end: number; span?: DraftSpan }[] = glossCacheValid
    ? glossCache!.spans
    : effectiveTokens;
  const segments = buildSegments(value, hoverRanges);
  // Suppressed while a real (multi-char) selection is active - a drag can
  // end with the pointer resting over a word, which would otherwise leave
  // this open at the same time as the phrase popover. Re-resolved every
  // render against whichever range set is authoritative (see hoverRanges
  // above) - this is what lets a naively-opened single word upgrade to a
  // wider real multi-word group once its gloss fetch lands, without the
  // popover's "open" identity ever changing.
  const openSpan =
    !phraseSelection && openOffset !== null && glossCacheValid
      ? glossCache!.spans.find((s) => openOffset >= s.start && openOffset < s.end)
      : undefined;
  const openPending = !phraseSelection && openOffset !== null && !glossCacheValid;
  const popoverState: "none" | "loading" | "error" | "ready" = phraseSelection
    ? "none"
    : openOffset === null
      ? "none"
      : openPending
        ? glossState === "error"
          ? "error"
          : "loading"
        : openSpan
          ? "ready"
          : "none";

  // Shared base for anything that changes the draft text - just the value
  // itself and clearing a stale phrase selection. Used directly (bypassing
  // handleValueChange's coach-clearing below) only by
  // handleSelectCoachOption, since picking one of the coach's OWN options
  // must not make its own popover disappear.
  const applyValue = (next: string) => {
    setValue(next);
    setPhraseSelection(null);
  };

  const handleValueChange = (next: string) => {
    applyValue(next);
    // Typing or a word-candidate replace means any previously fetched
    // coaching result is now of stale text, so drop it (and stop
    // streaming one that's still in flight).
    coachAbortRef.current?.abort();
    setCoachResult(null);
    setCoachState("idle");
    setCoachTranslationsPending(false);
    setSelectedCoachOptionIndex(null);
  };

  const handleReplace = (span: DraftSpan, translation: string) => {
    const next = value.slice(0, span.start) + translation + value.slice(span.end);
    handleValueChange(next);
    setOpenOffset(null);
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
  const handleWordMouseDown = (e: React.MouseEvent<HTMLSpanElement>, seg: Segment) => {
    if ((e.target as HTMLElement).closest(".word-candidates-popover")) return;
    e.preventDefault();
    const textarea = textareaRef.current;
    if (!textarea) return;
    const pos = charOffsetAtPoint(e.clientX, e.clientY) ?? seg.start;
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
    if (openOffset !== null) setOpenOffset(null);
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
    const seg = segments.find((s) => s.hoverable && offset >= s.start && offset <= s.end);
    if (!seg) return;
    setShowCoach(false);
    setOpenOffset(seg.start);
    ensureGlossSpans();
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
      setOpenOffset(null);
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
    glossSeqRef.current += 1; // same, for any outstanding gloss-spans request
    glossPendingKeyRef.current = null;
    setTokens([]);
    setGlossCache(null);
    setGlossState("idle");
    setOpenOffset(null);
    setAltToggles({});
  };

  // The help button's draft coaching - click-toggled rather than hover-
  // triggered (a hover-only trigger never fires on a touch device at all,
  // and the old globe button's only path to showing up on mobile was an
  // iOS "ghost hover" on a first tap, which left a second tap with nothing
  // left to do - no click handler ever existed for it to toggle). Reads
  // the streamed /translate/coach SSE response as it arrives: the "core"
  // event (feedback + the options themselves) lands first and is shown
  // immediately; a "translations" event for the SAME underlying LLM call
  // fills in each option's own English translation + word-by-word
  // breakdown a bit later (see api.coachDraftStream's docstring). Always
  // starts a fresh fetch (unlike the old cache-until-cleared version) -
  // callers that want the cached result left alone just don't call this;
  // see handleToggleCoach below for the one place that still caches
  // across opens of the SAME unchanged draft.
  const fetchCoach = () => {
    if (!value.trim()) return;
    coachAbortRef.current?.abort();
    const controller = new AbortController();
    coachAbortRef.current = controller;
    setCoachResult(null);
    setCoachState("loading");
    setCoachTranslationsPending(false);
    setSelectedCoachOptionIndex(null);

    (async () => {
      try {
        for await (const event of api.coachDraftStream({ text: value, session_id: sessionId }, controller.signal)) {
          if (controller.signal.aborted) return;
          if (event.type === "core") {
            setCoachResult({ meaning: event.data.meaning, feedback: event.data.feedback, options: event.data.options });
            setCoachState("idle");
            setCoachTranslationsPending(true);
          } else if (event.type === "translations") {
            setCoachResult((prev) => (prev ? { ...prev, options: event.data.options } : prev));
            setCoachTranslationsPending(false);
          } else if (event.type === "error") {
            setCoachState("error");
            setCoachTranslationsPending(false);
          }
        }
      } catch {
        if (!controller.signal.aborted) setCoachState("error");
      } finally {
        if (!controller.signal.aborted) setCoachTranslationsPending(false);
      }
    })();
  };

  // Picking one of the coach's own suggestions must NOT close/clear the
  // popover (see #9) - the learner can keep comparing other options
  // afterward, so this bypasses handleValueChange's coach-clearing
  // entirely (applyValue only) and just records which option is now
  // applied. When that option's own translations have already arrived,
  // seeds the chat input's lazy gloss cache directly from them (same
  // DraftSpan shape /translate/gloss-spans returns, already matched
  // against this exact spanish text server-side) - hovering a word in
  // the now-adopted suggestion shows its real translation immediately,
  // no extra fetch. If they haven't arrived yet, leaves the gloss cache
  // alone; the normal lazy fetch (ensureGlossSpans) takes over on the
  // first hover, same as any other edit.
  const handleSelectCoachOption = (option: CoachOption, index: number) => {
    applyValue(option.spanish);
    setSelectedCoachOptionIndex(index);
    if (option.english || option.spans.length > 0) {
      setGlossCache({ trimmedKey: option.spanish.trim(), spans: option.spans });
      setAltToggles({});
    }
    textareaRef.current?.focus();
  };

  const handleRegenerateCoach = () => fetchCoach();

  const closeCoach = () => {
    setShowCoach(false);
    coachAbortRef.current?.abort();
    textareaRef.current?.focus();
  };

  const handleToggleCoach = () => {
    if (!value.trim()) return;
    const next = !showCoach;
    setShowCoach(next);
    if (next) {
      // Reopening on the SAME draft that's already been coached (and
      // wasn't cleared by an edit since) just shows the cached result -
      // no need to re-run the whole streamed call. handleRegenerateCoach
      // is the explicit way to force a fresh one.
      if (!coachResult && coachState !== "loading") fetchCoach();
      setOpenOffset(null);
      setPhraseSelection(null);
    }
    // This button is tabIndex-focusable (for keyboard/a11y use), which
    // blurs the textarea on tap - checking a draft shouldn't cost the
    // keyboard when you're about to go right back to editing.
    textareaRef.current?.focus();
  };

  useEffect(() => {
    return () => coachAbortRef.current?.abort();
  }, []);

  return (
    <div className="chat-input">
      <div className="chat-input__editor">
        <div className="chat-input__highlight" aria-hidden="true" ref={highlightRef}>
          {segments.map((seg, i) =>
            seg.hoverable ? (
              <span
                key={i}
                className={`chat-input__hoverable${
                  openOffset !== null && openOffset >= seg.start && openOffset < seg.end
                    ? " chat-input__hoverable--active"
                    : ""
                }`}
                data-start={seg.start}
                data-end={seg.end}
                onMouseEnter={() => {
                  cancelHoverClose();
                  setOpenOffset(seg.start);
                  setShowCoach(false);
                  ensureGlossSpans();
                }}
                onMouseLeave={scheduleHoverClose}
                onMouseDown={(e) => handleWordMouseDown(e, seg)}
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
        {coachState === "loading" ? "⏳" : "💡"}
      </span>
      <button type="button" onClick={handleSend} disabled={!value.trim()}>
        Send
      </button>
      {/* Docked flush against the input bar's own top edge (see
          DockedPopover) rather than floating next to whatever word/button
          triggered it - a single, predictable, always-on-screen spot
          regardless of where in the draft that word or selection sits. */}
      {popoverState !== "none" && (
        <DockedPopover onMouseEnter={cancelHoverClose} onMouseLeave={scheduleHoverClose}>
          <WordCandidatesPopover
            loading={popoverState === "loading"}
            candidates={
              popoverState === "error"
                ? [{ translation: "(translation failed)" }]
                : openSpan
                  ? openSpan.candidates
                  : []
            }
            clickable={popoverState === "ready" && !!openSpan?.clickable}
            onSelect={(translation) => openSpan && handleReplace(openSpan, translation)}
            alternateGloss={openSpan?.alternate_gloss}
            showAlternate={openSpan ? !!altToggles[openSpan.start] : false}
            onToggleAlternate={() => {
              if (!openSpan) return;
              const key = openSpan.start;
              setAltToggles((prev) => ({ ...prev, [key]: !prev[key] }));
            }}
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
      {showCoach && (coachResult || coachState === "loading" || coachState === "error") && (
        <DockedPopover position="below">
          <CoachPopover
            meaning={coachResult?.meaning ?? ""}
            feedback={coachResult?.feedback ?? ""}
            options={coachResult?.options ?? []}
            loading={coachState === "loading"}
            error={coachState === "error"}
            translationsPending={coachTranslationsPending}
            selectedIndex={selectedCoachOptionIndex}
            onSelect={handleSelectCoachOption}
            onClose={closeCoach}
            onRegenerate={handleRegenerateCoach}
          />
        </DockedPopover>
      )}
    </div>
  );
}
