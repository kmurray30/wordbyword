import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { DraftSpan, TokenAnnotation } from "../api/client";
import { joinTokens } from "../lib/spacing";
import { diffRawWords, pickDiffReference, tokenizeSegments } from "../lib/wordDiff";
import { DockedPopover } from "./DockedPopover";
import { TranslatePopover } from "./TranslatePopover";
import { WordCandidatesPopover } from "./WordCandidatesPopover";
import { WordToken } from "./WordToken";
import { AudioButton } from "./AudioButton";
import { TranslationRow } from "./TranslationRow";
import "./ChatMessage.css";

const TRANSLATION_FAILED = "(translation failed)";
// Grace period before closing a word's popover on mouseleave - it's
// docked flush to the bottom of the whole bubble (see DockedPopover)
// rather than floating right next to the hovered word, so the pointer
// has to travel there, leaving the word's own bounds along the way.
const HOVER_CLOSE_DELAY_MS = 200;

export interface DisplayMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  tokens?: TokenAnnotation[];
  // For a fresh assistant message, the whole-sentence translation from the
  // exact same LLM call that produced tokens[].gloss (see /chat/turn and
  // app.translate.llm_translate.gloss_reply) - guaranteed to agree with
  // those glosses on word sense, unlike fetching translateText separately.
  // Empty/absent for a history-hydrated message (not persisted) or if that
  // call failed; either falls back to fetching it the old way on demand.
  translation?: string;
  // True for messages hydrated from GET /chat/history on page load, as
  // opposed to ones just generated this session. Passive-exposure credit
  // and the eager translate-on-mount fetch already happened for real the
  // first time a message was shown - replaying them on every reload would
  // re-award familiarity for words you've seen many times before and
  // re-fire an LLM translation call per historical message, for nothing.
  fromHistory?: boolean;
  // Only set once the backend has assigned a real id to a freshly-sent
  // USER message (see App.tsx's handleSend, which patches this in once the
  // turn's "done" event carries ChatTurnResponse.user_message_id). Used
  // only to persist this message's live-fetched interpretation back to the
  // server (see the persist effect below) - deliberately a separate field
  // from `id` (the stable local id used as this component's React key) so
  // learning the real id doesn't remount the component and lose state.
  serverId?: number;
  // Hydrated from GET /chat/history for a USER message (see ChatMessage.
  // native_text/target_text on the backend) - lets this component show its
  // two translation rows instantly instead of "Translating…" on every
  // reload. Empty/absent for a row that predates this persistence feature,
  // or whose original /translate/interpret call failed.
  native?: string;
  target?: string;
}

// How long an agent word can go un-hovered before we count that as passive
// recognition (a small familiarity boost) rather than "still pending".
const RESOLUTION_DELAY_MS = 6000;

export function ChatMessage({ message, voice }: { message: DisplayMessage; voice?: string }) {
  const resolvedLemmas = useRef<Set<string>>(new Set());

  // Which word's translation popover is open, if any - at most one per
  // message, docked flush to the bottom of this message's own bubble
  // (see the DockedPopover render below) rather than each word rendering
  // its own floating popover. Hover takes precedence for display (so
  // momentarily hovering a different word while one is pinned previews
  // that word instead), falling back to whatever's pinned.
  const [hoveredWordIndex, setHoveredWordIndex] = useState<number | null>(null);
  const [pinnedWordIndex, setPinnedWordIndex] = useState<number | null>(null);
  const openWordIndex = hoveredWordIndex ?? pinnedWordIndex;
  const hoverCloseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelHoverClose = () => {
    if (hoverCloseTimerRef.current !== null) {
      clearTimeout(hoverCloseTimerRef.current);
      hoverCloseTimerRef.current = null;
    }
  };
  const scheduleHoverClose = () => {
    cancelHoverClose();
    hoverCloseTimerRef.current = setTimeout(() => {
      hoverCloseTimerRef.current = null;
      setHoveredWordIndex(null);
    }, HOVER_CLOSE_DELAY_MS);
  };
  useEffect(() => {
    return () => {
      if (hoverCloseTimerRef.current !== null) clearTimeout(hoverCloseTimerRef.current);
    };
  }, []);

  // Hovering the message previews its translation (fetched eagerly below,
  // so it's normally instant); the toggle button also pins it open on
  // click, for touch/keyboard users and anyone who wants to read it without
  // holding the hover (e.g. to use an audio button).
  // The translation rows live in normal document flow directly under the
  // bubble - not an absolutely-positioned popover - so there's no gap for
  // the pointer to cross between them; hover can be driven safely off the
  // whole column without the old "vanishes before you can reach it" bug
  // that hover-based popovers elsewhere in the app had to work around.
  const [isHovering, setIsHovering] = useState(false);
  const [isPinned, setIsPinned] = useState(false);
  const showTranslation = isHovering || isPinned;

  // Assistant: single ES->EN translation row.
  const [assistantTranslation, setAssistantTranslation] = useState<string | null>(null);
  const [assistantTranslating, setAssistantTranslating] = useState(false);

  // User: two rows - a corrected/interpreted English restatement (in case
  // the learner typed a mix of English/Spanish, or made mistakes in
  // either) and its Spanish translation.
  const [userNative, setUserNative] = useState<string | null>(null);
  const [userTarget, setUserTarget] = useState<string | null>(null);
  const [userInterpreting, setUserInterpreting] = useState(false);

  // Per-word/group glossing for the learner's OWN sent message - same
  // backend call (tag_draft via /translate/gloss-spans) the chat input's
  // own hover-translate uses on a still-being-typed draft, reused here
  // read-only on the final, already-sent text. Lazy (fetched once on the
  // first word click, not eagerly for every message) since it's a real
  // LLM call, same convention as the input's own ensureGlossSpans.
  const [userSpans, setUserSpans] = useState<DraftSpan[] | null>(null);
  const [userSpansLoading, setUserSpansLoading] = useState(false);
  // The raw character offset of the clicked word, re-resolved against
  // userSpans on every render - mirrors ChatInput's openOffset: a word
  // clicked before userSpans has loaded can still resolve to a wider
  // multi-word group once it does, without re-clicking.
  const [openUserOffset, setOpenUserOffset] = useState<number | null>(null);

  const ensureUserSpans = () => {
    if (userSpans !== null || userSpansLoading) return;
    setUserSpansLoading(true);
    api
      .glossSpans({ text: message.text })
      .then((res) => setUserSpans(res.spans))
      .catch(() => setUserSpans([]))
      .finally(() => setUserSpansLoading(false));
  };

  const openUserSpan =
    openUserOffset !== null ? userSpans?.find((s) => openUserOffset >= s.start && openUserOffset < s.end) : undefined;

  useEffect(() => {
    if (message.role !== "assistant" || !message.tokens || message.fromHistory) return;
    const trackedLemmas = [...new Set(message.tokens.filter((t) => t.gloss).map((t) => t.lemma))];

    const timer = setTimeout(() => {
      for (const lemma of trackedLemmas) {
        if (resolvedLemmas.current.has(lemma)) continue;
        resolvedLemmas.current.add(lemma);
        api.rewardEvent({ event_type: "wordSeenNoHover", lemma }).catch(() => {});
      }
    }, RESOLUTION_DELAY_MS);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [message.id]);

  const handleHover = (lemma: string) => {
    if (resolvedLemmas.current.has(lemma)) return;
    resolvedLemmas.current.add(lemma);
    api.rewardEvent({ event_type: "wordHovered", lemma }).catch(() => {});
  };

  const fetchAssistantTranslation = () => {
    if (assistantTranslation !== null || assistantTranslating) return;
    if (message.translation) {
      setAssistantTranslation(message.translation);
      return;
    }
    setAssistantTranslating(true);
    api
      .translateText({ text: message.text, source_lang: "es", target_lang: "en" })
      .then((res) => setAssistantTranslation(res.translation))
      .catch(() => setAssistantTranslation("(translation failed)"))
      .finally(() => setAssistantTranslating(false));
  };

  const fetchUserInterpretation = () => {
    if (userNative !== null || userInterpreting) return;
    setUserInterpreting(true);
    api
      .interpretInput({ text: message.text })
      .then((res) => {
        setUserNative(res.native);
        setUserTarget(res.target);
      })
      .catch(() => {
        setUserNative(TRANSLATION_FAILED);
        setUserTarget(TRANSLATION_FAILED);
      })
      .finally(() => setUserInterpreting(false));
  };

  // A history-hydrated user message already carries its interpretation
  // from GET /chat/history (see DisplayMessage.native/target) - show it
  // immediately instead of leaving userNative/userTarget null, which would
  // otherwise only resolve once the learner actually hovers/pins the
  // translation (fetchUserInterpretation is on-demand, not eager, for a
  // history row - see the effect below). Absent/empty just means this row
  // predates the persistence feature or its original call failed; the
  // on-demand fetch still covers that case.
  useEffect(() => {
    if (!message.fromHistory || message.role !== "user") return;
    if (message.native && message.target) {
      setUserNative(message.native);
      setUserTarget(message.target);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [message.id]);

  // Persists a freshly-sent user message's live-fetched interpretation back
  // to the server, once BOTH pieces are known: the backend's real id for
  // this message (message.serverId, patched in by App.tsx once the turn's
  // "done" event arrives) and the interpretation itself (userNative/
  // userTarget, from fetchUserInterpretation above) - whichever resolves
  // last triggers this. Guarded by a ref (not just userNative !== null,
  // which fetchUserInterpretation's own guard already covers) so a
  // mid-flight state change can't fire this twice. Skipped entirely for a
  // history row (already persisted, nothing new to save) or a failed
  // translation (nothing worth persisting).
  const persistedInterpretationRef = useRef(false);
  useEffect(() => {
    if (persistedInterpretationRef.current) return;
    if (message.role !== "user" || message.fromHistory) return;
    if (message.serverId === undefined) return;
    if (userNative === null || userTarget === null) return;
    if (userNative === TRANSLATION_FAILED || userTarget === TRANSLATION_FAILED) return;
    persistedInterpretationRef.current = true;
    api.saveUserInterpretation({ message_id: message.serverId, native: userNative, target: userTarget }).catch(() => {});
  }, [message.role, message.fromHistory, message.serverId, userNative, userTarget]);

  // Kick the translation off in the background as soon as the message is
  // shown, rather than waiting for the toggle - the LLM-backed translation
  // is slow enough that pre-fetching means it's usually ready by the time
  // anyone actually opens it.
  useEffect(() => {
    if (message.fromHistory) return;
    if (message.role === "assistant") fetchAssistantTranslation();
    else fetchUserInterpretation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [message.id]);

  const fetchTranslation = () => {
    if (message.role === "assistant") fetchAssistantTranslation();
    else fetchUserInterpretation();
  };

  const handleMouseEnter = () => {
    setIsHovering(true);
    fetchTranslation();
  };

  const handleTogglePin = () => {
    setIsPinned((pinned) => {
      const next = !pinned;
      // A real click always lands while the mouse is still physically
      // over the button - onMouseEnter already fired and won't fire
      // again until the pointer actually leaves and comes back, so
      // showTranslation = isHovering || isPinned stays true from the
      // lingering hover alone, masking the toggle: a second click looked
      // like it did nothing. An explicit close should win over that.
      if (!next) setIsHovering(false);
      return next;
    });
    fetchTranslation();
  };

  // Only assistant replies get per-token annotations from the backend -
  // history rows for a user message come back with tokens: [] (truthy as
  // an array, so this has to check length, not just presence) and should
  // fall through to plain/diffed text below.
  const hasTokens = !!message.tokens && message.tokens.length > 0;
  const spaced = hasTokens ? joinTokens(message.tokens!.map((t) => t.surface)) : null;

  // One segment per word/non-word run of the raw typed message, each
  // carrying its own character offsets (for matching against userSpans
  // once that lazy fetch resolves - see openUserSpan) and whether it's
  // "flagged": looks different from both corrected versions, a
  // lightweight "this looked off" cue, not a full spell checker. Flagging
  // is skipped (flagged stays false throughout) until both corrections are
  // in, or if either failed, or for an accent-only difference (see
  // wordDiff.ts) - offsets are still computed regardless, since every word
  // is clickable from the start independent of whether flagging data has
  // arrived yet.
  const userWordSegments = useMemo(() => {
    if (message.role !== "user") return null;

    const rawSegments = tokenizeSegments(message.text);
    const rawWords = rawSegments.filter((s) => s.isWord).map((s) => s.text);

    let matched: boolean[] | null = null;
    if (
      rawWords.length > 0 &&
      userNative !== null &&
      userTarget !== null &&
      userNative !== TRANSLATION_FAILED &&
      userTarget !== TRANSLATION_FAILED
    ) {
      const nativeWords = tokenizeSegments(userNative)
        .filter((s) => s.isWord)
        .map((s) => s.text);
      const targetWords = tokenizeSegments(userTarget)
        .filter((s) => s.isWord)
        .map((s) => s.text);
      const reference = pickDiffReference(rawWords, nativeWords, targetWords);
      matched = diffRawWords(rawWords, reference);
    }

    let cursor = 0;
    let wordIndex = 0;
    return rawSegments.map((seg) => {
      const start = cursor;
      const end = start + seg.text.length;
      cursor = end;
      if (!seg.isWord) return { text: seg.text, start, end, isWord: false, flagged: false };
      const flagged = matched ? !matched[wordIndex] : false;
      wordIndex++;
      return { text: seg.text, start, end, isWord: true, flagged };
    });
  }, [message.role, message.text, userNative, userTarget]);

  return (
    <div className={`chat-message chat-message--${message.role}`}>
      <div className="chat-message__column">
        <div className="chat-message__bubble">
          {hasTokens && spaced ? (
            message.tokens!.map((tok, i) => (
              <span key={i}>
                {spaced[i].spaceBefore && " "}
                <WordToken
                  surface={tok.surface}
                  gloss={tok.gloss}
                  open={openWordIndex === i}
                  onOpen={() => {
                    cancelHoverClose();
                    setHoveredWordIndex(i);
                    handleHover(tok.lemma);
                  }}
                  onClose={scheduleHoverClose}
                  onTogglePin={() => {
                    setPinnedWordIndex((cur) => (cur === i ? null : i));
                    handleHover(tok.lemma);
                  }}
                />
              </span>
            ))
          ) : userWordSegments ? (
            userWordSegments.map((seg, i) =>
              seg.isWord ? (
                <span
                  key={i}
                  className={`word-token${openUserOffset === seg.start ? " word-token--active" : ""}${
                    seg.flagged ? " chat-message__typo" : ""
                  }`}
                  title={seg.flagged ? "Looks different from the corrected version" : undefined}
                  onClick={() => {
                    ensureUserSpans();
                    setOpenUserOffset((cur) => (cur === seg.start ? null : seg.start));
                  }}
                >
                  {seg.text}
                </span>
              ) : (
                <span key={i}>{seg.text}</span>
              ),
            )
          ) : (
            message.text
          )}
          {openWordIndex !== null && message.tokens?.[openWordIndex] && (
            <DockedPopover position="right" onMouseEnter={cancelHoverClose} onMouseLeave={scheduleHoverClose}>
              <TranslatePopover
                candidates={[
                  {
                    translation: message.tokens[openWordIndex].gloss,
                    description: message.tokens[openWordIndex].note || undefined,
                  },
                ]}
                literal={message.tokens[openWordIndex].literal || undefined}
              />
            </DockedPopover>
          )}
          {openUserOffset !== null && (
            // Mirrors the assistant popover right above: nested directly in
            // the bubble, docked sideways rather than "below" - growing
            // vertically here would overlap the actions/translations rows
            // beneath it (absolutely-positioned content reserves no layout
            // space of its own - see Fix 1's history on this component).
            // "left" rather than "right" since a user bubble sits flush
            // against the right edge of the viewport - "right" would run
            // off-screen there.
            <DockedPopover position="left">
              <WordCandidatesPopover
                candidates={openUserSpan?.candidates ?? []}
                clickable={false}
                onSelect={() => {}}
                loading={userSpansLoading}
                literal={openUserSpan?.literal}
              />
            </DockedPopover>
          )}
        </div>
        <div className="chat-message__actions">
          <button
            type="button"
            className={`chat-message__translate-toggle${isPinned ? " chat-message__translate-toggle--active" : ""}`}
            onClick={handleTogglePin}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={() => setIsHovering(false)}
            aria-label="Pin translation open"
            aria-pressed={isPinned}
            title="Hover to preview - click to keep it open"
          >
            🌐
          </button>
          {message.role === "assistant" && (
            <AudioButton text={message.text} language="es" voice={voice} label="Play pronunciation" />
          )}
        </div>
        {showTranslation && (
          <div className="chat-message__translations">
            {message.role === "assistant" ? (
              <TranslationRow
                variant="assistant"
                label="EN"
                text={assistantTranslation}
                loading={assistantTranslating}
                language="en"
              />
            ) : (
              <>
                <TranslationRow
                  variant="user-native"
                  label="EN"
                  text={userNative}
                  loading={userInterpreting}
                  language="en"
                />
                <TranslationRow
                  variant="user-target"
                  label="ES"
                  text={userTarget}
                  loading={userInterpreting}
                  language="es"
                  voice={voice}
                />
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
