import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { DraftSpan, ThreadEntry, TokenAnnotation } from "../api/client";
import { joinTokens } from "../lib/spacing";
import { diffRawWords, pickDiffReference, tokenizeSegments } from "../lib/wordDiff";
import { DockedPopover } from "./DockedPopover";
import { TranslatePopover } from "./TranslatePopover";
import { WordCandidatesPopover } from "./WordCandidatesPopover";
import { WordToken } from "./WordToken";
import { AudioButton } from "./AudioButton";
import { TranslationRow } from "./TranslationRow";
import { MessageThread } from "./MessageThread";
import type { LiveCoachReply } from "./MessageThread";
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
  // The side conversation about this message (see MessageThread): coach
  // checks behind a sent message, questions and the coach's replies.
  thread?: ThreadEntry[];
}

// How long an agent word can go un-hovered before we count that as passive
// recognition (a small familiarity boost) rather than "still pending".
const RESOLUTION_DELAY_MS = 6000;

export function ChatMessage({
  message,
  voice,
  threadOpen = false,
  onToggleThread,
  liveCoach = null,
}: {
  message: DisplayMessage;
  voice?: string;
  threadOpen?: boolean;
  onToggleThread?: () => void;
  // The coach's reply while it's still streaming, when it's about this
  // message.
  liveCoach?: LiveCoachReply | null;
}) {
  const resolvedLemmas = useRef<Set<string>>(new Set());
  // Checks count once each; a question and its answer count as one.
  const threadCount = (message.thread ?? []).filter((e) => e.kind !== "answer").length;

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

  // The 🌐 at the end of the bubble opens the whole message's translation
  // in the same kind of popup a word gets (fetched eagerly below, so it's
  // normally instant). One popup per message at a time.
  const [globeOpen, setGlobeOpen] = useState(false);
  const bubbleRef = useRef<HTMLDivElement | null>(null);

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

  // Set once a translation had to be fetched here (rather than arriving
  // saved with the message) - it then gets saved back (see below).
  const fetchedTranslationRef = useRef(false);

  // At most one translation request per message, however often it's asked.
  const translationRequestedRef = useRef(false);

  const fetchAssistantTranslation = () => {
    if (assistantTranslation !== null || translationRequestedRef.current) return;
    translationRequestedRef.current = true;
    setAssistantTranslating(true);
    api
      .translateText({ text: message.text, source_lang: "es", target_lang: "en" })
      .then((res) => {
        fetchedTranslationRef.current = true;
        setAssistantTranslation(res.translation);
      })
      .catch(() => setAssistantTranslation("(translation failed)"))
      .finally(() => setAssistantTranslating(false));
  };

  const fetchUserInterpretation = () => {
    if (userNative !== null || translationRequestedRef.current) return;
    translationRequestedRef.current = true;
    setUserInterpreting(true);
    api
      .interpretInput({ text: message.text })
      .then((res) => {
        fetchedTranslationRef.current = true;
        setUserNative(res.native);
        setUserTarget(res.target);
      })
      .catch(() => {
        setUserNative(TRANSLATION_FAILED);
        setUserTarget(TRANSLATION_FAILED);
      })
      .finally(() => setUserInterpreting(false));
  };

  // The translation is saved with the message (see DisplayMessage.native/
  // target/translation), so 🌐 normally opens instantly. A message without
  // one (sent before translations were saved, or its translation call
  // failed) fetches it now, in the background, and saves it back - once.
  useEffect(() => {
    if (message.role === "assistant") {
      const saved = message.translation || message.native;
      if (saved) setAssistantTranslation(saved);
      else fetchAssistantTranslation();
    } else if (message.native && message.target) {
      setUserNative(message.native);
      setUserTarget(message.target);
    } else {
      fetchUserInterpretation();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [message.id]);

  // Saves a translation fetched above, once both it and the message's
  // backend id are known (a just-sent user message only learns its id when
  // the turn finishes - see App.tsx's handleSend).
  const serverId = message.serverId ?? (message.id > 0 ? message.id : undefined);
  const savedTranslationRef = useRef(false);
  useEffect(() => {
    if (savedTranslationRef.current || !fetchedTranslationRef.current || serverId === undefined) return;
    const native = message.role === "assistant" ? assistantTranslation : userNative;
    const target = message.role === "assistant" ? "" : userTarget;
    if (!native || native === TRANSLATION_FAILED || target === null || target === TRANSLATION_FAILED) return;
    if (message.role === "user" && !target) return;
    savedTranslationRef.current = true;
    api.saveUserInterpretation({ message_id: serverId, native, target }).catch(() => {});
  }, [message.role, serverId, assistantTranslation, userNative, userTarget]);

  const closePopups = () => {
    cancelHoverClose();
    setHoveredWordIndex(null);
    setPinnedWordIndex(null);
    setOpenUserOffset(null);
    setGlobeOpen(false);
  };

  const handleGlobe = () => {
    if (globeOpen) {
      setGlobeOpen(false);
      return;
    }
    closePopups();
    setGlobeOpen(true);
    if (message.role === "assistant") fetchAssistantTranslation();
    else fetchUserInterpretation();
  };

  // A click anywhere closes this message's popup (see DockedPopover),
  // except on its own words and 🌐, which open/close/switch it themselves.
  const keepOpenFor = (target: Element) =>
    !!bubbleRef.current?.contains(target) && !!target.closest(".word-token, .chat-message__globe");

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
        <div className="chat-message__bubble" ref={bubbleRef}>
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
                    setGlobeOpen(false);
                    setHoveredWordIndex(i);
                    handleHover(tok.lemma);
                  }}
                  onClose={scheduleHoverClose}
                  onTogglePin={() => {
                    // Clicking the pinned word again closes it.
                    if (pinnedWordIndex === i) {
                      closePopups();
                      return;
                    }
                    setGlobeOpen(false);
                    setPinnedWordIndex(i);
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
                    setGlobeOpen(false);
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
          <button
            type="button"
            className={`chat-message__globe${globeOpen ? " chat-message__globe--active" : ""}`}
            onClick={handleGlobe}
            aria-label="Translate the whole message"
            aria-expanded={globeOpen}
            title="Translate the whole message"
          >
            🌐
          </button>
          {globeOpen && (
            <DockedPopover
              position={message.role === "assistant" ? "right" : "left"}
              onDismiss={() => setGlobeOpen(false)}
              keepOpenFor={keepOpenFor}
            >
              <div className="translate-popover chat-message__translation-popover" role="tooltip">
                {message.role === "assistant" ? (
                  <TranslationRow
                    variant="assistant"
                    label="EN"
                    text={assistantTranslation}
                    loading={assistantTranslating || assistantTranslation === null}
                    language="en"
                  />
                ) : (
                  <>
                    <TranslationRow
                      variant="user-native"
                      label="EN"
                      text={userNative}
                      loading={userInterpreting || userNative === null}
                      language="en"
                    />
                    <TranslationRow
                      variant="user-target"
                      label="ES"
                      text={userTarget}
                      loading={userInterpreting || userTarget === null}
                      language="es"
                      voice={voice}
                    />
                  </>
                )}
              </div>
            </DockedPopover>
          )}
          {openWordIndex !== null && message.tokens?.[openWordIndex] && (
            <DockedPopover
              key={openWordIndex}
              position="right"
              onMouseEnter={cancelHoverClose}
              onMouseLeave={scheduleHoverClose}
              onDismiss={closePopups}
              keepOpenFor={keepOpenFor}
            >
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
            <DockedPopover key={openUserOffset} position="left" onDismiss={closePopups} keepOpenFor={keepOpenFor}>
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
        {(message.role === "assistant" || threadCount > 0 || liveCoach) && (
          <div className="chat-message__actions">
            {message.role === "assistant" && (
              <AudioButton text={message.text} language="es" voice={voice} label="Play pronunciation" />
            )}
            {(threadCount > 0 || liveCoach) && (
              <button
                type="button"
                className={`chat-message__thread-toggle${threadOpen ? " chat-message__thread-toggle--active" : ""}`}
                onClick={onToggleThread}
                aria-expanded={threadOpen}
                title={threadOpen ? "Hide the side conversation" : "Show the side conversation about this message"}
              >
                💬 {threadCount + (liveCoach ? 1 : 0)}
              </button>
            )}
          </div>
        )}
        {threadOpen && (threadCount > 0 || liveCoach) && (
          <MessageThread entries={message.thread ?? []} live={liveCoach} />
        )}
      </div>
    </div>
  );
}
