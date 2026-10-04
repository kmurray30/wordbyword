import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { TokenAnnotation } from "../api/client";
import { joinTokens } from "../lib/spacing";
import { diffRawWords, pickDiffReference, tokenizeSegments } from "../lib/wordDiff";
import { DockedPopover } from "./DockedPopover";
import { TranslatePopover } from "./TranslatePopover";
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

  // Underlines words in the raw bubble that don't appear in either
  // corrected version - a lightweight "this looked off" cue, not a full
  // spell checker. Skipped until both translations are in (or if either
  // failed), and for accent-only differences (see wordDiff.ts).
  const userTypoSegments = useMemo(() => {
    if (message.role !== "user") return null;
    if (userNative === null || userTarget === null) return null;
    if (userNative === TRANSLATION_FAILED || userTarget === TRANSLATION_FAILED) return null;

    const rawSegments = tokenizeSegments(message.text);
    const rawWords = rawSegments.filter((s) => s.isWord).map((s) => s.text);
    if (rawWords.length === 0) return null;

    const nativeWords = tokenizeSegments(userNative)
      .filter((s) => s.isWord)
      .map((s) => s.text);
    const targetWords = tokenizeSegments(userTarget)
      .filter((s) => s.isWord)
      .map((s) => s.text);
    const reference = pickDiffReference(rawWords, nativeWords, targetWords);
    const matched = diffRawWords(rawWords, reference);

    let wordIndex = 0;
    return rawSegments.map((seg) => {
      if (!seg.isWord) return { text: seg.text, flagged: false };
      const flagged = !matched[wordIndex];
      wordIndex++;
      return { text: seg.text, flagged };
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
          ) : userTypoSegments ? (
            userTypoSegments.map((seg, i) =>
              seg.flagged ? (
                <span key={i} className="chat-message__typo" title="Looks different from the corrected version">
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
