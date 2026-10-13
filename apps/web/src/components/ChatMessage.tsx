import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { MessageGlosses, ThreadEntry, TokenAnnotation } from "../api/client";
import { joinTokens } from "../lib/spacing";
import { diffRawWords, pickDiffReference, tokenizeSegments } from "../lib/wordDiff";
import { DockedPopover } from "./DockedPopover";
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
  // What each word (and multi-word phrase) means - see MessageGlosses.
  // Saved with the message; null/absent until made (fetched below).
  glosses?: MessageGlosses | null;
}

// Where each token sits in the message text (tokens are its words and
// punctuation in order, so a forward scan finds them).
function tokenOffsets(text: string, tokens: TokenAnnotation[]): { start: number; end: number }[] {
  const lower = text.toLowerCase();
  let cursor = 0;
  return tokens.map((tok) => {
    const at = lower.indexOf(tok.surface.toLowerCase(), cursor);
    const start = at === -1 ? cursor : at;
    const end = at === -1 ? cursor : at + tok.surface.length;
    cursor = end;
    return { start, end };
  });
}

// The tapped word's own meaning, beside the bubble - and, under it, any
// note on its form (a verb's infinitive, person and tense).
function WordGlossPopup({ gloss, grammar, pending }: { gloss: string; grammar: string; pending: boolean }) {
  return (
    <div className="chat-message__popup chat-message__word-popover" role="tooltip" lang="en">
      <div className="chat-message__word-gloss">{gloss || (pending ? "…" : "(no translation)")}</div>
      {grammar && <div className="chat-message__word-grammar">{grammar}</div>}
    </div>
  );
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
  latest = false,
  savedWords,
  onSaveWord,
  onRemoveWord,
  onAskCoach,
  canAskCoach = true,
  sessionId = "",
}: {
  // Talk to the coach about this message (its 💬 conversation).
  onAskCoach?: (question: string) => void;
  // False while a reply is still coming in, or before a just-sent message
  // has its id.
  canAskCoach?: boolean;
  // For dictating into this message's coach box.
  sessionId?: string;
  message: DisplayMessage;
  // The newest reply in the chat: its popups open upward, so they don't
  // run off the bottom of the conversation.
  latest?: boolean;
  // Words the learner has saved (lowercased), and how to save/remove one:
  // tap a word to see its meaning, tap it again for "add word?", and once
  // more to add it (or "remove?" / remove, if it's saved already).
  savedWords?: Set<string>;
  onSaveWord?: (entry: { word: string; gloss: string; grammar: string; sentence: string; messageId?: number }) => void;
  onRemoveWord?: (word: string) => void;
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

  // The tapped (pinned) or hovered word, by its character offset in the
  // message. Its meaning shows beside the bubble; if it's part of a phrase,
  // the phrase's meaning shows above the bubble.
  const [hoverOffset, setHoverOffset] = useState<number | null>(null);
  const [pinnedOffset, setPinnedOffset] = useState<number | null>(null);
  const activeOffset = hoverOffset ?? pinnedOffset;
  // The pinned word is showing "add word?"/"remove?"; then, briefly, what
  // just happened.
  const [askOffset, setAskOffset] = useState<number | null>(null);
  const [flash, setFlash] = useState<{ offset: number; text: string; removed: boolean } | null>(null);
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (flashTimerRef.current) clearTimeout(flashTimerRef.current);
  }, []);
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
      setHoverOffset(null);
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

  // Word/phrase glosses: saved with the message. One that doesn't have
  // them yet (your own messages, older replies) gets them once, in the
  // background, and the server saves them for next time.
  const [fetchedGlosses, setFetchedGlosses] = useState<MessageGlosses | null>(null);
  const [glossesFailed, setGlossesFailed] = useState(false);
  const glosses = message.glosses ?? fetchedGlosses;
  const glossesRequestedRef = useRef(false);
  const glossServerId = message.serverId ?? (message.id > 0 ? message.id : undefined);
  useEffect(() => {
    if (message.glosses || glossServerId === undefined || glossesRequestedRef.current) return;
    glossesRequestedRef.current = true;
    api
      .messageGlosses({ message_id: glossServerId })
      .then((g) => setFetchedGlosses({ words: g.words ?? [], phrases: g.phrases ?? [] }))
      .catch(() => setGlossesFailed(true));
  }, [message.glosses, glossServerId]);

  const wordAt = (offset: number | null) =>
    offset === null ? undefined : glosses?.words.find((w) => w.start <= offset && offset < w.end);
  const phraseAt = (offset: number | null) =>
    offset === null ? undefined : glosses?.phrases.find((p) => p.start <= offset && offset < p.end);
  const activePhrase = phraseAt(activeOffset);
  const inActivePhrase = (start: number, end: number) =>
    !!activePhrase && start >= activePhrase.start && end <= activePhrase.end;

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
        setAssistantTranslation(res.translation ?? "");
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
        setUserNative(res.native ?? "");
        setUserTarget(res.target ?? "");
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
    setHoverOffset(null);
    setPinnedOffset(null);
    setAskOffset(null);
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
  const offsets = useMemo(
    () => (hasTokens ? tokenOffsets(message.text, message.tokens!) : []),
    [hasTokens, message.text, message.tokens],
  );

  // A tap pins the word; tapping the pinned word again closes it.
  // First tap shows the word's meaning; tapping that same word again asks
  // "add word?" (or "remove?"), and once more does it.
  const wordText = (offset: number) => {
    const w = wordAt(offset);
    if (w) return message.text.slice(w.start, w.end);
    const i = offsets.findIndex((o) => o.start <= offset && offset < o.end);
    if (i !== -1) return message.tokens?.[i]?.surface ?? "";
    return userWordSegments?.find((seg) => seg.start === offset)?.text ?? "";
  };
  const isSaved = (offset: number) => !!savedWords?.has(wordText(offset).toLowerCase());
  const togglePinned = (offset: number) => {
    if (pinnedOffset !== offset) {
      setGlobeOpen(false);
      setAskOffset(null);
      setPinnedOffset(offset);
      return;
    }
    if (askOffset !== offset) {
      setAskOffset(offset);
      return;
    }
    const word = wordText(offset);
    setAskOffset(null);
    if (!word) return;
    const saved = isSaved(offset);
    if (saved) onRemoveWord?.(word);
    else
      onSaveWord?.({
        word,
        gloss: wordAt(offset)?.gloss ?? fallbackGloss(offset),
        grammar: wordAt(offset)?.grammar ?? "",
        sentence: message.text,
        messageId: glossServerId,
      });
    setFlash({ offset, text: saved ? "Removed" : "Added ✓", removed: saved });
    if (flashTimerRef.current) clearTimeout(flashTimerRef.current);
    flashTimerRef.current = setTimeout(() => setFlash(null), 1300);
  };
  // What floats above a word: the question, or what just happened.
  const badgeFor = (offset: number) => {
    if (flash?.offset === offset)
      return (
        <span className={`word-ask word-ask--done${flash.removed ? " word-ask--removed" : ""}`} role="status">
          {flash.text}
        </span>
      );
    if (askOffset === offset && pinnedOffset === offset)
      return (
        <span className="word-ask" role="status">
          {isSaved(offset) ? "remove?" : "add word?"}
          <span className="word-ask__hint">(tap again)</span>
        </span>
      );
    return null;
  };
  // The word being asked about turns green (add) or red (remove).
  const toneFor = (offset: number): "add" | "remove" | null =>
    askOffset === offset && pinnedOffset === offset ? (isSaved(offset) ? "remove" : "add") : null;

  // A reply saved before word glosses existed still has its per-token
  // gloss - shown until the new ones arrive.
  const fallbackGloss = (offset: number) => {
    const i = offsets.findIndex((o) => o.start <= offset && offset < o.end);
    return i === -1 ? "" : (message.tokens?.[i]?.gloss ?? "");
  };

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
            message.tokens!.map((tok, i) => {
              const { start, end } = offsets[i];
              // The space before a word inside the open phrase is part of
              // the phrase's highlight too.
              const spaceInPhrase = i > 0 && inActivePhrase(offsets[i - 1].start, end);
              return (
                <span key={i}>
                  {spaced[i].spaceBefore && (spaceInPhrase ? <span className="word-token--phrase"> </span> : " ")}
                  <WordToken
                    surface={tok.surface}
                    gloss={tok.gloss || wordAt(start)?.gloss || ""}
                    open={activeOffset !== null && start <= activeOffset && activeOffset < end}
                    inPhrase={inActivePhrase(start, end)}
                    badge={badgeFor(start)}
                    tone={toneFor(start)}
                    onOpen={() => {
                      cancelHoverClose();
                      setGlobeOpen(false);
                      setHoverOffset(start);
                      handleHover(tok.lemma);
                    }}
                    onClose={scheduleHoverClose}
                    onTogglePin={() => {
                      togglePinned(start);
                      handleHover(tok.lemma);
                    }}
                  />
                </span>
              );
            })
          ) : userWordSegments ? (
            userWordSegments.map((seg, i) =>
              seg.isWord ? (
                <span
                  key={i}
                  className={`word-token${
                    activeOffset !== null && seg.start <= activeOffset && activeOffset < seg.end ? " word-token--active" : ""
                  }${inActivePhrase(seg.start, seg.end) ? " word-token--phrase" : ""}${seg.flagged ? " chat-message__typo" : ""}${
                    toneFor(seg.start) ? ` word-token--${toneFor(seg.start)}` : ""
                  }`}
                  title={seg.flagged ? "Looks different from the corrected version" : undefined}
                  onClick={() => togglePinned(seg.start)}
                >
                  {seg.text}
                  {badgeFor(seg.start)}
                </span>
              ) : (
                <span key={i} className={inActivePhrase(seg.start, seg.end) ? "word-token--phrase" : undefined}>
                  {seg.text}
                </span>
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
              overflow={latest ? "over" : "under"}
              onDismiss={() => setGlobeOpen(false)}
              keepOpenFor={keepOpenFor}
            >
              <div className="chat-message__popup chat-message__translation-popover" role="tooltip">
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
          {activeOffset !== null && (
            <DockedPopover
              key={`word-${activeOffset}`}
              position={message.role === "assistant" ? "right" : "left"}
              overflow="wrap"
              growUp={latest}
              onMouseEnter={cancelHoverClose}
              onMouseLeave={scheduleHoverClose}
              onDismiss={closePopups}
              keepOpenFor={keepOpenFor}
            >
              <WordGlossPopup
                gloss={wordAt(activeOffset)?.gloss ?? fallbackGloss(activeOffset)}
                grammar={wordAt(activeOffset)?.grammar ?? ""}
                pending={!glosses && !glossesFailed}
              />
            </DockedPopover>
          )}
          {activePhrase && (
            <DockedPopover
              key={`phrase-${activePhrase.start}`}
              position="over"
              align={message.role === "assistant" ? "start" : "end"}
              onMouseEnter={cancelHoverClose}
              onMouseLeave={scheduleHoverClose}
              onDismiss={closePopups}
              keepOpenFor={keepOpenFor}
            >
              <div className="chat-message__popup chat-message__phrase-popover" role="tooltip">
                <div className="chat-message__phrase-gloss">{activePhrase.gloss}</div>
                {activePhrase.note && <div className="chat-message__phrase-note">{activePhrase.note}</div>}
              </div>
            </DockedPopover>
          )}
        </div>
        <div className="chat-message__actions">
          {message.role === "assistant" && (
            <AudioButton text={message.text} language="es" voice={voice} label="Play pronunciation" />
          )}
          <button
            type="button"
            className={`chat-message__thread-toggle${threadOpen ? " chat-message__thread-toggle--active" : ""}`}
            onClick={onToggleThread}
            aria-expanded={threadOpen}
            aria-label="Talk to the coach about this message"
            title="Talk to the coach about this message"
          >
            💬{threadCount + (liveCoach ? 1 : 0) > 0 ? ` ${threadCount + (liveCoach ? 1 : 0)}` : ""}
          </button>
        </div>
        {threadOpen && (
          <MessageThread
            entries={message.thread ?? []}
            live={liveCoach}
            sessionId={sessionId}
            canAsk={canAskCoach}
            onAsk={(question) => onAskCoach?.(question)}
          />
        )}
      </div>
    </div>
  );
}
