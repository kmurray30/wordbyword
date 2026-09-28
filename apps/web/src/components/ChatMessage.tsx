import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { TokenAnnotation } from "../api/client";
import { joinTokens } from "../lib/spacing";
import { WordToken } from "./WordToken";
import { AudioButton } from "./AudioButton";
import { TranslationRow } from "./TranslationRow";
import "./ChatMessage.css";

export interface DisplayMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  tokens?: TokenAnnotation[];
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

  // Toggle, not hover: a hover-triggered popover that stays open forever
  // once fetched (the old behavior) is confusing - a click you control is
  // clearer, and it also gives history-hydrated messages an obvious way to
  // fetch a translation lazily since they skip the eager prefetch below.
  const [showTranslation, setShowTranslation] = useState(false);

  // Assistant: single ES->EN translation row.
  const [assistantTranslation, setAssistantTranslation] = useState<string | null>(null);
  const [assistantTranslating, setAssistantTranslating] = useState(false);

  // User: two rows - a corrected/interpreted English restatement (in case
  // the learner typed a mix of English/Spanish, or made mistakes in
  // either) and its Spanish translation.
  const [userNative, setUserNative] = useState<string | null>(null);
  const [userTarget, setUserTarget] = useState<string | null>(null);
  const [userInterpreting, setUserInterpreting] = useState(false);
  const [userTargetRetranslating, setUserTargetRetranslating] = useState(false);

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
        setUserNative("(translation failed)");
        setUserTarget("(translation failed)");
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

  const handleToggleTranslation = () => {
    const next = !showTranslation;
    setShowTranslation(next);
    if (!next) return;
    if (message.role === "assistant") fetchAssistantTranslation();
    else fetchUserInterpretation();
  };

  // The learner corrected the guessed English restatement - re-derive the
  // Spanish row from that corrected text rather than the original (possibly
  // mixed-language/misspelled) input, so a fixed English side also fixes
  // the Spanish side and its audio.
  const handleEditNative = (newNative: string) => {
    setUserNative(newNative);
    setUserTargetRetranslating(true);
    api
      .translateText({ text: newNative, source_lang: "en", target_lang: "es" })
      .then((res) => setUserTarget(res.translation))
      .catch(() => setUserTarget("(translation failed)"))
      .finally(() => setUserTargetRetranslating(false));
  };

  const spaced = message.tokens ? joinTokens(message.tokens.map((t) => t.surface)) : null;

  return (
    <div className={`chat-message chat-message--${message.role}`}>
      <div className="chat-message__column">
        <div className="chat-message__bubble">
          {message.tokens && spaced ? (
            message.tokens.map((tok, i) => (
              <span key={i}>
                {spaced[i].spaceBefore && " "}
                <WordToken surface={tok.surface} gloss={tok.gloss} isNew={tok.is_new} onHover={() => handleHover(tok.lemma)} />
              </span>
            ))
          ) : (
            message.text
          )}
          <button
            type="button"
            className={`chat-message__translate-toggle${showTranslation ? " chat-message__translate-toggle--active" : ""}`}
            onClick={handleToggleTranslation}
            aria-label="Show translation"
            aria-pressed={showTranslation}
            title="Show translation"
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
                  editable
                  onSave={handleEditNative}
                />
                <TranslationRow
                  variant="user-target"
                  label="ES"
                  text={userTarget}
                  loading={userInterpreting || userTargetRetranslating}
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
