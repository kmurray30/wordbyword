import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { TokenAnnotation } from "../api/client";
import { joinTokens } from "../lib/spacing";
import { WordToken } from "./WordToken";
import { TranslatePopover } from "./TranslatePopover";
import "./ChatMessage.css";

export interface DisplayMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  tokens?: TokenAnnotation[];
}

// How long an agent word can go un-hovered before we count that as passive
// recognition (a small familiarity boost) rather than "still pending".
const RESOLUTION_DELAY_MS = 6000;

export function ChatMessage({ message }: { message: DisplayMessage }) {
  const resolvedLemmas = useRef<Set<string>>(new Set());
  const [translation, setTranslation] = useState<string | null>(null);
  const [translating, setTranslating] = useState(false);
  const audioUrlRef = useRef<string | null>(null);
  const [speakState, setSpeakState] = useState<"idle" | "loading" | "error">("idle");

  useEffect(() => {
    return () => {
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (message.role !== "assistant" || !message.tokens) return;
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

  const handleTranslateAll = () => {
    if (translation !== null || translating) return;
    setTranslating(true);
    api
      .translateText({
        text: message.text,
        source_lang: message.role === "assistant" ? "es" : "en",
        target_lang: message.role === "assistant" ? "en" : "es",
      })
      .then((res) => setTranslation(res.translation))
      .catch(() => setTranslation("(translation failed)"))
      .finally(() => setTranslating(false));
  };

  const handleSpeak = () => {
    if (speakState === "loading") return;
    if (audioUrlRef.current) {
      new Audio(audioUrlRef.current).play().catch(() => setSpeakState("error"));
      return;
    }
    setSpeakState("loading");
    api
      .speak({ text: message.text, language: "es" })
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        audioUrlRef.current = url;
        setSpeakState("idle");
        new Audio(url).play().catch(() => setSpeakState("error"));
      })
      .catch(() => setSpeakState("error"));
  };

  const spaced = message.tokens ? joinTokens(message.tokens.map((t) => t.surface)) : null;

  return (
    <div className={`chat-message chat-message--${message.role}`}>
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
        <span className="chat-message__translate-all" onMouseEnter={handleTranslateAll}>
          🌐
          {translation !== null && <TranslatePopover candidates={[{ translation }]} />}
        </span>
        {message.role === "assistant" && (
          <button
            type="button"
            className={`chat-message__speak chat-message__speak--${speakState}`}
            onClick={handleSpeak}
            disabled={speakState === "loading"}
            aria-label="Play pronunciation"
            title={speakState === "error" ? "Couldn't play audio - try again" : "Play pronunciation"}
          >
            {speakState === "loading" ? "⏳" : "🔊"}
          </button>
        )}
      </div>
    </div>
  );
}
