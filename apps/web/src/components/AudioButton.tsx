import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import "./AudioButton.css";

interface AudioButtonProps {
  text: string;
  language: string;
  voice?: string;
  label?: string;
}

// Generalizes what used to be ChatMessage's one-off speak button so every
// translation row (not just the agent's raw Spanish message) can play its
// own audio - caches the last-fetched blob per (language, voice, text) so
// replaying doesn't re-hit the TTS API.
export function AudioButton({ text, language, voice, label = "Play audio" }: AudioButtonProps) {
  const audioUrlRef = useRef<string | null>(null);
  const audioKeyRef = useRef<string | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "error">("idle");

  useEffect(() => {
    return () => {
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  const handleClick = () => {
    if (state === "loading" || !text.trim()) return;
    const key = `${language}:${voice ?? ""}:${text}`;
    if (audioUrlRef.current && audioKeyRef.current === key) {
      new Audio(audioUrlRef.current).play().catch(() => setState("error"));
      return;
    }
    setState("loading");
    api
      .speak({ text, language, voice })
      .then((blob) => {
        if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
        const url = URL.createObjectURL(blob);
        audioUrlRef.current = url;
        audioKeyRef.current = key;
        setState("idle");
        new Audio(url).play().catch(() => setState("error"));
      })
      .catch(() => setState("error"));
  };

  return (
    <button
      type="button"
      className={`audio-button audio-button--${state}`}
      onClick={handleClick}
      disabled={state === "loading" || !text.trim()}
      aria-label={label}
      title={state === "error" ? "Couldn't play audio - try again" : label}
    >
      {state === "loading" ? "⏳" : "🔊"}
    </button>
  );
}
