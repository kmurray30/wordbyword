import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import "./AudioButton.css";

interface AudioButtonProps {
  text: string;
  language: string;
  voice?: string;
  label?: string;
}

// How long the button will sit on its loading spinner before giving up and
// showing an error, regardless of what api.speak() itself does. api.speak()
// already has its own ~50s abort, but that guarantee lives in a setTimeout
// too - a backgrounded/throttled tab can stall browser timers well past
// their nominal delay (observed live: a "stuck on the hourglass" report
// with no matching backend error, consistent with the abort simply firing
// very late). A second, independent timer here means this button always
// recovers into a clickable, retryable state within a bounded time even if
// that one is delayed or something else about the fetch never settles. Kept
// a bit above api.speak()'s own timeout so that one gets first crack at
// producing the real error.
const HARD_TIMEOUT_MS = 55_000;

// Generalizes what used to be ChatMessage's one-off speak button so every
// translation row (not just the agent's raw Spanish message) can play its
// own audio - caches the last-fetched blob per (language, voice, text) so
// replaying doesn't re-hit the TTS API.
export function AudioButton({ text, language, voice, label = "Play audio" }: AudioButtonProps) {
  const audioUrlRef = useRef<string | null>(null);
  const audioKeyRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);
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
    const requestId = ++requestIdRef.current;
    const hardTimeout = setTimeout(() => {
      if (requestIdRef.current === requestId) setState("error");
    }, HARD_TIMEOUT_MS);
    api
      .speak({ text, language, voice })
      .then((blob) => {
        if (requestIdRef.current !== requestId) return;
        if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
        const url = URL.createObjectURL(blob);
        audioUrlRef.current = url;
        audioKeyRef.current = key;
        setState("idle");
        new Audio(url).play().catch(() => setState("error"));
      })
      .catch(() => {
        if (requestIdRef.current === requestId) setState("error");
      })
      .finally(() => clearTimeout(hardTimeout));
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
