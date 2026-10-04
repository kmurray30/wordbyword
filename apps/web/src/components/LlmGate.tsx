import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../api/client";
import "./LlmGate.css";

const POLL_INTERVAL_MS = 3000;
// Only show the "waking up" screen if the very first check hasn't resolved
// by this point - a backend that's already warm (the common case) answers
// within a normal network round trip, well under this, so the real app
// just renders directly with no flash at all. A genuine cold start still
// shows the message promptly; this only hides the flash for the fast case.
const SHOW_DELAY_MS = 300;
// After this many consecutive not-ready polls, stop assuming it's just a
// slow wake-up and show a different message instead. The backend's only
// dependency now is the OpenAI API key being configured (no self-hosted
// model to load into memory) - a cold Railway start resolves within a
// handful of polls, so staying not-ready this long means something's
// actually wrong (missing/invalid key, backend down), not "still loading".
const GIVE_UP_AFTER_ATTEMPTS = 20;

// Blocks rendering of the real app until the chat backend is confirmed
// ready. Nearly every interactive feature here (chat, translation, hover
// glosses) calls through to it, and each one fails in its own confusing
// way if it's mid-cold-start (Railway's serverless sleep on the backend
// process itself) - holding the whole UI back until it's actually ready,
// rather than letting the first real interaction be the one to discover
// that, avoids that entire class of bug. Polling GET /health/llm is itself
// the wake-up trigger for the sleep case (Railway wakes a serverless
// service on any inbound request), so no separate "start it up" call is
// needed.
export function LlmGate({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [showLoading, setShowLoading] = useState(false);
  const [gaveUp, setGaveUp] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;

    const showTimer = setTimeout(() => {
      if (!cancelled) setShowLoading(true);
    }, SHOW_DELAY_MS);

    const poll = () => {
      attempts += 1;
      api
        .healthLlm()
        .then((res) => {
          if (cancelled) return;
          if (res.ready) {
            setReady(true);
          } else if (attempts >= GIVE_UP_AFTER_ATTEMPTS) {
            setGaveUp(true);
          } else {
            pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
          }
        })
        .catch(() => {
          if (cancelled) return;
          if (attempts >= GIVE_UP_AFTER_ATTEMPTS) {
            setGaveUp(true);
          } else {
            pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
          }
        });
    };
    poll();

    return () => {
      cancelled = true;
      clearTimeout(pollTimer);
      clearTimeout(showTimer);
    };
  }, []);

  if (!ready) {
    // Still within the grace window - render nothing rather than flash the
    // loading screen for what's usually about to resolve immediately.
    if (!showLoading) return null;
    if (gaveUp) {
      return (
        <div className="llm-gate">
          <p className="llm-gate__title">Still can't reach the chat backend.</p>
          <p className="llm-gate__hint">
            It may be misconfigured (e.g. a missing or invalid API key) rather than just slow to wake up. Try
            reloading in a bit, or check the backend's logs.
          </p>
        </div>
      );
    }
    return (
      <div className="llm-gate">
        <div className="llm-gate__spinner" aria-hidden="true" />
        <p className="llm-gate__title">Waking up the backend…</p>
        <p className="llm-gate__hint">This can take up to a minute after a period of inactivity.</p>
      </div>
    );
  }

  return <>{children}</>;
}
