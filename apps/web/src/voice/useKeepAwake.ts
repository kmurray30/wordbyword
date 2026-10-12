import { useCallback, useEffect, useRef } from "react";
import { silentWav } from "./audioQueue";

// How voice mode survives the phone's screen turning off. Two strategies,
// picked at runtime, because iOS treats the two ways of opening this app
// differently (see the on-device spike, public/voice-spike.html):
//
// "safari-tab": a page in a plain Safari tab keeps its microphone and audio
//   session alive across lock AS LONG AS something is playing - so a
//   near-silent looping <audio> element is kept playing for the whole
//   session, with Media Session metadata so the lock screen shows what's
//   going on (and its pause button ends the session cleanly).
//
// "dim": a home-screen ("Add to Home Screen") web app loses the mic the
//   moment the screen locks, and nothing a web page does can change that.
//   The best available fallback is to keep the screen ON via the Screen
//   Wake Lock API and let the learner turn the brightness down - the panel
//   renders a black overlay for this mode.
export type KeepAwakeMode = "safari-tab" | "dim";

function isStandalone(): boolean {
  return typeof navigator !== "undefined" && (navigator as { standalone?: boolean }).standalone === true;
}

type WakeLockSentinelLike = { release: () => Promise<void> };

export function useKeepAwake({ onPauseRequested }: { onPauseRequested: () => void }) {
  const mode: KeepAwakeMode = isStandalone() ? "dim" : "safari-tab";
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lockRef = useRef<WakeLockSentinelLike | null>(null);
  const activeRef = useRef(false);
  const onPauseRef = useRef(onPauseRequested);
  useEffect(() => {
    onPauseRef.current = onPauseRequested;
  }, [onPauseRequested]);

  const requestLock = useCallback(async () => {
    const wakeLock = (navigator as { wakeLock?: { request: (type: "screen") => Promise<WakeLockSentinelLike> } }).wakeLock;
    if (!wakeLock) return;
    try {
      lockRef.current = await wakeLock.request("screen");
    } catch {
      // Denied/unsupported here - nothing else to fall back to.
    }
  }, []);

  // Call from inside the user gesture that starts voice mode (the audio
  // element can only start playing from a gesture).
  const start = useCallback(() => {
    if (activeRef.current) return;
    activeRef.current = true;
    if (mode === "safari-tab") {
      const el = new Audio(silentWav(1));
      el.loop = true;
      el.setAttribute("playsinline", "");
      void el.play().catch(() => {});
      audioRef.current = el;
      if ("mediaSession" in navigator) {
        navigator.mediaSession.metadata = new MediaMetadata({ title: "wordbyword", artist: "voice conversation - listening" });
        try {
          navigator.mediaSession.setActionHandler("pause", () => onPauseRef.current());
          navigator.mediaSession.setActionHandler("play", () => {});
        } catch {
          // older WebKit without these actions
        }
      }
    } else {
      void requestLock();
    }
  }, [mode, requestLock]);

  const stop = useCallback(() => {
    activeRef.current = false;
    const el = audioRef.current;
    audioRef.current = null;
    if (el) {
      el.pause();
      el.removeAttribute("src");
      el.load();
    }
    if ("mediaSession" in navigator) {
      try {
        navigator.mediaSession.setActionHandler("pause", null);
        navigator.mediaSession.setActionHandler("play", null);
      } catch {
        // ignore
      }
      navigator.mediaSession.metadata = null;
    }
    const lock = lockRef.current;
    lockRef.current = null;
    void lock?.release().catch(() => {});
  }, []);

  // A wake lock is dropped whenever the page is hidden; re-request on
  // return so dim mode keeps working after a glance at another app.
  useEffect(() => {
    const onVisibility = () => {
      if (!document.hidden && activeRef.current && mode === "dim" && !lockRef.current) void requestLock();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [mode, requestLock]);

  useEffect(() => stop, [stop]);

  return { mode, start, stop };
}
