import { useEffect, useRef, useState } from "react";

// iOS keeps a backgrounded home-screen app (or Safari tab) alive and
// resumes it later WITHOUT reloading - so after a deploy the phone can keep
// running the old build against the new backend for hours. Whenever the
// app comes back to the foreground (and every so often while it's open),
// compare this build's version with the deployed /version.json. If they
// differ: reload right away when `canReloadNow()` says nothing would be
// lost, otherwise report `updateReady` so the app can offer a button.
const CHECK_EVERY_MS = 10 * 60 * 1000;

export function useAppUpdate(canReloadNow: () => boolean): { updateReady: boolean; reload: () => void } {
  const [updateReady, setUpdateReady] = useState(false);
  const canReloadRef = useRef(canReloadNow);
  useEffect(() => {
    canReloadRef.current = canReloadNow;
  });

  useEffect(() => {
    if (import.meta.env.DEV) return;
    let checking = false;
    const check = async () => {
      if (checking || document.visibilityState !== "visible") return;
      checking = true;
      try {
        const res = await fetch(`/version.json?t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) return;
        const { version } = (await res.json()) as { version?: string };
        if (!version || version === __APP_VERSION__) return;
        if (canReloadRef.current()) window.location.reload();
        else setUpdateReady(true);
      } catch {
        // offline or mid-deploy - try again next time
      } finally {
        checking = false;
      }
    };
    const onVisible = () => {
      if (document.visibilityState === "visible") void check();
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    window.addEventListener("pageshow", onVisible);
    const timer = window.setInterval(() => void check(), CHECK_EVERY_MS);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
      window.removeEventListener("pageshow", onVisible);
      window.clearInterval(timer);
    };
  }, []);

  return { updateReady, reload: () => window.location.reload() };
}
