import { useEffect, useState } from "react";

// Keeps the app pinned to the part of the screen you can actually see.
// On iOS the on-screen keyboard doesn't shrink the page; it scrolls the
// page up under the keyboard instead, which pushed the header off the top.
// The visual viewport is the visible part: publish its height and offset
// as CSS variables so .app (position: fixed, see App.css) always fills
// exactly that, with only the chat scrolling inside it.
// Also callable directly: while the keyboard slides, code following it
// re-syncs every frame instead of waiting for the next viewport event.
export function syncVisibleViewport(): void {
  const vv = window.visualViewport;
  const root = document.documentElement;
  const height = `${vv ? vv.height : window.innerHeight}px`;
  const top = `${vv ? vv.offsetTop : 0}px`;
  if (root.style.getPropertyValue("--vv-height") !== height) root.style.setProperty("--vv-height", height);
  if (root.style.getPropertyValue("--vv-top") !== top) root.style.setProperty("--vv-top", top);
}

export function trackVisibleViewport(): void {
  const vv = window.visualViewport;
  const update = syncVisibleViewport;
  update();
  vv?.addEventListener("resize", update);
  vv?.addEventListener("scroll", update);
  window.addEventListener("resize", update);
}

// Whether the on-screen keyboard is up: the visible height has dropped well
// below the tallest it's been (reset when the phone turns).
export function useKeyboardUp(): boolean {
  const [up, setUp] = useState(false);
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    let tallest = vv.height;
    const check = () => {
      tallest = Math.max(tallest, vv.height);
      setUp(vv.height < tallest - 120);
    };
    const reset = () => {
      tallest = vv.height;
      check();
    };
    vv.addEventListener("resize", check);
    window.addEventListener("orientationchange", reset);
    return () => {
      vv.removeEventListener("resize", check);
      window.removeEventListener("orientationchange", reset);
    };
  }, []);
  return up;
}
