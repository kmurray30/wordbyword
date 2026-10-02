import { useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

const MARGIN = 10;

interface FloatingPopoverProps {
  // Anchor point in viewport coordinates (e.g. from getBoundingClientRect())
  // - the popover is centered horizontally on this point, same as before.
  anchor: { top: number; left: number };
  direction: "up" | "down";
  className?: string;
  children: ReactNode;
}

// Portals `children` to document.body, positioned near `anchor` and
// horizontally (and vertically) clamped to stay fully on screen -
// centering purely on the anchor point (the old CSS left:50%/
// transform:translateX(-50%) approach, with no viewport awareness) pushed
// a real chunk of the popover off-screen for any word near a mobile
// viewport's edge - on a ~360-390px-wide phone that's most of them, since
// these popovers run 260-420px wide. Measures its own rendered size after
// every render (not just on mount) since content varies - cycling
// candidates, toggling the Spanish/English column - and a stale
// measurement would mis-clamp a popover that just changed size.
export function FloatingPopover({ anchor, direction, className, children }: FloatingPopoverProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [style, setStyle] = useState<{ top: number; left: number } | null>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { offsetWidth: width, offsetHeight: height } = el;
    const left = Math.min(Math.max(anchor.left - width / 2, MARGIN), window.innerWidth - width - MARGIN);
    const rawTop = direction === "down" ? anchor.top : anchor.top - height;
    const top = Math.min(Math.max(rawTop, MARGIN), window.innerHeight - height - MARGIN);
    // Bail out (return the same object) once the measurement has converged -
    // calling setStyle with a fresh object on every run, unconditionally,
    // re-triggers this effect every render with no way to stabilize, which
    // React detects as an infinite loop ("Maximum update depth exceeded")
    // and crashes the whole popover before it ever becomes visible.
    setStyle((prev) => (prev && prev.top === top && prev.left === left ? prev : { top, left }));
  });

  return createPortal(
    <div
      ref={ref}
      className={className}
      style={{
        position: "fixed",
        zIndex: 1000,
        top: style?.top ?? anchor.top,
        left: style?.left ?? anchor.left,
        // Hidden until the first real measurement lands, so it never
        // flashes at the old, unclamped (anchor-centered) position first.
        visibility: style ? "visible" : "hidden",
      }}
    >
      {children}
    </div>,
    document.body,
  );
}
