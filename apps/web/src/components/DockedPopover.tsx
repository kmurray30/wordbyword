import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import "./DockedPopover.css";

// How close to the screen's edge a "right"/"left" popover may reach before
// it drops under its bubble instead - matches FloatingPopover's MARGIN.
const EDGE_MARGIN = 8;
// The gap between a bubble and a popover beside it (padding in the CSS).
const SIDE_GAP = 8;

interface DockedPopoverProps {
  children: ReactNode;
  // "above" (default) docks flush against the top edge of the relative
  // parent - used for the input's word/phrase lookups, which need to stay
  // near the draft being edited. "below" docks flush against its bottom
  // edge instead - used for the help button's suggestion, which reads
  // better out of the way of the draft itself. "right" docks flush
  // against the right edge instead of growing the layout vertically -
  // used for an assistant chat bubble's per-word gloss (those bubbles are
  // left-aligned, with open room to their right), which otherwise stacked
  // underneath the bubble's own full-message translation rows. "left" is
  // the mirror of "right" - for a user bubble's own per-word gloss, since
  // those bubbles sit flush against the right edge of the viewport;
  // docking "right" there would run off-screen.
  // "inline" sits in normal flow instead of floating - for the message
  // box, which is pinned to the bottom of the screen with no room below
  // it: the popup opens right under the box and pushes the bars up.
  position?: "above" | "below" | "right" | "left" | "inline";
  // For a hover-triggered popover (the word-tap one, which still opens on
  // desktop hover): docking it away from the hovered word/bubble means the
  // mouse now has to travel there, almost always leaving the original
  // hover target's bounds along the way and firing its onMouseLeave before
  // a click inside the popover ever lands. Forwarding hover here lets the
  // caller keep its state open while the pointer is over the popover
  // itself, regardless of what's happening back at the original target.
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  // Any click/tap anywhere closes the popover - on the popover itself too,
  // except on a control inside it (a candidate to pick, an audio button,
  // "next option"), which does its own thing. `keepOpenFor` exempts
  // elements outside it that manage the popover themselves (the word that
  // toggles it, the box being typed in).
  onDismiss?: () => void;
  keepOpenFor?: (target: Element) => boolean;
}

// For popovers that should sit flush against a fixed anchor (the chat
// input bar, a message bubble) rather than floating next to whatever
// word/button triggered them - unlike FloatingPopover's anchor-and-clamp
// math (still used elsewhere), this needs no JS measurement at all:
// positioned with plain CSS relative to the nearest `position: relative`
// ancestor, it inherits that ancestor's own on-screen position for free -
// including however a mobile keyboard has already repositioned it -
// instead of recomputing viewport bounds. Docks in the same place every
// time rather than hugging whichever word/button triggered it.
export function DockedPopover({
  children,
  position = "above",
  onMouseEnter,
  onMouseLeave,
  onDismiss,
  keepOpenFor,
}: DockedPopoverProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const dismissRef = useRef({ onDismiss, keepOpenFor });
  useEffect(() => {
    dismissRef.current = { onDismiss, keepOpenFor };
  });
  const dismissible = !!onDismiss;
  useEffect(() => {
    if (!dismissible) return;
    // Capture phase, added after the click that opened the popover has
    // already passed through - so only LATER clicks count.
    const handleClick = (e: MouseEvent) => {
      const target = e.target;
      if (!(target instanceof Element)) return;
      const inside = !!wrapRef.current?.contains(target);
      if (inside && target.closest("button, a, input, textarea, select")) return;
      if (!inside && dismissRef.current.keepOpenFor?.(target)) return;
      dismissRef.current.onDismiss?.();
    };
    document.addEventListener("click", handleClick, true);
    return () => document.removeEventListener("click", handleClick, true);
  }, [dismissible]);
  // "right"/"left" sit beside their anchor (a chat bubble) when there's
  // room; when the popup is wider than the space beside the bubble, it
  // drops to just under the bubble instead, flush against it and lined up
  // with the bubble's own edge. Re-checked on every render, since the
  // popup can grow after it opens ("…" turning into a long translation).
  const [dropped, setDropped] = useState(false);
  // "below" grows the page's own scrollable overflow downward from a
  // container that's often already flush with the bottom of the viewport
  // (the input bar) - without this, opening it leaves the popover rendered
  // entirely past the fold with no visual sign that the page can even
  // scroll there, let alone that it should. Only runs once per mount (the
  // popover only mounts once per open), so this doesn't fight the user if
  // they scroll away afterward.
  useEffect(() => {
    if (position === "below" || position === "right" || position === "left") wrapRef.current?.scrollIntoView({ block: "nearest" });
  }, [position]);

  useLayoutEffect(() => {
    if (position !== "right" && position !== "left") return;
    const wrap = wrapRef.current;
    const anchor = wrap?.parentElement;
    const box = wrap?.firstElementChild as HTMLElement | null;
    if (!wrap || !anchor || !box) return;
    const a = anchor.getBoundingClientRect();
    const room = position === "right" ? window.innerWidth - EDGE_MARGIN - a.right : a.left - EDGE_MARGIN;
    // Dropped, the box is at least as wide as it was beside the bubble, so
    // it never flips back and forth. (Callers remount it for new content -
    // a different word - so that gets a fresh look at the side first.)
    const drop = box.offsetWidth + SIDE_GAP > room;
    setDropped((prev) => (prev === drop ? prev : drop));
  });

  // Once dropped under the bubble, bring it into view like "below" does.
  useEffect(() => {
    if (dropped) wrapRef.current?.scrollIntoView({ block: "nearest" });
  }, [dropped]);

  return (
    <div
      ref={wrapRef}
      className={`docked-popover-wrap docked-popover-wrap--${position}${dropped ? " docked-popover-wrap--dropped" : ""}`}
    >
      <div className="docked-popover" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
        {children}
      </div>
    </div>
  );
}
