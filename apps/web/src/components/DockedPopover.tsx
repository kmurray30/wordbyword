import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import "./DockedPopover.css";

// How far off the viewport's own edge a "right"/"left" popover is allowed
// to sit before getting nudged back - matches FloatingPopover's own MARGIN.
const EDGE_MARGIN = 8;

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
  position?: "above" | "below" | "right" | "left";
  // For a hover-triggered popover (the word-tap one, which still opens on
  // desktop hover): docking it away from the hovered word/bubble means the
  // mouse now has to travel there, almost always leaving the original
  // hover target's bounds along the way and firing its onMouseLeave before
  // a click inside the popover ever lands. Forwarding hover here lets the
  // caller keep its state open while the pointer is over the popover
  // itself, regardless of what's happening back at the original target.
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
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
export function DockedPopover({ children, position = "above", onMouseEnter, onMouseLeave }: DockedPopoverProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  // Only "right"/"left" need this: they dock flush against their anchor's
  // own edge (see the CSS), and that anchor (a chat bubble) can sit
  // anywhere horizontally - a short bubble near either side of a narrow
  // mobile viewport leaves too little room for the popover to grow into
  // before running off-screen. "above"/"below" don't have this problem -
  // they're already full-width-with-margin relative to their anchor (see
  // DockedPopover.css), which never exceeds the viewport on its own.
  const [edgeShift, setEdgeShift] = useState(0);

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

  // Nudges a "right"/"left" popover back on screen if it would otherwise
  // run past the viewport's own left/right edge - e.g. a short reply's
  // bubble sitting close to the edge, combined with a long translation/
  // note, could dock a popover that's mostly off-screen (unreadable, and
  // the exact "should always fit on the mobile screen" bug report this
  // guards against) without this. Deliberately NOT measured via the
  // anchor's own position (no FloatingPopover-style portal+clamp here) -
  // this keeps the CSS-inherited vertical/side docking (including however
  // a mobile keyboard has already repositioned the anchor) and only
  // corrects the one axis that can actually go wrong. Runs on every
  // render, not just mount, since the popover's own width can change after
  // its content finishes loading (a short "…" placeholder growing into a
  // real, possibly long, translation) - the bail-out check below (only
  // call setState when the value actually changed) keeps that from
  // looping, same pattern FloatingPopover already uses for the same
  // reason.
  useLayoutEffect(() => {
    if (position !== "right" && position !== "left") {
      setEdgeShift((prev) => (prev === 0 ? prev : 0));
      return;
    }
    const el = wrapRef.current;
    if (!el) return;
    // getBoundingClientRect() reflects whatever `edgeShift` is ALREADY
    // applied as a transform right now - computing the new shift directly
    // from it (instead of from the un-shifted position) would measure an
    // already-corrected box, "fix" it again on top of that, overcorrect,
    // measure THAT as the new baseline next render, and so on - an
    // infinite oscillation (caught live: "Maximum update depth exceeded").
    // Subtracting the currently-applied shift first recovers the natural,
    // untransformed position every time, so each run's answer depends only
    // on the anchor's own (stable) position and this popover's own
    // (stable, once loaded) size - exactly the idempotent math
    // FloatingPopover's analogous clamp relies on for the same reason.
    const rect = el.getBoundingClientRect();
    const naturalLeft = rect.left - edgeShift;
    const naturalRight = rect.right - edgeShift;
    let shift = 0;
    if (naturalRight > window.innerWidth - EDGE_MARGIN) shift = window.innerWidth - EDGE_MARGIN - naturalRight;
    else if (naturalLeft < EDGE_MARGIN) shift = EDGE_MARGIN - naturalLeft;
    setEdgeShift((prev) => (prev === shift ? prev : shift));
  });

  return (
    <div
      ref={wrapRef}
      className={`docked-popover-wrap docked-popover-wrap--${position}`}
      style={edgeShift ? { transform: `translateX(${edgeShift}px)` } : undefined}
    >
      <div className="docked-popover" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
        {children}
      </div>
    </div>
  );
}
