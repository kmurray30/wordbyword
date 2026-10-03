import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import "./DockedPopover.css";

interface DockedPopoverProps {
  children: ReactNode;
  // "above" (default) docks flush against the top edge of the relative
  // parent - used for the input's word/phrase lookups, which need to stay
  // near the draft being edited. "below" docks flush against its bottom
  // edge instead - used for the help button's suggestion, which reads
  // better out of the way of the draft itself.
  position?: "above" | "below";
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

  // "below" grows the page's own scrollable overflow downward from a
  // container that's often already flush with the bottom of the viewport
  // (the input bar) - without this, opening it leaves the popover rendered
  // entirely past the fold with no visual sign that the page can even
  // scroll there, let alone that it should. Only runs once per mount (the
  // popover only mounts once per open), so this doesn't fight the user if
  // they scroll away afterward.
  useEffect(() => {
    if (position === "below") wrapRef.current?.scrollIntoView({ block: "nearest" });
  }, [position]);

  return (
    <div ref={wrapRef} className={`docked-popover-wrap docked-popover-wrap--${position}`}>
      <div className="docked-popover" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
        {children}
      </div>
    </div>
  );
}
