import type { ReactNode } from "react";
import "./DockedPopover.css";

interface DockedPopoverProps {
  children: ReactNode;
  // For a hover-triggered popover (the word-tap one, which still opens on
  // desktop hover): docking it flush against the input bar means the mouse
  // now has to travel from the hovered word to wherever the input bar is,
  // which almost always means leaving the word's own bounds along the way
  // and firing its onMouseLeave before the click on a candidate inside the
  // popover ever lands. Forwarding hover here lets the caller keep its
  // state open while the pointer is over the popover itself, regardless of
  // what's happening back at the original word.
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

// For the chat input's own popovers (word/phrase lookup, the help button) -
// unlike FloatingPopover's anchor-and-clamp math (still used for the chat
// bubbles' word popover), this needs no JS measurement at all: positioned
// with plain CSS flush against the input bar's own top edge (bottom: 100%
// of .chat-input, which must be position: relative), it inherits the input
// bar's own on-screen position for free - including however a mobile
// keyboard has already repositioned it - instead of recomputing viewport
// bounds and clamping/flipping a floating box. Docks in the same place
// every time rather than hugging whichever word/button triggered it.
export function DockedPopover({ children, onMouseEnter, onMouseLeave }: DockedPopoverProps) {
  return (
    <div className="docked-popover-wrap">
      <div className="docked-popover" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
        {children}
      </div>
    </div>
  );
}
