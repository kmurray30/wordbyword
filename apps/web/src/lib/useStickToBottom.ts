import { useLayoutEffect } from "react";
import type { RefObject } from "react";

// Keeps a scrolling log pinned to its newest line: whenever `deps` change
// (a new entry, more streamed text), scroll the element to the bottom.
export function useStickToBottom(ref: RefObject<HTMLElement | null>, deps: unknown[]): void {
  useLayoutEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
