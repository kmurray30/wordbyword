import { useEffect, useRef } from "react";
import type { RefObject } from "react";

// Keeps the chat pinned to its newest line - on opening the app, as replies
// stream in, as your draft grows, and when the keyboard shrinks the screen -
// unless you've scrolled up to read something, in which case it leaves you
// there until you come back to the bottom.
const NEAR_BOTTOM_PX = 120;

export function useChatAutoScroll(scrollRef: RefObject<HTMLElement | null>, contentRef: RefObject<HTMLElement | null>) {
  const stickRef = useRef(true);

  useEffect(() => {
    const scroller = scrollRef.current;
    const content = contentRef.current;
    if (!scroller || !content) return;
    const toBottom = () => {
      if (stickRef.current) scroller.scrollTop = scroller.scrollHeight;
    };
    const onScroll = () => {
      stickRef.current = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < NEAR_BOTTOM_PX;
    };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    const observer = new ResizeObserver(toBottom);
    observer.observe(scroller);
    observer.observe(content);
    toBottom();
    return () => {
      scroller.removeEventListener("scroll", onScroll);
      observer.disconnect();
    };
  }, [scrollRef, contentRef]);

  return {
    // Jump to the bottom now and stay there (sending a message).
    toBottom: () => {
      stickRef.current = true;
      const scroller = scrollRef.current;
      if (scroller) scroller.scrollTop = scroller.scrollHeight;
    },
    // Leave the view where it is while something opens up in place (a
    // message's coach conversation unfolding below it) - until you scroll
    // back to the bottom or send.
    hold: () => {
      stickRef.current = false;
    },
  };
}
