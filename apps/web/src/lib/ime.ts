import type { KeyboardEvent } from "react";

// A Japanese/Chinese/Korean input method is still composing: its Enter
// confirms the characters, it isn't the learner pressing Enter to send.
// (keyCode 229 is how Safari reports keys during composition.)
export function isComposing(e: KeyboardEvent): boolean {
  return e.nativeEvent.isComposing || e.keyCode === 229;
}
