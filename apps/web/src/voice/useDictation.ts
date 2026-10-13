import { useCallback, useRef } from "react";
import type { VoiceTranscriptEvent } from "../api/client";
import { LANGUAGES, getLanguage } from "../lib/language";
import { playEarcon } from "./earcons";
import { stopSpeech } from "./speech";
import { useMicStream } from "./useMicStream";

interface UseDictationOptions {
  sessionId: string;
  // Called with the full text the box should show: whatever was there when
  // dictation started, plus everything heard so far (settled words plus the
  // still-being-spoken tail).
  onText: (text: string) => void;
}

// Japanese and Chinese run words together without spaces.
function join(...parts: string[]): string {
  return parts
    .map((p) => p.trim())
    .filter(Boolean)
    .join(LANGUAGES[getLanguage()].spaced ? " " : "");
}

// Voice input as plain dictation: words stream into the chat input as the
// learner speaks, and nothing else happens until they press Done - a pause
// is just a pause. done() waits for the recognizer's last words and returns
// the final text, which the caller then submits exactly like typed text.
export function useDictation({ sessionId, onText }: UseDictationOptions) {
  const prefixRef = useRef("");
  const finalsRef = useRef<string[]>([]);
  const interimRef = useRef("");
  const onTextRef = useRef(onText);
  onTextRef.current = onText;

  const current = () => join(prefixRef.current, ...finalsRef.current, interimRef.current);

  const onEvent = useCallback((event: VoiceTranscriptEvent) => {
    if (event.type === "interim") {
      interimRef.current = event.text;
    } else if (event.type === "final") {
      if (event.text.trim()) finalsRef.current.push(event.text);
      interimRef.current = "";
    } else {
      return;
    }
    onTextRef.current(current());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const mic = useMicStream({ sessionId, onEvent });
  const micRef = useRef(mic);
  micRef.current = mic;

  const start = useCallback(async (existingText: string) => {
    stopSpeech();
    prefixRef.current = existingText;
    finalsRef.current = [];
    interimRef.current = "";
    await micRef.current.start();
    playEarcon(micRef.current.audioContext(), "listening");
  }, []);

  // Ends dictation and resolves with the box's final text (a word still
  // being spoken when Done was pressed is kept rather than dropped).
  const done = useCallback(async (): Promise<string> => {
    playEarcon(micRef.current.audioContext(), "heard");
    await micRef.current.finish();
    const text = current();
    interimRef.current = "";
    onTextRef.current(text);
    return text;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cancel = useCallback(() => {
    micRef.current.stop();
  }, []);

  return {
    // "listening" while words stream in, "finishing" between Done and the
    // last words arriving.
    state: mic.state,
    error: mic.error,
    active: mic.state === "starting" || mic.state === "listening" || mic.state === "finishing",
    start,
    done,
    cancel,
  };
}
