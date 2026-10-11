import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { VoiceDialogTurn, VoiceTranscriptEvent, VoiceTranscriptWord } from "../api/client";
import { AudioQueue } from "./audioQueue";
import { takeCompleteSentences } from "./sentences";
import { useMicStream } from "./useMicStream";

export type VoiceState =
  | "idle"
  | "listening"
  | "thinking"
  | "speaking_coach"
  | "sending"
  | "speaking_partner"
  | "error";

interface UseVoiceSessionOptions {
  sessionId: string;
  // The app's normal send path (App.tsx's handleSend) - resolves with the
  // partner's reply once the turn is done, or null if it failed.
  onSend: (text: string) => Promise<{ text: string } | null>;
  partnerVoice?: string;
}

// The spoken-conversation state machine:
//
//   listening --(utterance_end)--> thinking --(coach stream)--> speaking_coach
//      ^                                                            |
//      |   verdict=revise / question answered / cancel              |
//      +<-----------------------------------------------------------+
//      |   verdict=ready, or "yes"/"that one" with a draft          |
//      +<--- speaking_partner <--- sending <-------------------------+
//
// Half-duplex: the mic is muted from the moment the learner's utterance
// ends until the last clip has finished playing, so the phone's speaker
// never gets transcribed as the learner. Everything is driven by mic/WS/
// audio events, not timers (iOS throttles timers with the screen off).
export function useVoiceSession({ sessionId, onSend, partnerVoice }: UseVoiceSessionOptions) {
  const [state, setState] = useState<VoiceState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [dialog, setDialog] = useState<VoiceDialogTurn[]>([]);
  const [coachSay, setCoachSay] = useState("");
  const [draft, setDraft] = useState("");
  const queueRef = useRef<AudioQueue | null>(null);
  if (!queueRef.current) queueRef.current = new AudioQueue();
  // Text settled by the recognizer since the last utterance boundary, plus
  // its per-word confidences - becomes one coach turn at utterance_end.
  const finalsRef = useRef<string[]>([]);
  const wordsRef = useRef<VoiceTranscriptWord[]>([]);
  const dialogRef = useRef<VoiceDialogTurn[]>([]);
  const busyRef = useRef(false);
  const runRef = useRef(0);
  const optionsRef = useRef({ onSend, partnerVoice });
  useEffect(() => {
    optionsRef.current = { onSend, partnerVoice };
  }, [onSend, partnerVoice]);

  const micEvent = useCallback((event: VoiceTranscriptEvent) => {
    if (event.type === "final") {
      finalsRef.current.push(event.text.trim());
      wordsRef.current.push(...event.words);
    } else if (event.type === "utterance_end") {
      const transcript = finalsRef.current.join(" ").trim();
      const words = wordsRef.current;
      finalsRef.current = [];
      wordsRef.current = [];
      if (transcript && !busyRef.current) void handleUtterance(transcript, words);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const mic = useMicStream({ sessionId, onEvent: micEvent });
  const micRef = useRef(mic);
  micRef.current = mic;

  const setDialogBoth = (next: VoiceDialogTurn[]) => {
    dialogRef.current = next;
    setDialog(next);
  };

  async function handleUtterance(transcript: string, words: VoiceTranscriptWord[]) {
    const queue = queueRef.current!;
    const run = ++runRef.current;
    const live = () => run === runRef.current;
    busyRef.current = true;
    micRef.current.setMuted(true);
    setState("thinking");
    setError(null);
    setCoachSay("");
    const dialogBefore = dialogRef.current;
    setDialogBoth([...dialogBefore, { role: "learner", text: transcript }]);

    let intent = "attempt";
    let verdict = "revise";
    let nextDraft = "";
    let say = "";
    let buffer = "";
    const speak = (sentence: string) => {
      if (!live()) return;
      setState("speaking_coach");
      void queue.enqueue(api.speak({ text: sentence, language: "en", role: "coach" })).catch(() => {
        // One failed clip shouldn't end the turn - the text is on screen.
      });
    };

    try {
      for await (const event of api.voiceCoachStream({
        transcript,
        words,
        coach_dialog: dialogBefore,
        session_id: sessionId,
      })) {
        if (!live()) return;
        if (event.type === "intent") intent = event.data.intent;
        else if (event.type === "verdict") verdict = event.data.verdict;
        else if (event.type === "draft") {
          nextDraft = event.data.draft;
          setDraft(nextDraft);
        } else if (event.type === "say_chunk") {
          buffer += event.data.delta;
          say += event.data.delta;
          setCoachSay(say);
          const { sentences, rest } = takeCompleteSentences(buffer);
          buffer = rest;
          sentences.forEach(speak);
        } else if (event.type === "say_complete") {
          say = event.data.say;
          setCoachSay(say);
          if (buffer.trim()) speak(buffer.trim());
          buffer = "";
        } else if (event.type === "error") {
          throw new Error(event.data.message);
        }
      }
      if (buffer.trim()) speak(buffer.trim());
      await queue.drain();
      if (!live()) return;

      setDialogBoth([...dialogRef.current, { role: "coach", text: say }]);

      const shouldSend = Boolean(nextDraft) && (verdict === "ready" || intent === "confirm" || intent === "pick");
      if (intent === "cancel") {
        setDialogBoth([]);
        setDraft("");
      } else if (shouldSend) {
        setState("sending");
        const reply = await optionsRef.current.onSend(nextDraft);
        if (!live()) return;
        setDialogBoth([]);
        setDraft("");
        if (reply?.text) {
          setState("speaking_partner");
          await queue
            .enqueue(api.speak({ text: reply.text, language: "es", voice: optionsRef.current.partnerVoice, role: "partner" }))
            .catch(() => {});
          await queue.drain();
        }
      }
    } catch (err) {
      if (live()) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (live()) {
        busyRef.current = false;
        micRef.current.setMuted(false);
        setState(micRef.current.state === "listening" ? "listening" : "error");
      }
    }
  }

  const start = useCallback(async () => {
    queueRef.current!.unlock();
    setError(null);
    setDialogBoth([]);
    setCoachSay("");
    setDraft("");
    await micRef.current.start();
    setState("listening");
  }, []);

  const stop = useCallback(() => {
    runRef.current += 1;
    busyRef.current = false;
    queueRef.current!.stop();
    micRef.current.stop();
    finalsRef.current = [];
    wordsRef.current = [];
    setState("idle");
  }, []);

  // The mic hook reporting an error (e.g. the track ending) while we're
  // listening is a session-level error too.
  useEffect(() => {
    if (mic.state === "error" && state !== "idle") {
      setError(mic.error);
      setState("error");
    }
  }, [mic.state, mic.error, state]);

  return { state, error, dialog, coachSay, draft, interim: mic.interim, start, stop };
}
