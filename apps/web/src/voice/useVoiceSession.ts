import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { BrainstormStreamEvent, VoiceTranscriptEvent, VoiceTranscriptWord } from "../api/client";
import type { BrainstormAsk } from "../brainstorm/useBrainstorm";
import { AudioQueue } from "./audioQueue";
import { playEarcon } from "./earcons";
import { takeCompleteSentences } from "./sentences";
import { useKeepAwake } from "./useKeepAwake";
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
  // The chat input's normal send path - resolves with the partner's reply
  // once the turn is done, or null if it failed.
  onSend: (text: string) => Promise<{ text: string } | null>;
  partnerVoice?: string;
  // Everything spoken is brainstorming: each finished utterance becomes a
  // Brainstorm turn (useBrainstorm's `stream`), and the coach's answer is
  // spoken back.
  streamBrainstorm: (ask: BrainstormAsk) => AsyncGenerator<BrainstormStreamEvent>;
  // What's in the Reply box right now - sent to the coach as context, and
  // what actually gets sent on a hands-free "send it" (so the learner's own
  // edits to the coach's draft win).
  getReplyDraft: () => string;
  // What the learner has said so far in the current utterance (shown in
  // the Brainstorm field; "" once it's been handed to the coach), and the
  // coach's proposed sentence for the Reply box.
  onTranscript?: (text: string) => void;
  onDraft?: (draft: string) => void;
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
export function useVoiceSession({
  sessionId,
  onSend,
  partnerVoice,
  streamBrainstorm,
  getReplyDraft,
  onTranscript,
  onDraft,
}: UseVoiceSessionOptions) {
  const [state, setState] = useState<VoiceState>("idle");
  const [error, setError] = useState<string | null>(null);
  // True from the moment a "ready" verdict (with a draft) is known until
  // the send happens - lights the Send button green while the coach is
  // still saying "Sending: ...".
  const [readyToSend, setReadyToSend] = useState(false);
  const readyRef = useRef(false);
  const markReady = (next: boolean) => {
    readyRef.current = next;
    setReadyToSend(next);
  };
  // Set by sendNow() while a coach turn is mid-readout: skip whatever the
  // coach still has to say and send right away (optionally a learner-edited
  // version of the draft instead of the coach's own).
  const skipSpeechRef = useRef(false);
  const sendOverrideRef = useRef<string | null>(null);
  const queueRef = useRef<AudioQueue | null>(null);
  if (!queueRef.current) queueRef.current = new AudioQueue();
  // Text settled by the recognizer since the last utterance boundary, plus
  // its per-word confidences - becomes one coach turn at utterance_end.
  const finalsRef = useRef<string[]>([]);
  const wordsRef = useRef<VoiceTranscriptWord[]>([]);
  const busyRef = useRef(false);
  const runRef = useRef(0);
  const optionsRef = useRef({ onSend, partnerVoice, streamBrainstorm, getReplyDraft, onTranscript, onDraft });
  useEffect(() => {
    optionsRef.current = { onSend, partnerVoice, streamBrainstorm, getReplyDraft, onTranscript, onDraft };
  }, [onSend, partnerVoice, streamBrainstorm, getReplyDraft, onTranscript, onDraft]);

  const micEvent = useCallback((event: VoiceTranscriptEvent) => {
    if (event.type === "final") {
      finalsRef.current.push(event.text.trim());
      wordsRef.current.push(...event.words);
      optionsRef.current.onTranscript?.(finalsRef.current.join(" ").trim());
    } else if (event.type === "utterance_end") {
      const transcript = finalsRef.current.join(" ").trim();
      const words = wordsRef.current;
      finalsRef.current = [];
      wordsRef.current = [];
      if (transcript && !busyRef.current) {
        optionsRef.current.onTranscript?.("");
        void handleUtterance(transcript, words);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const mic = useMicStream({ sessionId, onEvent: micEvent });
  const micRef = useRef(mic);
  micRef.current = mic;
  const stopRef = useRef<() => void>(() => {});
  const keepAwake = useKeepAwake({ onPauseRequested: () => stopRef.current() });

  async function handleUtterance(transcript: string, words: VoiceTranscriptWord[]) {
    const queue = queueRef.current!;
    const run = ++runRef.current;
    const live = () => run === runRef.current;
    busyRef.current = true;
    micRef.current.setMuted(true);
    playEarcon(micRef.current.audioContext(), "heard");
    setState("thinking");
    setError(null);
    markReady(false);
    skipSpeechRef.current = false;
    sendOverrideRef.current = null;

    let intent = "attempt";
    let verdict = "revise";
    let nextDraft = "";
    let buffer = "";
    const speak = (sentence: string) => {
      if (!live() || skipSpeechRef.current) return;
      setState("speaking_coach");
      void queue.enqueue(api.speak({ text: sentence, language: "en", role: "coach" })).catch(() => {
        // One failed clip shouldn't end the turn - the text is on screen.
      });
    };

    try {
      for await (const event of optionsRef.current.streamBrainstorm({
        text: transcript,
        words,
        reply_draft: optionsRef.current.getReplyDraft(),
        source: "voice",
      })) {
        if (!live()) return;
        if (event.type === "intent") intent = event.data.intent;
        else if (event.type === "verdict") verdict = event.data.verdict;
        else if (event.type === "draft") {
          nextDraft = event.data.draft;
          if (nextDraft) optionsRef.current.onDraft?.(nextDraft);
          if (nextDraft && (verdict === "ready" || intent === "confirm" || intent === "pick")) markReady(true);
        } else if (event.type === "say_chunk") {
          buffer += event.data.delta;
          const { sentences, rest } = takeCompleteSentences(buffer);
          buffer = rest;
          sentences.forEach(speak);
        } else if (event.type === "say_complete") {
          if (buffer.trim()) speak(buffer.trim());
          buffer = "";
        }
      }
      if (buffer.trim()) speak(buffer.trim());
      await queue.drain();
      if (!live()) return;

      const shouldSend = Boolean(nextDraft) && (verdict === "ready" || intent === "confirm" || intent === "pick");
      if (intent === "cancel") {
        optionsRef.current.onDraft?.("");
      } else if (shouldSend) {
        // Whatever the Reply box holds now - the coach's draft, plus any
        // edits the learner made while it was talking.
        const text = sendOverrideRef.current ?? (optionsRef.current.getReplyDraft().trim() || nextDraft);
        await deliver(text, live);
      }
    } catch (err) {
      if (live()) {
        setError(err instanceof Error ? err.message : String(err));
        playEarcon(micRef.current.audioContext(), "error");
      }
    } finally {
      finishTurn(live);
    }
  }

  // Sends `text` through the app's normal send path and speaks the
  // partner's reply - the tail end of a coach turn that judged the
  // attempt ready, and the whole of a learner-initiated send (sendNow).
  async function deliver(text: string, live: () => boolean) {
    const queue = queueRef.current!;
    setState("sending");
    playEarcon(micRef.current.audioContext(), "sent");
    const reply = await optionsRef.current.onSend(text);
    if (!live()) return;
    markReady(false);
    if (reply?.text) {
      setState("speaking_partner");
      await queue
        .enqueue(api.speak({ text: reply.text, language: "es", voice: optionsRef.current.partnerVoice, role: "partner" }))
        .catch(() => {});
      await queue.drain();
    }
  }

  function finishTurn(live: () => boolean) {
    if (!live()) return;
    busyRef.current = false;
    markReady(false);
    skipSpeechRef.current = false;
    sendOverrideRef.current = null;
    micRef.current.setMuted(false);
    const listening = micRef.current.state === "listening";
    if (listening) playEarcon(micRef.current.audioContext(), "listening");
    setState(listening ? "listening" : "error");
  }

  // The learner tapped Send while a voice chat is on. Mid-turn, that only
  // means something once the coach has a ready draft (readyToSend): skip
  // the rest of its readout and send now. Between turns it just sends the
  // draft as typed/spoken - through this hook rather than the plain text
  // path so the partner's reply still gets spoken.
  const sendNow = useCallback((text: string) => {
    if (busyRef.current) {
      if (!readyRef.current) return;
      sendOverrideRef.current = text.trim() || null;
      skipSpeechRef.current = true;
      queueRef.current!.stop();
      return;
    }
    if (micRef.current.state !== "listening" || !text.trim()) return;
    void (async () => {
      const run = ++runRef.current;
      const live = () => run === runRef.current;
      busyRef.current = true;
      micRef.current.setMuted(true);
      setError(null);
      try {
        await deliver(text.trim(), live);
      } catch (err) {
        if (live()) {
          setError(err instanceof Error ? err.message : String(err));
          playEarcon(micRef.current.audioContext(), "error");
        }
      } finally {
        finishTurn(live);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = useCallback(async () => {
    // Both of these must happen inside the tap itself (before any await):
    // the playback element and the keep-alive loop can only start from a
    // user gesture on iOS.
    queueRef.current!.unlock();
    keepAwake.start();
    setError(null);
    markReady(false);
    await micRef.current.start();
    playEarcon(micRef.current.audioContext(), "listening");
    setState("listening");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stop = useCallback(() => {
    runRef.current += 1;
    busyRef.current = false;
    queueRef.current!.stop();
    keepAwake.stop();
    micRef.current.stop();
    finalsRef.current = [];
    wordsRef.current = [];
    markReady(false);
    setState("idle");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  stopRef.current = stop;

  // The mic hook reporting an error (e.g. the track ending) while we're
  // listening is a session-level error too.
  useEffect(() => {
    if (mic.state === "error" && state !== "idle") {
      setError(mic.error);
      setState("error");
    }
  }, [mic.state, mic.error, state]);

  return {
    state,
    error,
    readyToSend,
    interim: mic.interim,
    keepAwakeMode: keepAwake.mode,
    start,
    stop,
    sendNow,
  };
}
