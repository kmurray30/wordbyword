import { useCallback, useEffect, useRef, useState } from "react";
import { voiceListenUrl } from "../api/client";
import type { VoiceTranscriptEvent } from "../api/client";

export type MicState = "idle" | "starting" | "listening" | "error";

interface UseMicStreamOptions {
  sessionId: string;
  onEvent: (event: VoiceTranscriptEvent) => void;
}

// Streams the microphone to WS /voice/listen as 16kHz PCM (via
// public/pcm-worklet.js) and hands back the transcript events the server
// relays from Deepgram. Owns the whole mic/AudioContext/WebSocket lifecycle;
// callers just start()/stop() and read `state`/`interim`.
//
// Everything that has to keep running with the phone's screen off is driven
// by audio-worklet and WebSocket events - never timers, which iOS throttles
// hard when locked while still delivering audio frames as long as the audio
// session is live.
export function useMicStream({ sessionId, onEvent }: UseMicStreamOptions) {
  const [state, setState] = useState<MicState>("idle");
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const mutedRef = useRef(false);
  // Always call the LATEST onEvent, even though the socket handlers were
  // bound once at start() - otherwise a stale closure would see stale
  // draft state in the parent.
  const onEventRef = useRef(onEvent);
  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  const teardown = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "stop" }));
      } catch {
        // already closing
      }
    }
    try {
      ws?.close();
    } catch {
      // ignore
    }
    nodeRef.current?.port.close();
    nodeRef.current?.disconnect();
    nodeRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    void ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
    mutedRef.current = false;
    setInterim("");
  }, []);

  const stop = useCallback(() => {
    teardown();
    setState("idle");
  }, [teardown]);

  const fail = useCallback(
    (message: string) => {
      teardown();
      setError(message);
      setState("error");
    },
    [teardown],
  );

  const start = useCallback(async () => {
    if (ctxRef.current) return;
    setError(null);
    setState("starting");
    // Created synchronously inside the click/tap that called start() - iOS
    // only lets an AudioContext start from a user gesture, and an `await`
    // before this point would lose that.
    const ctx = new AudioContext();
    ctxRef.current = ctx;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      if (ctxRef.current !== ctx) {
        // stop() raced the permission prompt
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
      stream.getAudioTracks()[0]?.addEventListener("ended", () => {
        // iOS ends the track itself when the page loses the mic (e.g. a
        // home-screen app being backgrounded) - surface it rather than
        // silently streaming nothing.
        if (streamRef.current === stream) fail("Microphone was interrupted - tap to listen again");
      });

      await ctx.resume();
      await ctx.audioWorklet.addModule("/pcm-worklet.js");
      const node = new AudioWorkletNode(ctx, "pcm-downsampler", { numberOfOutputs: 0 });
      nodeRef.current = node;
      ctx.createMediaStreamSource(stream).connect(node);

      const ws = new WebSocket(voiceListenUrl());
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error("Couldn't reach the voice server"));
      });
      if (wsRef.current !== ws) return;
      ws.send(JSON.stringify({ type: "start", session_id: sessionId }));
      ws.onerror = () => fail("Voice connection failed");
      ws.onclose = () => {
        if (wsRef.current === ws) fail("Voice connection closed");
      };
      ws.onmessage = (msg) => {
        let event: VoiceTranscriptEvent;
        try {
          event = JSON.parse(msg.data as string) as VoiceTranscriptEvent;
        } catch {
          return;
        }
        if (event.type === "error") {
          fail(event.message);
          return;
        }
        if (event.type === "interim") setInterim(event.text);
        else if (event.type === "final" || event.type === "utterance_end") setInterim("");
        onEventRef.current(event);
      };
      node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        if (mutedRef.current) return;
        if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
      };
      setState("listening");
    } catch (err) {
      fail(err instanceof Error ? err.message : String(err));
    }
  }, [fail, sessionId]);

  // Half-duplex: while the app's own audio plays, don't feed the mic (it
  // would just hear the speaker). Dropped client-side AND told to the
  // server, which stops forwarding to the provider too.
  const setMuted = useCallback((muted: boolean) => {
    mutedRef.current = muted;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: muted ? "mute" : "unmute" }));
  }, []);

  useEffect(() => teardown, [teardown]);

  return { state, interim, error, start, stop, setMuted };
}
