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
//
// iOS can also END the mic track or drop the socket while the page is
// hidden (a home-screen app being locked, a long detour to another app).
// Neither is treated as fatal while hidden: it's remembered, and on the
// page becoming visible again the mic and/or socket are re-acquired in
// place - the documented WebKit workaround - so the session resumes rather
// than dying. Only a failure to re-acquire surfaces as an error.
export function useMicStream({ sessionId, onEvent }: UseMicStreamOptions) {
  const [state, setState] = useState<MicState>("idle");
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const mutedRef = useRef(false);
  const needsReconnectRef = useRef(false);
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
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
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    void ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
    mutedRef.current = false;
    needsReconnectRef.current = false;
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

  const reconnect = useCallback(async () => {
    needsReconnectRef.current = false;
    const ctx = ctxRef.current;
    const node = nodeRef.current;
    if (!ctx || !node) return;
    try {
      if (ctx.state !== "running") await ctx.resume();
      const track = streamRef.current?.getAudioTracks()[0];
      if (!track || track.readyState === "ended") await openMic(ctx, node);
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        try {
          ws?.close();
        } catch {
          // ignore
        }
        await openSocket();
      }
      setState("listening");
    } catch {
      fail("Microphone was interrupted - tap to listen again");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fail]);

  async function openMic(ctx: AudioContext, node: AudioWorkletNode): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    if (ctxRef.current !== ctx) {
      // stop() raced the permission prompt
      stream.getTracks().forEach((t) => t.stop());
      throw new Error("stopped");
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    sourceRef.current?.disconnect();
    streamRef.current = stream;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(node);
    sourceRef.current = source;
    stream.getAudioTracks()[0]?.addEventListener("ended", () => {
      if (streamRef.current !== stream) return;
      if (document.hidden) {
        needsReconnectRef.current = true;
        return;
      }
      void reconnect();
    });
  }

  function openSocket(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(voiceListenUrl());
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "start", session_id: sessionIdRef.current }));
        if (mutedRef.current) ws.send(JSON.stringify({ type: "mute" }));
        resolve();
      };
      ws.onerror = () => reject(new Error("Couldn't reach the voice server"));
      ws.onclose = () => {
        if (wsRef.current !== ws) return;
        if (document.hidden) {
          needsReconnectRef.current = true;
          return;
        }
        fail("Voice connection closed");
      };
      ws.onmessage = (msg) => {
        if (wsRef.current !== ws) return;
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
    });
  }

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
      await ctx.resume();
      await ctx.audioWorklet.addModule("/pcm-worklet.js");
      if (ctxRef.current !== ctx) return;
      const node = new AudioWorkletNode(ctx, "pcm-downsampler", { numberOfOutputs: 0 });
      nodeRef.current = node;
      // Reads wsRef at send time, so a socket re-opened by reconnect() is
      // picked up without rewiring the worklet.
      node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        if (mutedRef.current) return;
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
      };
      await openMic(ctx, node);
      await openSocket();
      if (ctxRef.current !== ctx) return;
      setState("listening");
    } catch (err) {
      if (ctxRef.current === ctx) fail(err instanceof Error ? err.message : String(err));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fail]);

  // Half-duplex: while the app's own audio plays, don't feed the mic (it
  // would just hear the speaker). Dropped client-side AND told to the
  // server, which stops forwarding to the provider too.
  const setMuted = useCallback((muted: boolean) => {
    mutedRef.current = muted;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: muted ? "mute" : "unmute" }));
  }, []);

  useEffect(() => {
    const onVisibility = () => {
      if (!document.hidden && needsReconnectRef.current && ctxRef.current) void reconnect();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [reconnect]);

  useEffect(() => teardown, [teardown]);

  const audioContext = useCallback(() => ctxRef.current, []);

  return { state, interim, error, start, stop, setMuted, audioContext };
}
