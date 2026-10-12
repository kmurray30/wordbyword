// Short synthesized cues for the moments the learner can't see: with the
// screen off these ARE the UI. Generated with an oscillator on the mic's
// own AudioContext (already unlocked by the start gesture) rather than
// static audio files - nothing to load, nothing to fetch.
export type Earcon = "listening" | "heard" | "sent" | "error";

const PATTERNS: Record<Earcon, { freqs: number[]; noteMs: number; gain: number }> = {
  listening: { freqs: [523, 659], noteMs: 90, gain: 0.08 },
  heard: { freqs: [880], noteMs: 100, gain: 0.07 },
  sent: { freqs: [659, 523], noteMs: 90, gain: 0.08 },
  error: { freqs: [220, 220], noteMs: 160, gain: 0.09 },
};

export function playEarcon(ctx: AudioContext | null, kind: Earcon): void {
  if (!ctx || ctx.state !== "running") return;
  const { freqs, noteMs, gain } = PATTERNS[kind];
  const noteSeconds = noteMs / 1000;
  const gapSeconds = 0.03;
  let t = ctx.currentTime + 0.01;
  for (const freq of freqs) {
    const osc = ctx.createOscillator();
    const env = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(gain, t + 0.01);
    env.gain.linearRampToValueAtTime(0, t + noteSeconds);
    osc.connect(env);
    env.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + noteSeconds + 0.01);
    t += noteSeconds + gapSeconds;
  }
}
