// AudioWorklet processor: turns the microphone's float samples (at whatever
// rate the AudioContext runs - 48kHz on iOS) into 16kHz mono 16-bit PCM
// chunks, posted to the main thread as transferable ArrayBuffers of ~100ms
// each. Served as a plain static file (not bundled) because
// audioWorklet.addModule() needs a URL, and it must stay dependency-free.
//
// Deliberately timer-free: it emits purely on the audio clock, which keeps
// running on iOS while the screen is off as long as the audio session is
// alive (timers don't - see useMicStream.ts).
const TARGET_RATE = 16000;
const CHUNK_SAMPLES = 1600; // 100ms at 16kHz

class PcmDownsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.phase = 0; // fractional read position into the current block
    this.out = new Int16Array(CHUNK_SAMPLES);
    this.outLen = 0;
    this.prev = 0; // last sample of the previous block, for interpolation
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    const n = channel.length;
    let pos = this.phase;
    while (pos < n) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const a = i === 0 ? this.prev : channel[i - 1];
      const b = channel[i];
      const sample = a + (b - a) * frac;
      const clamped = Math.max(-1, Math.min(1, sample));
      this.out[this.outLen++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      if (this.outLen === CHUNK_SAMPLES) {
        this.port.postMessage(this.out.buffer, [this.out.buffer]);
        this.out = new Int16Array(CHUNK_SAMPLES);
        this.outLen = 0;
      }
      pos += this.ratio;
    }
    this.phase = pos - n;
    this.prev = channel[n - 1];
    return true;
  }
}

registerProcessor("pcm-downsampler", PcmDownsampler);
