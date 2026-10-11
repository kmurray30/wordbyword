// One persistent <audio> element plus a FIFO of clips. iOS only allows
// playback that originates from a user gesture - but an element that was
// played ONCE inside a gesture stays "unlocked", so every later programmatic
// play() on that same element succeeds. unlock() must therefore be called
// synchronously from the tap that starts voice mode; after that the coach
// and partner clips can play whenever they're ready.
//
// Queue items are promises of blobs, not blobs: TTS fetches start the moment
// a sentence is known (several in flight at once) while earlier clips are
// still playing, so the gap between sentences is just the fetch that was
// already running, not a fresh round trip.

// 1 frame of near-silence as a WAV data URI - the "something to play inside
// the gesture" that unlocks the element.
function silentWav(): string {
  const rate = 8000;
  const n = 400;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  w(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, n * 2, true);
  let s = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return "data:audio/wav;base64," + btoa(s);
}

type Item = { blob: Promise<Blob>; resolve: () => void; reject: (err: Error) => void };

export class AudioQueue {
  private el: HTMLAudioElement | null = null;
  private items: Item[] = [];
  private playing = false;
  private generation = 0;
  onStateChange: ((playing: boolean) => void) | null = null;

  // Call from inside a user gesture.
  unlock(): void {
    if (this.el) return;
    const el = new Audio();
    el.setAttribute("playsinline", "");
    el.src = silentWav();
    void el.play().catch(() => {
      // If even this fails we'll find out on the first real clip.
    });
    this.el = el;
    if ("mediaSession" in navigator) {
      navigator.mediaSession.metadata = new MediaMetadata({ title: "wordbyword", artist: "voice conversation" });
    }
  }

  // Resolves once this clip has finished playing (or been skipped by stop()).
  enqueue(blob: Promise<Blob>): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.items.push({ blob, resolve, reject });
      void this.pump();
    });
  }

  // Resolves when everything queued so far has played.
  drain(): Promise<void> {
    if (!this.playing && this.items.length === 0) return Promise.resolve();
    return new Promise<void>((resolve) => {
      this.items.push({ blob: Promise.resolve(new Blob()), resolve, reject: () => resolve() });
      void this.pump();
    });
  }

  stop(): void {
    this.generation += 1;
    const pending = this.items;
    this.items = [];
    pending.forEach((item) => item.resolve());
    if (this.el) {
      this.el.pause();
      this.el.removeAttribute("src");
      this.el.load();
    }
    this.setPlaying(false);
  }

  private setPlaying(next: boolean): void {
    if (this.playing === next) return;
    this.playing = next;
    this.onStateChange?.(next);
  }

  private async pump(): Promise<void> {
    if (this.playing) return;
    const el = this.el;
    if (!el) {
      this.items.splice(0).forEach((item) => item.reject(new Error("audio not unlocked")));
      return;
    }
    const generation = this.generation;
    this.setPlaying(true);
    while (this.items.length > 0 && generation === this.generation) {
      const item = this.items.shift()!;
      try {
        const blob = await item.blob;
        if (generation !== this.generation) {
          item.resolve();
          break;
        }
        if (blob.size === 0) {
          item.resolve();
          continue;
        }
        const url = URL.createObjectURL(blob);
        await new Promise<void>((resolve, reject) => {
          el.onended = () => resolve();
          el.onerror = () => reject(new Error("audio playback failed"));
          el.src = url;
          el.play().catch(reject);
        }).finally(() => URL.revokeObjectURL(url));
        item.resolve();
      } catch (err) {
        item.reject(err instanceof Error ? err : new Error(String(err)));
      }
    }
    this.setPlaying(false);
  }
}
