import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { TextReadings } from "../api/client";
import { LANGUAGES, apiLanguage, useLanguage, useReadingSettings } from "./language";

// Readings (romanization, furigana, Mandarin in the learner's characters)
// for text in the language being learned - fetched from the server, which
// computes them deterministically. Requests made in the same tick go out as
// one; each text is fetched once per language.

const cache = new Map<string, Promise<TextReadings>>();
const resolved = new Map<string, TextReadings>();
type Words = [number, number][];
let queue: {
  key: string;
  text: string;
  words: Words | null;
  lang: string;
  resolve: (r: TextReadings) => void;
  reject: (e: unknown) => void;
}[] = [];
let scheduled = false;

const cacheKey = (lang: string, text: string, words: Words | null) =>
  `${lang}\u0000${text}\u0000${words ? words.map((w) => w.join("-")).join(",") : ""}`;

function flush() {
  scheduled = false;
  const batch = queue;
  queue = [];
  const byLang = new Map<string, typeof batch>();
  for (const item of batch) byLang.set(item.lang, [...(byLang.get(item.lang) ?? []), item]);
  for (const [lang, items] of byLang) {
    for (let i = 0; i < items.length; i += 100) {
      const chunk = items.slice(i, i + 100);
      api
        .readings({ texts: chunk.map((item) => item.text), words: chunk.map((item) => item.words), language: lang })
        .then((res) =>
          chunk.forEach((item, j) => {
            const value = res.items[j] ?? { text: item.text, readings: [] };
            resolved.set(item.key, value);
            item.resolve(value);
          }),
        )
        .catch((err) => chunk.forEach((item) => item.reject(err)));
    }
  }
}

// `words` (the text's word boundaries, where known) let Japanese readings
// split at them - particles read as particles.
export function fetchReadings(text: string, lang: string = apiLanguage(), words: Words | null = null): Promise<TextReadings> {
  const key = cacheKey(lang, text, words);
  let promise = cache.get(key);
  if (!promise) {
    promise = new Promise<TextReadings>((resolve, reject) => {
      queue.push({ key, text, words, lang, resolve, reject });
      if (!scheduled) {
        scheduled = true;
        setTimeout(flush, 0);
      }
    });
    cache.set(key, promise);
    promise.catch(() => cache.delete(key));
  }
  return promise;
}

// `text`'s readings once they're in (null before, and always for a
// language written in the Latin alphabet).
export function useReadings(text: string | null | undefined, words: Words | null = null): TextReadings | null {
  const lang = useLanguage();
  useReadingSettings(); // Mandarin's characters are part of apiLanguage()
  const wanted = !!text && !!LANGUAGES[lang].readings;
  const key = wanted ? cacheKey(apiLanguage(lang), text, words) : "";
  const [state, setState] = useState<{ key: string; text: string; value: TextReadings } | null>(null);
  useEffect(() => {
    if (!wanted || resolved.has(key)) return;
    let live = true;
    fetchReadings(text, apiLanguage(lang), words)
      .then((value) => live && setState({ key, text, value }))
      .catch(() => {});
    return () => {
      live = false;
    };
    // `key` covers `words` (a new array each render).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, wanted, text, lang]);
  if (!wanted) return null;
  // While a refined request (same text, words now known) is out, the last
  // readings for the text stay up rather than flashing the bare script.
  return resolved.get(key) ?? (state && (state.key === key || state.text === text) ? state.value : null);
}
