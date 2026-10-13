import { useSyncExternalStore } from "react";

// Every language the learner can study, and the one they're studying now.
// Each language is its own conversation (its own session - see
// lib/session.ts); switching remounts the app (see main.tsx).

export type LanguageCode = "es" | "fr" | "ja" | "ko" | "zh";

export interface LanguageConfig {
  code: LanguageCode;
  name: string;
  nativeName: string;
  flag: string;
  // The empty chat's nudge.
  greeting: string;
  // Words are separated by spaces (not in Japanese or Chinese).
  spaced: boolean;
  // Written in its own script, with readings available (romanization,
  // furigana) - see ReadingSettings.
  readings: null | {
    // What romanization is called: "Romaji", "Pinyin", "Romanization".
    roman: string;
    // The script itself, as its readers call it.
    script: string;
  };
}

export const LANGUAGES: Record<LanguageCode, LanguageConfig> = {
  es: { code: "es", name: "Spanish", nativeName: "Español", flag: "🇪🇸", greeting: "¡Hola!", spaced: true, readings: null },
  fr: { code: "fr", name: "French", nativeName: "Français", flag: "🇫🇷", greeting: "Bonjour !", spaced: true, readings: null },
  ja: {
    code: "ja",
    name: "Japanese",
    nativeName: "日本語",
    flag: "🇯🇵",
    greeting: "こんにちは！",
    spaced: false,
    readings: { roman: "Romaji", script: "日本語" },
  },
  ko: {
    code: "ko",
    name: "Korean",
    nativeName: "한국어",
    flag: "🇰🇷",
    greeting: "안녕하세요!",
    spaced: true,
    readings: { roman: "Romanized", script: "한글" },
  },
  zh: {
    code: "zh",
    name: "Mandarin",
    nativeName: "中文",
    flag: "🇨🇳",
    greeting: "你好！",
    spaced: false,
    readings: { roman: "Pinyin", script: "汉字" },
  },
};

export const LANGUAGE_CODES = Object.keys(LANGUAGES) as LanguageCode[];

const LANGUAGE_KEY = "wordbyword.lang";

function isLanguageCode(value: string | null): value is LanguageCode {
  return !!value && value in LANGUAGES;
}

function loadLanguage(): LanguageCode {
  try {
    const saved = localStorage.getItem(LANGUAGE_KEY);
    if (isLanguageCode(saved)) return saved;
  } catch {
    // storage blocked - Spanish for this page load
  }
  return "es";
}

let current: LanguageCode = loadLanguage();
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function emit() {
  for (const listener of listeners) listener();
}

export function getLanguage(): LanguageCode {
  return current;
}

export function setLanguage(code: LanguageCode) {
  if (code === current) return;
  current = code;
  try {
    localStorage.setItem(LANGUAGE_KEY, code);
  } catch {
    // per-viewer convenience only
  }
  emit();
}

export function useLanguage(): LanguageCode {
  return useSyncExternalStore(subscribe, getLanguage);
}

// --- How Japanese, Korean and Chinese are shown ---------------------------

export interface ReadingSettings {
  // "roman": romanization only; "both": the script with readings over it
  // (furigana/romaji, pinyin, romanization); "script": the script alone.
  mode: "roman" | "both" | "script";
  // Japanese, with readings: kana over kanji, and romaji under.
  furigana: boolean;
  romaji: boolean;
  // Japanese: kanji, or kana in its place.
  kanji: boolean;
  // Mandarin: which characters.
  chineseScript: "simplified" | "traditional";
}

const DEFAULT_SETTINGS: ReadingSettings = {
  mode: "both",
  furigana: true,
  romaji: false,
  kanji: true,
  chineseScript: "simplified",
};

const settingsKey = (code: LanguageCode) => `wordbyword.reading.${code}`;
const settingsCache = new Map<LanguageCode, ReadingSettings>();

export function getReadingSettings(code: LanguageCode = current): ReadingSettings {
  let settings = settingsCache.get(code);
  if (!settings) {
    settings = DEFAULT_SETTINGS;
    try {
      const saved = localStorage.getItem(settingsKey(code));
      if (saved) settings = { ...DEFAULT_SETTINGS, ...(JSON.parse(saved) as Partial<ReadingSettings>) };
    } catch {
      // unreadable - defaults
    }
    settingsCache.set(code, settings);
  }
  return settings;
}

export function setReadingSettings(patch: Partial<ReadingSettings>, code: LanguageCode = current) {
  const next = { ...getReadingSettings(code), ...patch };
  settingsCache.set(code, next);
  try {
    localStorage.setItem(settingsKey(code), JSON.stringify(next));
  } catch {
    // per-viewer convenience only
  }
  emit();
}

export function useReadingSettings(): ReadingSettings {
  return useSyncExternalStore(subscribe, () => getReadingSettings(current));
}

// What the server is told the language is: Mandarin carries the learner's
// characters ("zh-Hant" = Traditional), so replies are written in them.
export function apiLanguage(code: LanguageCode = current): string {
  if (code !== "zh") return code;
  return getReadingSettings("zh").chineseScript === "traditional" ? "zh-Hant" : "zh-Hans";
}

// For the lang="" attribute on text in the language being learned (picks
// the right glyphs for Chinese characters, Japanese vs Chinese).
export function htmlLang(code: LanguageCode = current): string {
  if (code !== "zh") return code;
  return getReadingSettings("zh").chineseScript === "traditional" ? "zh-Hant" : "zh-Hans";
}
