import type { ReactNode } from "react";
import type { Reading } from "../api/client";
import { LANGUAGES, htmlLang, useLanguage, useReadingSettings } from "../lib/language";
import { useReadings } from "../lib/readings";
import "./ReadingText.css";

// Text in the language being learned, shown the way the learner set it up
// (☰ menu): for Japanese, Korean and Chinese - romanization only, the
// script with readings over it (furigana and/or romaji, pinyin, Korean
// romanization), or the script alone; Mandarin in Simplified or
// Traditional characters. Spanish and French show as written.
//
// `words` are the ranges the caller makes interactive (tappable words);
// `renderPiece` draws one piece of one of them (a word can be split in
// pieces when a reading boundary falls inside it - every piece gets the
// same handlers, `first` says whether it's the word's first).

export interface WordRange {
  start: number;
  end: number;
}

export interface Piece {
  start: number;
  end: number;
  // Index into `words`, or -1 for text between words.
  word: number;
}

interface Unit {
  start: number;
  end: number;
  reading: Reading | null;
  pieces: Piece[];
  // Just whitespace.
  space: boolean;
}

// Splits the text at every word, reading and whitespace boundary, then
// groups the pieces each reading covers into one unit (one ruby).
function buildUnits(text: string, words: WordRange[], readings: Reading[]): Unit[] {
  const cuts = new Set<number>([0, text.length]);
  for (const w of words) cuts.add(w.start).add(w.end);
  for (const r of readings) cuts.add(r.start).add(r.end);
  for (const m of text.matchAll(/\s+/g)) cuts.add(m.index ?? 0).add((m.index ?? 0) + m[0].length);
  const points = [...cuts].filter((p) => p >= 0 && p <= text.length).sort((a, b) => a - b);

  const units: Unit[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const [a, b] = [points[i], points[i + 1]];
    if (a === b) continue;
    const word = words.findIndex((w) => w.start <= a && a < w.end);
    const reading = readings.find((r) => r.start <= a && a < r.end) ?? null;
    const last = units[units.length - 1];
    if (reading && last && last.reading === reading) {
      const lastPiece = last.pieces[last.pieces.length - 1];
      if (lastPiece.word === word) lastPiece.end = b;
      else last.pieces.push({ start: a, end: b, word });
      last.end = b;
      continue;
    }
    units.push({ start: a, end: b, reading, pieces: [{ start: a, end: b, word }], space: !text.slice(a, b).trim() });
  }
  return units;
}

const plain = (_piece: Piece, content: ReactNode) => content;

function segmentWords(text: string, locale: string): WordRange[] {
  if (typeof Intl.Segmenter !== "function") return [];
  return [...new Intl.Segmenter(locale, { granularity: "word" }).segment(text)]
    .filter((seg) => seg.isWordLike)
    .map((seg) => ({ start: seg.index, end: seg.index + seg.segment.length }));
}

// Full-width punctuation reads oddly in romanized text: 。 -> ". " etc.
const ROMAN_PUNCTUATION: Record<string, string> = {
  "。": ". ", "、": ", ", "，": ", ", "！": "! ", "？": "? ", "：": ": ", "；": "; ",
  "「": "“", "」": "”", "『": "“", "』": "”", "（": " (", "）": ") ", "　": " ", "・": " ",
};
function romanPunctuation(text: string): string | null {
  if (![...text].some((c) => c in ROMAN_PUNCTUATION)) return null;
  return [...text].map((c) => ROMAN_PUNCTUATION[c] ?? c).join("");
}

export function ReadingText({
  text,
  words = [],
  renderPiece = plain,
}: {
  text: string;
  words?: WordRange[];
  renderPiece?: (piece: Piece, content: ReactNode, first: boolean) => ReactNode;
}) {
  const lang = useLanguage();
  const config = LANGUAGES[lang];
  const settings = useReadingSettings();
  const loaded = useReadings(config.readings ? text : null);
  const shown = loaded && loaded.text.length === text.length ? loaded.text : text;
  const readings = loaded?.readings ?? [];
  // With no words given (a coach suggestion), the browser's segmenter
  // finds them - so a Chinese word's pinyin still reads as one.
  const units = buildUnits(text, words.length || lang !== "zh" ? words : segmentWords(text, htmlLang(lang)), readings);

  const seen = new Set<number>();
  const piece = (p: Piece, content: ReactNode, key: number) => {
    const first = p.word !== -1 && !seen.has(p.word);
    if (p.word !== -1) seen.add(p.word);
    return <span key={key}>{renderPiece(p, content, first)}</span>;
  };
  // A unit drawn as its pieces, or (romanized, kana for kanji) as one text
  // on its first piece - the others then still count as shown.
  const body = (u: Unit, replacement: string | null) =>
    replacement === null
      ? u.pieces.map((p, i) => piece(p, shown.slice(p.start, p.end), i))
      : u.pieces.map((p, i) => piece(p, i === 0 ? replacement : "", i));

  const japanese = lang === "ja";
  const kanaFor = (u: Unit) => (japanese && !settings.kanji && u.reading?.kana ? u.reading.kana : null);

  let content: ReactNode;
  if (!config.readings || !loaded) {
    content = units.map((u, i) => <span key={i}>{body(u, null)}</span>);
  } else if (settings.mode === "roman") {
    let prev: Unit | null = null;
    content = units.map((u, i) => {
      const r = u.reading;
      // Romanized words are spaced apart (Japanese and Chinese have no
      // spaces to keep) - but not the syllables of one Chinese word (its
      // readings are per character).
      const lastWord = prev?.pieces[prev.pieces.length - 1].word ?? -1;
      const sameWord = lang === "zh" && lastWord !== -1 && lastWord === u.pieces[0].word;
      const gap = !!r && !!prev?.reading && prev.end === u.start && !r.cont && !sameWord;
      prev = u;
      return (
        <span key={i}>
          {gap ? " " : null}
          {body(u, r ? r.roman : romanPunctuation(shown.slice(u.start, u.end)))}
        </span>
      );
    });
  } else if (settings.mode === "script" || (japanese && !settings.furigana && !settings.romaji)) {
    content = units.map((u, i) => <span key={i}>{body(u, kanaFor(u))}</span>);
  } else {
    // The script with readings: every unit is a column (reading over it,
    // romaji under it for Japanese), lined up at the top.
    const top = japanese ? settings.furigana && settings.kanji : true;
    const bottom = japanese && settings.romaji;
    // Chinese readings are per character: a word's characters share one
    // column, its pinyin written together (喜欢 under "xǐhuān").
    const columns: Unit[][] = [];
    for (const u of units) {
      const prevColumn = columns[columns.length - 1];
      const last = prevColumn?.[prevColumn.length - 1];
      const word = u.pieces[0].word;
      const joins =
        lang === "zh" && !!last?.reading && !!u.reading && word !== -1 &&
        last.pieces[last.pieces.length - 1].word === word && last.end === u.start;
      if (joins) prevColumn.push(u);
      else columns.push([u]);
    }
    content = columns.map((column, i) => {
      if (column.length === 1 && column[0].space) return <span key={i}>{body(column[0], null)}</span>;
      const over = column.map((u) => (japanese ? u.reading?.kana : u.reading?.roman) ?? "").join("");
      const under = column.map((u) => u.reading?.roman ?? "").join("");
      return (
        <span key={i} className="reading">
          {top && <span className="reading__over">{over || "\u00a0"}</span>}
          <span className="reading__base">
            {column.map((u, j) => (
              <span key={j}>{body(u, kanaFor(u))}</span>
            ))}
          </span>
          {bottom && <span className="reading__under">{under || "\u00a0"}</span>}
        </span>
      );
    });
  }
  return (
    <span lang={htmlLang(lang)} className={`reading-text${loaded && settings.mode === "both" ? " reading-text--both" : ""}`}>
      {content}
    </span>
  );
}
