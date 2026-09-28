// Flags words in a learner's raw typed message that differ from the
// LLM-corrected version of it, so the chat bubble can underline them in
// place - a lightweight "here's what looked off" cue, not a full spell
// checker. Two corrected versions exist (an English restatement and a
// Spanish translation - see llm_translate.interpret_user_input), and the
// raw text might be mostly in either language, so the caller picks
// whichever one shares more words with the raw text before diffing
// against it - diffing English input against a from-scratch Spanish
// translation would flag nearly every word as "wrong", which isn't the
// point.

export interface Segment {
  text: string;
  isWord: boolean;
}

// Splits into alternating word / non-word (whitespace, punctuation) runs.
// Rejoining every segment's `text` in order exactly reproduces the input.
const WORD_RE = /[\p{L}\p{N}]+/gu;

export function tokenizeSegments(text: string): Segment[] {
  const segments: Segment[] = [];
  let cursor = 0;
  for (const match of text.matchAll(WORD_RE)) {
    const start = match.index ?? 0;
    if (start > cursor) segments.push({ text: text.slice(cursor, start), isWord: false });
    segments.push({ text: match[0], isWord: true });
    cursor = start + match[0].length;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), isWord: false });
  return segments;
}

// Case- and accent-insensitive, so "esta"/"está" compare equal - an
// accent-only difference isn't the kind of mistake this is meant to flag.
export function normalizeWord(word: string): string {
  return word.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

function wordOverlapScore(rawWords: string[], candidateWords: string[]): number {
  if (rawWords.length === 0) return 0;
  const candidateSet = new Set(candidateWords.map(normalizeWord));
  const matches = rawWords.filter((w) => candidateSet.has(normalizeWord(w))).length;
  return matches / rawWords.length;
}

// Of the two corrected versions, picks whichever one shares more words with
// the raw input - a same-language correction should overlap heavily; a
// translation into the other language mostly won't.
export function pickDiffReference(rawWords: string[], nativeWords: string[], targetWords: string[]): string[] {
  return wordOverlapScore(rawWords, nativeWords) >= wordOverlapScore(rawWords, targetWords)
    ? nativeWords
    : targetWords;
}

// Standard word-level LCS diff, reported from the raw side only: for each
// raw word, whether it's part of the longest common subsequence with the
// corrected words (matched - leave alone) or not (flag it). Corrected-only
// insertions aren't reported; there's nothing to annotate them onto in the
// raw text.
export function diffRawWords(rawWords: string[], correctedWords: string[]): boolean[] {
  const n = rawWords.length;
  const m = correctedWords.length;
  const rawNorm = rawWords.map(normalizeWord);
  const corrNorm = correctedWords.map(normalizeWord);

  const lcs: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] =
        rawNorm[i] === corrNorm[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }

  const matched: boolean[] = new Array(n).fill(false);
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (rawNorm[i] === corrNorm[j]) {
      matched[i] = true;
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      i++;
    } else {
      j++;
    }
  }
  return matched;
}
