// Splits streamed coach text into complete sentences as they arrive, so
// each one can be sent to TTS (and start playing) before the model has
// finished the whole reply. Anything after the last sentence boundary is
// returned as `rest` to be prepended to the next chunk.
const BOUNDARY = /([.!?…]+["'”’)\]]?)(\s+|$)/g;

// A boundary inside an open quotation is not a sentence end for TTS
// purposes: the coach reads Spanish in quoted phrases ("Say: “Hola, ¿cómo
// estás? ¿Te gusta correr?”"), and splitting at the inner "?" would hand
// the voice two half-phrases (observed live).
function insideOpenQuote(text: string): boolean {
  let curly = 0;
  let straight = 0;
  let single = 0;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "“" || ch === "‘") curly += 1;
    else if (ch === "”" || ch === "’") curly = Math.max(0, curly - 1);
    else if (ch === '"') straight += 1;
    else if (ch === "'") {
      // A straight single quote is an apostrophe inside a word (don't)
      // unless it sits at a word edge: opening after a space/colon/bracket,
      // closing before space/punctuation/end.
      const before = i === 0 ? " " : text[i - 1];
      const after = i === text.length - 1 ? " " : text[i + 1];
      if (/[\s:([]/.test(before)) single += 1;
      else if (/[\s.,;:!?)\]]/.test(after)) single = Math.max(0, single - 1);
    }
  }
  return curly > 0 || straight % 2 === 1 || single > 0;
}

export function takeCompleteSentences(buffer: string): { sentences: string[]; rest: string } {
  const sentences: string[] = [];
  let cursor = 0;
  for (const match of buffer.matchAll(BOUNDARY)) {
    // A boundary at the very end of the buffer might still be followed by
    // more of the same sentence in the next chunk (e.g. "3." then "5") -
    // only split on it once whitespace has confirmed it.
    if (match[2] === "" && match.index! + match[0].length === buffer.length) break;
    const end = match.index! + match[0].length;
    if (insideOpenQuote(buffer.slice(cursor, end))) continue;
    // A quoted question/exclamation that the sentence carries on past
    // ("Try “¿Qué tal?” instead.") isn't a boundary either - the next word
    // starting lowercase gives it away.
    if (/["'”’)\]]$/.test(match[1]) && /^\p{Ll}/u.test(buffer.slice(end))) continue;
    const sentence = buffer.slice(cursor, end).trim();
    if (sentence) sentences.push(sentence);
    cursor = end;
  }
  return { sentences, rest: buffer.slice(cursor) };
}
