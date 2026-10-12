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
  for (const ch of text) {
    if (ch === "“") curly += 1;
    else if (ch === "”") curly = Math.max(0, curly - 1);
    else if (ch === '"') straight += 1;
  }
  return curly > 0 || straight % 2 === 1;
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
    const sentence = buffer.slice(cursor, end).trim();
    if (sentence) sentences.push(sentence);
    cursor = end;
  }
  return { sentences, rest: buffer.slice(cursor) };
}
