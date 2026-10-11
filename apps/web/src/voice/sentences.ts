// Splits streamed coach text into complete sentences as they arrive, so
// each one can be sent to TTS (and start playing) before the model has
// finished the whole reply. Anything after the last sentence boundary is
// returned as `rest` to be prepended to the next chunk.
const BOUNDARY = /([.!?…]+["')\]]?)(\s+|$)/g;

export function takeCompleteSentences(buffer: string): { sentences: string[]; rest: string } {
  const sentences: string[] = [];
  let cursor = 0;
  for (const match of buffer.matchAll(BOUNDARY)) {
    // A boundary at the very end of the buffer might still be followed by
    // more of the same sentence in the next chunk (e.g. "3." then "5") -
    // only split on it once whitespace has confirmed it.
    if (match[2] === "" && match.index! + match[0].length === buffer.length) break;
    const end = match.index! + match[0].length;
    const sentence = buffer.slice(cursor, end).trim();
    if (sentence) sentences.push(sentence);
    cursor = end;
  }
  return { sentences, rest: buffer.slice(cursor) };
}
