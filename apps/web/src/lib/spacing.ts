// Heuristic for re-joining tokenized text with roughly correct spacing:
// no space before closing punctuation, no space after an opening bracket or
// the inverted Spanish "¿"/"¡". Good enough for display; not linguistically
// exhaustive.
const NO_SPACE_BEFORE = /^[.,;:!?)\]}%]/;
const NO_SPACE_AFTER = /^[¿¡([{]$/;

export function joinTokens(surfaces: string[]): { surface: string; spaceBefore: boolean }[] {
  return surfaces.map((surface, i) => {
    if (i === 0) return { surface, spaceBefore: false };
    const prev = surfaces[i - 1];
    const spaceBefore = !NO_SPACE_BEFORE.test(surface) && !NO_SPACE_AFTER.test(prev);
    return { surface, spaceBefore };
  });
}
