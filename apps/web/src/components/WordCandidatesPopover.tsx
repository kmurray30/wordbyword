import { CandidateCycler } from "./CandidateCycler";
import "./WordCandidatesPopover.css";

export interface WordCandidate {
  translation: string;
  description?: string;
}

interface WordCandidatesPopoverProps {
  candidates: WordCandidate[];
  // False when there's nothing to replace (the span is already natural,
  // correct Spanish as typed) - just its in-context gloss. True when it
  // offers a Spanish replacement to swap in.
  clickable: boolean;
  onSelect: (translation: string) => void;
  // True while the real (LLM-backed) gloss for this word/group is still
  // being fetched - candidates is meaningless until this clears.
  loading?: boolean;
  // Non-empty only for a standalone word that's ALSO a legitimate,
  // different word in the other language (see DraftSpan.alternate_gloss in
  // app/schemas.py) - e.g. "once" (Spanish "eleven", also an English
  // word). Showing it is a toggle, not a second set of candidates, since
  // it's rare and should stay low-profile.
  alternateGloss?: string;
  showAlternate?: boolean;
  onToggleAlternate?: () => void;
  // Non-empty only for a multi-word span (see DraftSpan.literal) - a short
  // word-by-word breakdown of how the group's individual words combine,
  // shown as a small secondary line so the learner can see how the phrase
  // is built, not just its overall gloss.
  literal?: string;
}

// The one popover widget for hovering/tapping a word or group in the chat
// input, and for highlighting a phrase. No Spanish/English toggle for the
// common case - the backend's LLM call already has full sentence context,
// so it picks the one sense that applies here rather than needing two
// independent readings reconciled in the UI. A span with multiple senses
// still steps through them via CandidateCycler. The alternate-reading
// toggle below is a separate, much rarer case (a true cross-language
// cognate), not the old blanket ES/EN column switch.
export function WordCandidatesPopover({
  candidates,
  clickable,
  onSelect,
  loading,
  alternateGloss,
  showAlternate,
  onToggleAlternate,
  literal,
}: WordCandidatesPopoverProps) {
  if (loading) {
    return (
      <div className="word-candidates-popover" role="tooltip">
        <div className="word-candidates-popover__loading">Translating…</div>
      </div>
    );
  }
  if (candidates.length === 0 && !alternateGloss) return null;

  const displayCandidates = showAlternate && alternateGloss ? [{ translation: alternateGloss }] : candidates;

  return (
    <div className="word-candidates-popover" role="tooltip">
      <div className="word-candidates-popover__main">
        <CandidateCycler candidates={displayCandidates} clickable={!showAlternate && clickable} onSelect={onSelect} />
        {literal && !showAlternate && <div className="word-candidates-popover__literal">{literal}</div>}
      </div>
      {alternateGloss && (
        <button
          type="button"
          className="word-candidates-popover__alt-toggle"
          onClick={(e) => {
            e.stopPropagation();
            onToggleAlternate?.();
          }}
          title={showAlternate ? "Show the Spanish reading" : "Also an English word - show that reading"}
        >
          {showAlternate ? "ES" : "EN"}
        </button>
      )}
    </div>
  );
}
