import type { CoachOption } from "../api/client";
import "./CoachPopover.css";

interface CoachPopoverProps {
  meaning: string;
  feedback: string;
  options: CoachOption[];
  loading: boolean;
  error: boolean;
  // True while the SAME underlying streamed call's second phase (each
  // option's own English translation + word-by-word breakdown) is still
  // in flight - options themselves (and picking one) are already usable
  // before this clears.
  translationsPending: boolean;
  // Which option the learner has picked, if any - stays set (and this
  // popover stays open/rendered) after a pick, so they can compare a
  // different option afterward. Not the same as "closed": only onClose
  // or onRegenerate make the popover go away/refetch.
  selectedIndex: number | null;
  onSelect: (option: CoachOption, index: number) => void;
  onClose: () => void;
  onRegenerate: () => void;
  // True when the model judged the draft basically already correct
  // (empty `feedback`) - shows a compact "looks good, send as typed?"
  // confirmation instead of the full meaning/feedback/options breakdown,
  // since there's nothing to correct and no real choice to make.
  clean: boolean;
  onConfirmSend: () => void;
}

const FORMALITY_LABEL: Record<string, string> = { neutral: "Natural", casual: "Casual", formal: "Formal" };

export function CoachPopover({
  meaning,
  feedback,
  options,
  loading,
  error,
  translationsPending,
  selectedIndex,
  onSelect,
  onClose,
  onRegenerate,
  clean,
  onConfirmSend,
}: CoachPopoverProps) {
  return (
    <div className="coach-popover" role="tooltip">
      <div className="coach-popover__header-buttons">
        <button
          type="button"
          className="coach-popover__regenerate"
          onClick={onRegenerate}
          disabled={loading}
          aria-label="Get new suggestions"
          title="Get new suggestions"
        >
          ↻
        </button>
        <button type="button" className="coach-popover__close" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </div>
      {loading && <div className="coach-popover__status">Thinking…</div>}
      {error && <div className="coach-popover__status coach-popover__status--error">Couldn't get suggestions - try again</div>}
      {!loading && !error && clean && (
        <div className="coach-popover__confirm">
          <span className="coach-popover__confirm-text">
            Looks good. You meant to say <em>"{meaning}"</em>?
          </span>
          <button type="button" className="coach-popover__confirm-send" onClick={onConfirmSend}>
            Send
          </button>
        </div>
      )}
      {!loading && !error && !clean && (
        <>
          {meaning && (
            <div className="coach-popover__row">
              <span className="coach-popover__label">You mean</span>
              {meaning}
            </div>
          )}
          {feedback && <div className="coach-popover__feedback">{feedback}</div>}
          <div className="coach-popover__options">
            {options.map((opt, i) => {
              const selected = selectedIndex === i;
              return (
                <div key={i} className={`coach-popover__option${selected ? " coach-popover__option--selected" : ""}`}>
                  <button type="button" className="coach-popover__option-main" onClick={() => onSelect(opt, i)}>
                    <span className="coach-popover__option-label">{FORMALITY_LABEL[opt.formality] ?? opt.formality}</span>
                    <span className="coach-popover__option-text">{opt.spanish}</span>
                  </button>
                  {selected &&
                    (opt.english ? (
                      <div className="coach-popover__option-english">{opt.english}</div>
                    ) : translationsPending ? (
                      <div className="coach-popover__option-english coach-popover__option-english--loading">Translating…</div>
                    ) : null)}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
