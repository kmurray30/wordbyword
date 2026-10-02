import "./CoachPopover.css";

export interface CoachOption {
  formality: string;
  spanish: string;
}

interface CoachPopoverProps {
  meaning: string;
  feedback: string;
  options: CoachOption[];
  loading: boolean;
  error: boolean;
  onSelect: (spanish: string) => void;
}

const FORMALITY_LABEL: Record<string, string> = { neutral: "Natural", casual: "Casual", formal: "Formal" };

// The input's "help" button's popover: unlike the single-candidate draft
// translate preview it replaces, this shows the model's best guess at what
// the learner meant, a short note on how close their attempt already was,
// and a few ways to actually phrase it in Spanish (at different formality
// levels) - clicking one applies it to the draft, same as before.
export function CoachPopover({ meaning, feedback, options, loading, error, onSelect }: CoachPopoverProps) {
  return (
    <div className="coach-popover" role="tooltip">
      {loading && <div className="coach-popover__status">Thinking…</div>}
      {error && <div className="coach-popover__status coach-popover__status--error">Couldn't get suggestions - try again</div>}
      {!loading && !error && (
        <>
          {meaning && (
            <div className="coach-popover__row">
              <span className="coach-popover__label">You mean</span>
              {meaning}
            </div>
          )}
          {feedback && <div className="coach-popover__feedback">{feedback}</div>}
          <div className="coach-popover__options">
            {options.map((opt, i) => (
              <button key={i} type="button" className="coach-popover__option" onClick={() => onSelect(opt.spanish)}>
                <span className="coach-popover__option-label">{FORMALITY_LABEL[opt.formality] ?? opt.formality}</span>
                <span className="coach-popover__option-text">{opt.spanish}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
