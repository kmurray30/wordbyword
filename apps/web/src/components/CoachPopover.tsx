import type { CoachOptionState } from "./ChatInput";
import "./CoachPopover.css";

interface CoachPopoverProps {
  phase: "idle" | "loading" | "error";
  // Known the INSTANT the "verdict" event arrives - well before `phase`
  // settles back to "idle" for the rest of the check. null until then.
  verdict: "clean" | "minor" | "fix" | null;
  meaning: string;
  meaningDone: boolean;
  // Only meaningful for a "minor" verdict.
  suggestion: string;
  suggestionDone: boolean;
  // Only meaningful for a "fix" verdict - 3 entries, seeded as empty
  // skeletons the instant the verdict arrives (see ChatInput.tsx's
  // fetchCoach), each filling in independently as its own events stream.
  options: CoachOptionState[];
  // Which option the learner has picked, if any - stays set (and this
  // popover stays open/rendered) after a pick, so they can compare a
  // different option afterward. Not the same as "closed": only onClose
  // or onRegenerate make the popover go away/refetch.
  selectedIndex: number | null;
  onSelect: (option: CoachOptionState, index: number) => void;
  onClose: () => void;
  onRegenerate: () => void;
  // "clean": sends the draft verbatim. "minor": applies the (by-then-
  // complete) suggestion into the draft first, then sends - see
  // ChatInput.tsx's handleConfirmMinorSend.
  onConfirmSend: () => void;
  onConfirmMinorSend: () => void;
}

const FORMALITY_LABEL: Record<string, string> = { neutral: "Natural", casual: "Casual", formal: "Formal" };

export function CoachPopover({
  phase,
  verdict,
  meaning,
  meaningDone,
  suggestion,
  suggestionDone,
  options,
  selectedIndex,
  onSelect,
  onClose,
  onRegenerate,
  onConfirmSend,
  onConfirmMinorSend,
}: CoachPopoverProps) {
  const error = phase === "error";
  // Before the verdict itself is known, there's nothing yet to show but
  // the generic placeholder - once it arrives, the meaning row (framed as
  // a statement or a question depending on the verdict) takes over even
  // if the rest of the check ("loading") is still in flight.
  const waitingForVerdict = phase === "loading" && verdict === null;

  return (
    <div className="coach-popover" role="tooltip">
      <div className="coach-popover__header-buttons">
        <button
          type="button"
          className="coach-popover__regenerate"
          onClick={onRegenerate}
          disabled={phase === "loading"}
          aria-label="Get new suggestions"
          title="Get new suggestions"
        >
          ↻
        </button>
        <button type="button" className="coach-popover__close" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </div>
      {waitingForVerdict && <div className="coach-popover__status">Thinking…</div>}
      {error && <div className="coach-popover__status coach-popover__status--error">Couldn't get suggestions - try again</div>}
      {!error && verdict !== null && (
        <>
          {meaning && (
            <div className="coach-popover__row">
              <span className="coach-popover__label">{verdict === "fix" ? "Did you mean" : "You mean"}</span>
              {meaning}
              {verdict === "fix" ? "?" : ""}
            </div>
          )}

          {verdict === "clean" && meaningDone && (
            <div className="coach-popover__confirm">
              <span className="coach-popover__confirm-text">Looks good - send as typed?</span>
              <button type="button" className="coach-popover__confirm-send" onClick={onConfirmSend}>
                Send
              </button>
            </div>
          )}

          {verdict === "minor" && meaningDone && (
            <div className="coach-popover__minor">
              <div className="coach-popover__suggestion">
                <span className="coach-popover__label">Tiny fix</span>
                {suggestion || "…"}
              </div>
              <button type="button" className="coach-popover__confirm-send" onClick={onConfirmMinorSend} disabled={!suggestionDone}>
                Send
              </button>
            </div>
          )}

          {verdict === "fix" && (
            <div className="coach-popover__options">
              {options.map((opt, i) => {
                const selected = selectedIndex === i;
                const isSkeleton = !opt.spanish && !opt.spanishDone;
                return (
                  <div
                    key={i}
                    className={`coach-popover__option${selected ? " coach-popover__option--selected" : ""}${
                      isSkeleton ? " coach-popover__option--skeleton" : ""
                    }`}
                  >
                    <button
                      type="button"
                      className="coach-popover__option-main"
                      onClick={() => onSelect(opt, i)}
                      disabled={!opt.spanishDone}
                    >
                      <span className="coach-popover__option-label">{FORMALITY_LABEL[opt.formality] ?? opt.formality}</span>
                      <span className="coach-popover__option-text">{opt.spanish || "···"}</span>
                    </button>
                    {opt.spanish && (
                      <div className={`coach-popover__option-english${opt.english ? "" : " coach-popover__option-english--loading"}`}>
                        {opt.english || "…"}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
