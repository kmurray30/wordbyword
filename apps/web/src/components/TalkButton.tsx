import type { MicState } from "../voice/useMicStream";
import "./TalkButton.css";

// The mic button. While you're talking it turns into a speech bubble with
// moving dots - "you're mid-sentence" - rather than a stop or done button;
// tapping it finishes the sentence.
export function TalkButton({
  state,
  onClick,
  finishLabel,
}: {
  state: MicState;
  onClick: () => void;
  finishLabel: string;
}) {
  const talking = state === "starting" || state === "listening" || state === "finishing";
  return (
    <button
      type="button"
      className={`talk-button${talking ? " talk-button--talking" : ""}${state === "finishing" ? " talk-button--finishing" : ""}`}
      onClick={onClick}
      disabled={state === "starting" || state === "finishing"}
      aria-label={talking ? finishLabel : "Talk instead of typing"}
      title={talking ? finishLabel : "Talk instead of typing - tap again when you've finished"}
    >
      {talking ? (
        <span className="talk-button__bubble" aria-hidden="true">
          <span className="talk-button__dot" />
          <span className="talk-button__dot" />
          <span className="talk-button__dot" />
        </span>
      ) : (
        "🎤"
      )}
    </button>
  );
}
